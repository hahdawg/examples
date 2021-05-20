import argparse
from collections import deque
import logging
import logging.config as logcfg
import os
import pickle
from pprint import pformat
from typing import Generator, Tuple

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim

LOGGING_INTERVAL = 1000
LOSS_TARGET = 1.0
NUM_ROW = 100_000
RESULTS_PATH = "/tmp/results.parquet"
logger = logging.getLogger(__name__)


def make_data(N: int, K: int = 2) -> Tuple[torch.Tensor]:
    """
    Make some nonlinear regression data.
    """
    X = np.random.randn(N, K)
    fcns = (np.square, np.abs, np.sin, np.cos)
    Z = np.concatenate([f(X) for f in fcns], axis=1)
    Z = (Z - Z.mean(axis=0))/Z.std(axis=0)
    y = Z.sum(axis=1) + LOSS_TARGET*np.random.randn(N)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


def init_logging(filename: str, logger_name: str = "") -> None:
    path = os.path.join("/tmp", filename)
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,  # Need this in order to get log info from other modules
        'formatters': {
            'standard': {
                'format': '%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d]: %(message)s'
            },
        },
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard"
            },
            "file": {
                "level": "INFO",
                "class": "logging.FileHandler",
                "filename": path,
                "formatter": "standard",
                "mode": "a"
            }
        },
        "loggers": {
            logger_name: {"handlers": ["console", "file"], "level": "INFO"}
        }
    }
    logcfg.dictConfig(logging_config)


class FF(nn.Module):
    """
    Standard feed forward network.
    """
    def __init__(self, input_dim: int, width: int, depth: int):
        super().__init__()
        widths = [input_dim] + depth*[width]
        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_in, n_out) for (n_in, n_out) in zip(widths[:-1], widths[1:])
        ])
        self.output_layer = nn.Linear(width, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
        return self.output_layer(x).flatten()


def batch_generator(X, y, batch_size, num_epochs) -> Generator:
    """
    Standard batch generator.
    """
    for _ in range(num_epochs):
        for i in range(0, X.shape[0], batch_size):
            Xb = X[i:i + batch_size]
            yb = y[i:i + batch_size]
            yield Xb, yb


def fit(
    params: dict,
    batch_size: int,
    num_epochs: int,
    X_tr: torch.Tensor,
    X_val: torch.Tensor,
    y_tr: torch.Tensor,
    y_val: torch.Tensor
) -> None:

    # NOTE: Initialize logger in function that'll be distributed
    logger_name = f"{__name__}.fit"
    init_logging(filename="fit.log", logger_name=logger_name)
    logger_worker = logging.getLogger(logger_name)

    # NOTE: These are the parameters we're optimizing over
    width = params["width"]
    depth = params["depth"]
    lr = params["lr"]

    # Train the model
    bg_tr = batch_generator(X_tr, y_tr, batch_size, num_epochs)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FF(
        input_dim=X_tr.shape[1],
        width=width,
        depth=depth
    ).to(device)

    adam = optim.Adam(lr=lr, params=model.parameters())
    loss_fcn = nn.MSELoss()
    loss_tr = deque(maxlen=LOGGING_INTERVAL)
    for step, (Xb, yb) in enumerate(bg_tr):
        Xb = Xb.to(device)
        yb = yb.to(device)
        adam.zero_grad()
        y_hat = model(Xb)
        loss = loss_fcn(y_hat, yb)
        loss.backward()
        adam.step()

        with torch.no_grad():
            loss_tr.append(loss.cpu().item())

            if not step % LOGGING_INTERVAL:
                logging_loss = np.mean(loss_tr)
                msg = f"[step {step}]: loss: {logging_loss:0.5f}"
                logger_worker.info(msg)

    # Evaluate the model
    bg_val = batch_generator(X_val, y_val, batch_size, 1)
    loss_val = []
    with torch.no_grad():
        for Xb, yb in bg_val:
            Xb = Xb.to(device)
            yb = yb.to(device)
            y_hat = model(Xb)
            loss = loss_fcn(y_hat, yb)
            loss_val.append(loss.cpu().item())
    rmse_val = np.sqrt(np.mean(loss_val))

    # NOTE: This is the only place we're using ray
    tune.report(loss=rmse_val)


def main(
    distributed: bool,
    num_samples: int = 5,
    batch_size: int = 512,
    num_epochs: int = 10
) -> None:
    init_logging("main.log")
    logger.info("Running main ...")
    if distributed:
        ray.init(
            address="localhost:6379",
            _redis_password=os.getenv("RAY_REDIS_PWD"),
            ignore_reinit_error=True
        )
    else:
        ray.init(ignore_reinit_error=True)

    X, y = make_data(NUM_ROW)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)

    # NOTE: Hyperopt config
    metric = "loss"
    mode = "min"
    hp_search = HyperOptSearch(metric=metric, mode=mode)

    # NOTE: Like functools.partial, but stores data in object store
    objective = tune.with_parameters(
        fit, X_tr=X_tr, X_val=X_val, y_tr=y_tr, y_val=y_val,
        batch_size=batch_size, num_epochs=num_epochs
    )

    # NOTE: Define the support of the parameters we're optimizing over
    param_space = {
        "width": tune.choice((2**np.arange(5, 11)).astype(int)),
        "depth": tune.choice(range(1, 5)),
        "lr": tune.loguniform(1e-4, 5e-2)
    }

    logger.info("Starting hyperparameter search ...")
    analysis = tune.run(
        objective,
        num_samples=num_samples,
        config=param_space,
        search_alg=hp_search,
        resources_per_trial={"cpu": 2, "gpu": 0.5},
        metric=metric,
        mode=mode
    )
    best_config = analysis.get_best_config(metric=metric, mode=mode)
    logger.info("Best config:\n%s", best_config)
    with open("/tmp/analysis.p", "wb") as f:
        pickle.dump(analysis, f)
    logger.info("Best results %s", pformat(analysis.results))
    analysis.results_df.to_parquet(RESULTS_PATH)


def load_results() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distributed",
        dest="distributed",
        required=True,
        type=lambda x: (str(x).lower() in ("true", "1", "yes"))
    )
    parser.add_argument("--num-samples", dest="num_samples", type=int, required=True)
    parser.add_argument("--num-epochs", dest="num_epochs", type=int, required=True)
    main(**vars(parser.parse_args()))
