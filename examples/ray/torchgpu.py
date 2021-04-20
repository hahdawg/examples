from collections import deque
import os

import numpy as np
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim

LOGGING_INTERVAL = 1000


def make_data(N, K=2):
    X = np.random.randn(N, K)
    fcns = (np.square, np.abs, np.sin, np.cos)
    Z = np.concatenate([f(X) for f in fcns], axis=1)
    Z = (Z - Z.mean(axis=0))/Z.std(axis=0)
    y = Z.sum(axis=1) + 0.5*np.random.randn(N)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()


class FF(nn.Module):

    def __init__(self, input_dim, width, depth):
        super().__init__()
        widths = [input_dim] + depth*[width]
        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_in, n_out) for (n_in, n_out) in zip(widths[:-1], widths[1:])
        ])
        self.output_layer = nn.Linear(width, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
        return self.output_layer(x).flatten()


def batch_generator(X, y, batch_size, num_epochs):
    for _ in range(num_epochs):
        for i in range(0, X.shape[0], batch_size):
            Xb = X[i:i + batch_size]
            yb = y[i:i + batch_size]
            yield Xb, yb


def fit(params, batch_size, num_epochs, X_tr, X_val, y_tr, y_val):
    width = params["width"]
    depth = params["depth"]
    lr = params["lr"]
    bg_tr = batch_generator(X_tr, y_tr, batch_size, num_epochs)
    device = "cuda"

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
                print(msg)

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
    tune.report(loss=rmse_val)


def main():
    N = 1_000_000
    num_samples = 30
    batch_size = 512
    num_epochs = 10
    ray.init(
        address="localhost:6379",
        _redis_password=os.getenv("RAY_REDIS_PWD"),
        ignore_reinit_error=True
    )
    X, y = make_data(N)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2)
    metric = "loss"
    mode = "min"
    param_space = {
        "width": tune.choice((2**np.arange(5, 11)).astype(int)),
        "depth": tune.choice(range(1, 5)),
        "lr": tune.loguniform(1e-5, 1e-2)
    }
    hp_search = HyperOptSearch(param_space, metric=metric, mode=mode)
    objective = tune.with_parameters(
        fit, X_tr=X_tr, X_val=X_val, y_tr=y_tr, y_val=y_val,
        batch_size=batch_size, num_epochs=num_epochs
    )
    analysis = tune.run(
        objective,
        num_samples=num_samples,
        search_alg=hp_search,
        resources_per_trial={"cpu": 8, "gpu": 1},
        metric=metric,
        mode=mode
    )
    print(analysis.best_config)


if __name__ == "__main__":
    main()