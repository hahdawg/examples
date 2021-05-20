from contextlib import contextmanager
import os
import time
from typing import List, Tuple

import numpy as np
import ray
from sklearn.datasets import load_boston
import sklearn.ensemble as ens
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def load_model_data() -> Tuple[np.array, np.array]:
    X, y = load_boston(return_X_y=True)
    return X, y


class RandomForestRegressor:

    def __init__(self, n_estimators: int, max_features: str = "sqrt"):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self._trees: List[DecisionTreeRegressor] = None

    def fit(self, X: np.array, y: np.array) -> None:
        X_id = ray.put(X)
        y_id = ray.put(y)
        self_id = ray.put(self)
        tasks = [_fit_tree.remote(self_id, X_id, y_id) for _ in range(self.n_estimators)]
        self._trees = ray.get(tasks)

    def predict_rf(self, X: np.array) -> np.array:
        X_id = ray.put(X)
        tasks = [_predict_tree.remote(tree, X_id) for tree in self._trees]
        pred = ray.get(tasks)
        pred = np.stack(pred).mean(axis=0)
        return pred


@ray.remote(num_cpus=1)
def _fit_tree(rf: RandomForestRegressor, X: np.array, y: np.array) -> DecisionTreeRegressor:
    n, k = X.shape
    if rf.max_features == "sqrt":
        num_features = int(np.sqrt(k))
    else:
        raise RuntimeError()
    feature_idx = np.random.choice(range(k), size=num_features, replace=False)
    row_idx = np.random.choice(range(n), size=n, replace=True)
    X = X[row_idx.reshape(-1, 1), feature_idx]
    y = y[row_idx]
    model = DecisionTreeRegressor()
    model.feature_idx = feature_idx
    model.fit(X, y)
    return model


@ray.remote(num_cpus=1)
def _predict_tree(tree: DecisionTreeRegressor, X: np.array) -> np.array:
    X = X[:, tree.feature_idx]
    return tree.predict(X)


def compute_metric(model: RandomForestRegressor, X: np.array, y: np.array) -> float:
    y_hat = model.predict(X)
    r_sq = r2_score(y_true=y, y_pred=y_hat)
    return r_sq


@contextmanager
def log_time_usage(prefix: str = "") -> None:
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed_seconds = float("%.2f" % (end - start))
        print(f"{prefix}: elapsed seconds: {elapsed_seconds}")


def main() -> None:
    n_estimators = 5000
    X, y = load_model_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    ray.init(
        address="localhost:6379",
        _redis_password=os.getenv("RAY_REDIS_PWD"),
        ignore_reinit_error=True
    )

    with log_time_usage("distributed"):
        rf = RandomForestRegressor(n_estimators=n_estimators)
        rf.fit(X_tr, y_tr)
        ray.shutdown()
        r_sq_te = compute_metric(rf, X_te, y_te)

    r_sq_tr = compute_metric(rf, X_tr, y_tr)
    print(f"r-square on train set: {r_sq_tr:0.4f}")
    print(f"r-square on test set: {r_sq_te:0.4f}\n")

    with log_time_usage("sklearn"):
        rf = ens.RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            max_features="sqrt"
        )
        rf.fit(X_tr, y_tr)
        r_sq_te = compute_metric(rf, X_te, y_te)
    print(f"r-square on train set: {r_sq_tr:0.4f}")


if __name__ == "__main__":
    main()
