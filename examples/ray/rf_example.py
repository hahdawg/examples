import os
from typing import List, Tuple

import numpy as np
import ray
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def load_model_data() -> Tuple[np.array, np.array]:
    X, y = load_boston(return_X_y=True)
    return X, y


@ray.remote(num_cpus=1)
def fit_tree(X, y) -> DecisionTreeRegressor:
    n, k = X.shape
    num_features = int(np.sqrt(k))
    feature_idx = np.random.choice(range(k), size=num_features, replace=False)
    row_idx = np.random.choice(range(n), size=n, replace=True)
    X = X[row_idx.reshape(-1, 1), feature_idx]
    y = y[row_idx]
    model = DecisionTreeRegressor()
    model.feature_idx = feature_idx
    model.fit(X, y)
    return model


def fit_rf(X: np.array, y: np.array, n_estimators: int) -> List[DecisionTreeRegressor]:
    X_id = ray.put(X)
    y_id = ray.put(y)
    tasks = [fit_tree.remote(X_id, y_id) for _ in range(n_estimators)]
    model = ray.get(tasks)
    return model


@ray.remote(num_cpus=1)
def predict_tree(tree: DecisionTreeRegressor, X: np.array) -> np.array:
    X = X[:, tree.feature_idx]
    return tree.predict(X)


def predict_rf(rf: List[DecisionTreeRegressor], X: np.array) -> np.array:
    X_id = ray.put(X)
    tasks = [predict_tree.remote(model, X_id) for model in rf]
    pred = ray.get(tasks)
    pred = np.stack(pred).mean(axis=0)
    return pred


def main() -> None:
    X, y = load_model_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    ray.init(
        address="localhost:6379",
        _redis_password=os.getenv("RAY_REDIS_PWD"),
        ignore_reinit_error=True
    )
    rf = fit_rf(X_tr, y_tr, 1000)
    y_tr_hat = predict_rf(rf, X_tr)
    y_te_hat = predict_rf(rf, X_te)
    ray.shutdown()
    r_sq_tr = r2_score(y_true=y_tr, y_pred=y_tr_hat)
    r_sq_te = r2_score(y_true=y_te, y_pred=y_te_hat)
    print(f"r-square on train set: {r_sq_tr:0.4f}")
    print(f"r-square on test set: {r_sq_te:0.4f}")


if __name__ == "__main__":
    main()
