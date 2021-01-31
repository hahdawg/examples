import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

filepath = pathlib.Path(__file__).parent.absolute()
data_dir = os.path.join(filepath, "data")
data_path = os.path.join(data_dir, "data.csv")
pred_path = os.path.join(data_dir, "pred.csv")


def make_data(N=100, K=2):
    randn = np.random.randn
    X = randn(N, K)
    w = np.ones(K)
    y = (X*w).sum(axis=1) + 1e-5*randn(N)
    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(K)])
    data["y"] = y
    data.to_csv(data_path)


def load_data():
    res = pd.read_csv(data_path)
    X = res.drop("y", axis=1)
    y = res["y"]
    return X, y


def predict(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_hat = pd.DataFrame(model.predict(X), columns=["y"])
    return y_hat


def main():
    print(os.listdir(filepath))
    X, y = load_data()
    y_hat = predict(X, y)
    y_hat.to_csv(pred_path)
    print(f"wrote predictions to {pred_path}")


if __name__ == "__main__":
    main()
