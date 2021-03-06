from typing import List, Tuple
import os

import numpy as np
import pandas as pd
import ray


class TargetEncoder:

    def __init__(self, columns: List[str]):
        self.columns = columns
        self.target_col = "target"
        self.mappings = None

    @staticmethod
    def _init_ray() -> None:
        ray.init(
            address="localhost:6379",
            _redis_password=os.getenv("RAY_REDIS_PWD"),
            ignore_reinit_error=True
        )

    @staticmethod
    def _encoded_name(c):
        return f"{c}_encoded"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        df = X.copy()
        df[self.target_col] = y
        self._init_ray()
        df_id = ray.put(df)
        mappings = ray.get([
            _fit_column.remote(self, c, df_id) for c in self.columns
        ])
        self.mappings = dict(mappings)
        ray.shutdown()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._init_ray()
        X_id = ray.put(X)
        xform =  ray.get(
            [_transform_column.remote(self, c, X_id) for c in self.columns]
        )
        ray.shutdown()
        xform = pd.concat(xform, axis=1)
        non_cat = X.columns.difference(self.columns).to_list()
        return pd.concat(X[non_cat, xform], axis=1)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)


# pylint: disable=W0212
@ray.remote
def _fit_column(te: TargetEncoder, col: str, df: pd.DataFrame) -> Tuple[str, pd.Series]:
    """
    NOTE: Remote functions can't be methods.
    """
    encoded = df.groupby(col)[te.target_col].mean()
    encoded.name = f"{col}_encoded"
    encoded.name = te._encoded_name(col)
    return col, encoded


# pylint: disable=W0212
@ray.remote
def _transform_column(te: TargetEncoder, col: str, X: pd.DataFrame) -> pd.Series:
    """
    NOTE: Remote functions can't be methods.
    """
    merged = X.merge(te.mappings[col], left_on=col, how="left", right_index=True)[te._encoded_name(col)]
    merged = merged.fillna(np.nan)
    merged.index = X.index
    return merged


def main():
    N = 1_000_000
    K = 100
    X = pd.DataFrame(np.random.randint(0, 1000, size=(N, K)))
    X.columns = X.columns.astype(str)
    y = np.random.randn(N)
    te = TargetEncoder(columns=X.columns)
    X = te.fit_transform(X, y)
    print(X)

if __name__ == "__main__":
    main()
