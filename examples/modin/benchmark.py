from contextlib import contextmanager
import time
import logging


import numpy as np
import pandas as pd
import modin.pandas as md
import ray


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@contextmanager
def log_time_usage(prefix=""):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed_seconds = float("%.2f" % (end - start))
        logger.info('%s: elapsed seconds: %s', prefix, elapsed_seconds)


def groupby(df):
    return df.groupby("cat").sum()


def merge(df1, df2):
    return df1.merge(df2, on="cat")


def convert_dtype(df):
    return df.astype(np.float32)


def sort_column(df):
    return df.sort_values(by=df.columns[0])


def standardize(df):
    mu = df.mean(axis=0)
    sig = df.std(axis=0)
    return (df - mu)/(sig + 1e-6)


def make_data(N, K, card, ispd):
    df = pd.DataFrame(np.random.randn(N, K))
    df["cat"] = np.random.choice(range(card), size=N)

    df_merge = pd.DataFrame(np.random.randn(card, K))
    df_merge["cat"] = np.random.choice(range(card), size=card)
    if not ispd:
        df = md.DataFrame(df)
        df_merge = md.DataFrame(df_merge)
    return df, df_merge


def benchmark(N, K, card, ispd):
    df, df_merge = make_data(N, K, card, ispd)

    if ispd:
        prefix = "pandas"
    else:
        prefix = "modin"

    with log_time_usage(f"{prefix}-groupby"):
        _ = groupby(df)

    with log_time_usage(f"{prefix}-merge"):
        _ = merge(df, df_merge)

    with log_time_usage(f"{prefix}-convert-dtype"):
        _ = convert_dtype(df)

    with log_time_usage(f"{prefix}-standardize"):
        _ = standardize(df)


def main():
    N = 20_000_000
    K = 100
    card = 50

    logger.info("Pandas benchmarks ************************")
    benchmark(N, K, card, True)

    logger.info("Modin benchmarks ************************")
    ray.init()
    benchmark(N, K, card, False)
    ray.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )
    main()
