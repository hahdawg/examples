import multiprocessing as mp

import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd


from examples.dask.config import df_path

KEY = "df"


def make_dataset():
    """
    Make a DataFrame and store in HDF. Note we have to use
    format='table'.
    """
    million = int(1e6)
    N = 100*million
    cat = np.random.randint(0, 100, size=N)
    val = np.random.randn(N)
    df = pd.DataFrame({"cat": cat, "val": val})
    df.to_hdf(df_path, key="df", mode="w", format="table")


def group_data():
    """
    Do a groupby. Note the context manager allows us to run this
    more than once in IPython.
    """
    with LocalCluster(
        n_workers=mp.cpu_count() // 2,
        processes=True,
        threads_per_worker=1,
        ip="tcp://localhost:9895"
    ) as cluster, \
    Client(cluster) as _:
        df = dd.read_hdf(df_path, key=KEY, mode="r")
        grouped = df.groupby("cat")["val"].std()
        grouped = grouped.compute()
    print(grouped)


def run_example():
    """
    This should half the cores.
    """
    make_dataset()
    group_data()
