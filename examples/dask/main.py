import time
import multiprocessing as mp

import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import pandas as pd


from examples.dask.config import df_path

KEY = "df"

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def make_dataset():
    """
    Make a DataFrame and store in HDF. Note we have to use
    format='table'.
    """
    million = int(1e6)
    N = 100*million
    cat = np.random.randint(0, 50, size=N)
    val = np.random.randn(N)
    df = pd.DataFrame({"cat": cat, "val": val})
    df.to_hdf(df_path, key="df", mode="w", format="table")


def process_df(df):
    df = df.loc[df["val"] >= 0]
    grouped = df.groupby("cat")["val"].std()
    return grouped


@timeit
def process_data_dask():
    """
    Process data in dask. Note the context manager allows us to run this
    more than once in IPython.
    """
    with LocalCluster(
        n_workers=mp.cpu_count(),
        processes=True,
        threads_per_worker=1,
        ip="tcp://localhost:9895"
    ) as cluster, \
    Client(cluster) as _:
        df = dd.read_hdf(df_path, key=KEY, mode="r")
        df = process_df(df)
        df = df.compute()
    return df


@timeit
def process_data_pandas():
    """
    Process data in pandas.
    """
    df = pd.read_hdf(df_path)
    df = process_df(df)
    return df


def run_example():
    make_dataset()
    process_data_dask()
    process_data_pandas()
