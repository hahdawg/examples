from glob import glob
import os
from os.path import join
import pickle
import time

import dask.dataframe as dd
import lmdb
import pandas as pd
import numpy as np
import ray
import torch
from torch.utils.data import Dataset, DataLoader

MAP_SIZE = int(1024**3)
WORK_DIR = "/tmp/lmdb_example"
DB_DIR = join(WORK_DIR, "db")
DF_PATH = join(WORK_DIR, "data.parquet")


def make_parquet(N=100_000, K=300) -> None:
    df = pd.DataFrame(np.random.randn(N, K))
    df["y"] = np.random.choice([0, 1], p=[0.95, 0.05], size=N)
    df.columns = df.columns.astype(str)
    df = dd.from_pandas(df, npartitions=os.cpu_count())
    df.to_parquet(DF_PATH)


@ray.remote
def convert_partition_to_lmdb(partition_path: str) -> None:
    df = pd.read_parquet(partition_path)
    lmdb_path = join(DB_DIR, os.path.basename(partition_path) + ".lmdb")
    env = lmdb.open(lmdb_path, map_size=MAP_SIZE, create=True)
    with env.begin(write=True) as txn:
        for i, (_, row) in enumerate(df.iterrows()):
            x = row[:-1].values
            y = row.iloc[-1]
            txn.put(str(i).encode(), pickle.dumps([x, y]))


def convert_parquet_to_lmdb():
    os.system(f"rm -rf {DB_DIR}/*")
    pq_paths = sorted(glob(f"{DF_PATH}/*.parquet"))
    ray.init(ignore_reinit_error=True)
    conversions = [
        convert_partition_to_lmdb.remote(partition_path=path)
        for path in pq_paths
    ]
    ray.get(conversions)
    ray.shutdown()


class LmdbDataset(Dataset):

    def __init__(self):
        super().__init__()
        db_paths = sorted(glob(f"{DB_DIR}/*.lmdb"))
        self.envs = [
            lmdb.open(path, readonly=True, meminit=False, max_readers=1024, lock=False)
            for path in db_paths
        ]
        self.row_counts = self._compute_row_counts()

    def _compute_row_counts(self) -> np.array:
        row_counts = np.array([db.stat()["entries"] for db in self.envs])
        row_counts = np.cumsum(row_counts)
        return row_counts

    def __len__(self):
        return self.row_counts[-1]

    def __getitem__(self, index: int):
        db_idx = np.searchsorted(self.row_counts, index, side="right")
        if db_idx > 0:
            index -= self.row_counts[db_idx - 1]
        index = int(index)
        with self.envs[db_idx].begin() as txn:
            return pickle.loads(txn.get(str(index).encode()))


def compute_dataloader(batch_size: int) -> DataLoader:
    dataset = LmdbDataset()
    return DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count(), prefetch_factor=os.cpu_count())


def run_benchmark(batch_size: int, gpu_train_step_seconds: float = 0.5, num_training_steps: int = 10):
    dl = compute_dataloader(batch_size=batch_size)
    start = time.time()
    # pylint: disable=W0612
    for i, batch in enumerate(dl):
        if i >= num_training_steps:
            break
        time.sleep(gpu_train_step_seconds)
    total_train_time = time.time() - start
    gpu_time = num_training_steps*gpu_train_step_seconds
    cpu_time = (total_train_time - gpu_time) / num_training_steps
    return cpu_time


def make_example():
    os.system(f"rm -rf {WORK_DIR}")
    os.mkdir(WORK_DIR)
    os.mkdir(DB_DIR)
    make_parquet()
    convert_parquet_to_lmdb()
