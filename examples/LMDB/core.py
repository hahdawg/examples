from glob import glob
import os
from os.path import join
import pickle
import time
from typing import List

import dask.dataframe as dd
import lmdb
import pandas as pd
import numpy as np
import ray
from torch.utils.data import Dataset, DataLoader
from torch import tensor
from tqdm import tqdm

MS_PER_SECOND = 1000
NUM_ROW = int(1e6)
NUM_COL = 300
MAP_SIZE = int(1024**3)
WORK_DIR = "/tmp/lmdb_example"
DB_DIR = join(WORK_DIR, "db")
DF_PATH = join(WORK_DIR, "data.parquet")


def make_parquet() -> None:
    """
    Make a parquet dataset that we want to generate batches from.
    """
    df = pd.DataFrame(np.random.randn(NUM_ROW, NUM_COL))
    df["y"] = np.random.choice([0, 1], p=[0.95, 0.05], size=len(df))
    df.columns = df.columns.astype(str)
    df = dd.from_pandas(df, npartitions=os.cpu_count())
    df.to_parquet(DF_PATH)


@ray.remote
def convert_partition_to_lmdb(partition_path: str) -> None:
    """
    Convert a parquet partition to one lmdb database.
    """
    df = pd.read_parquet(partition_path)
    lmdb_path = join(DB_DIR, os.path.basename(partition_path) + ".lmdb")
    env = lmdb.open(lmdb_path, map_size=MAP_SIZE, create=True)
    with env.begin(write=True) as txn:
        for i, (_, row) in enumerate(df.iterrows()):
            x = row[:-1].values
            y = row.iloc[-1]
            txn.put(str(i).encode(), pickle.dumps([x, y]))


def convert_parquet_to_lmdb():
    """
    Convert N parquet partitions to N lmdb databases.
    """
    os.system(f"rm -rf {DB_DIR}/*")
    # TODO: make {DB_DIR}
    pq_paths = sorted(glob(f"{DF_PATH}/*.parquet"))
    ray.init(ignore_reinit_error=True)  # change one LOC here to make this distributed
    conversions = [
        convert_partition_to_lmdb.remote(partition_path=path)
        for path in pq_paths
    ]
    ray.get(conversions)
    ray.shutdown()


class LmdbDataset(Dataset):
    """
    Torch dataset. Must implement __getitem__ and __len__.

    We'll use lmdb for this.
    """
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


def compute_dataloader(**dataloader_kwargs):
    """
    Torch data loader (i.e., batch generator).
    """
    dataset = LmdbDataset()
    return DataLoader(dataset=dataset, **dataloader_kwargs)


# pylint: disable=W0613
def exec_gpu_training_step(batch: List[tensor], num_seconds: int) -> None:
    """
    Simulate gpu training step.
    """
    time.sleep(num_seconds)


class Benchmark:
    """
    Benchmark batch loading times.
    """
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def run_sequential_benchmark(self, num_trials: int = 5) -> None:
        """
        Measure cost of loading data sequentially.
        """
        dl = compute_dataloader(
            batch_size=self.batch_size,
            num_workers=1,
            prefetch_factor=1
        )
        start = time.time()
        for i, _ in enumerate(tqdm(dl, total=num_trials)):
            if i >= num_trials:
                break
        batch_time = int(MS_PER_SECOND*(time.time() - start) / num_trials)
        print(f"Non-parallel time per batch: {batch_time} ms")

    def run_parallel_benchmark(self, gpu_train_step_seconds: float = 1.0, num_training_steps: int = 10):
        """
        Measure cost of loading data in parallel.
        """
        dl = compute_dataloader(
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            prefetch_factor=os.cpu_count()
        )
        loop_start = time.time()
        gpu_time = 0.0
        for i, batch in tqdm(enumerate(dl), total=num_training_steps):
            if i >= num_training_steps:
                break
            gpu_start = time.time()
            exec_gpu_training_step(batch=batch, num_seconds=gpu_train_step_seconds)
            gpu_time += time.time() - gpu_start
        total_train_time = time.time() - loop_start
        cpu_time = int(MS_PER_SECOND*(total_train_time - gpu_time) / num_training_steps)
        print(f"Parallel time per batch: {cpu_time} ms")


def make_example():
    os.system(f"rm -rf {WORK_DIR}")
    os.mkdir(WORK_DIR)
    os.mkdir(DB_DIR)
    make_parquet()
    convert_parquet_to_lmdb()


if __name__ == "__main__":
    print("hello")
    # bmk = Benchmark(batch_size=10000)
    # bmk.run_sequential_benchmark(num_trials=10)
    # bmk.run_parallel_benchmark(gpu_train_step_seconds=1.0, num_training_steps=60)
