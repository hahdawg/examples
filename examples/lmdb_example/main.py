import os
import pickle
import torch

import lmdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DB_PATH = "/tmp/mydb"


def make_dataset(N=50_000_000):
    df = pd.DataFrame(np.random.randn(N, 4), columns=list("abcd"))
    df["y"] = np.random.choice([0, 1], p=[0.95, 0.05], size=N)
    return df


def save_dataset(df: pd.DataFrame):
    map_size = int(1e12)
    os.system(f"rm -rf {DB_PATH}")
    env = lmdb.open(DB_PATH, map_size=map_size, create=True)

    with env.begin(write=True) as txn:
        for i, (_, row) in enumerate(tqdm(df.iterrows(), total=df.shape[0])):
            txn.put(str(i).encode(), pickle.dumps(row))


def load_dataset():
    env = lmdb.open(DB_PATH, readonly=True)
    output = []
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for k, v in tqdm(cursor):
            output.append((pickle.loads(k), pickle.loads(v)))
    return output


class LmdbDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.env = lmdb.open(DB_PATH, readonly=True, meminit=False, max_readers=1024, lock=False)
        self.txn  = self.env.begin(write=False)
        self._len = self.env.stat()["entries"]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        index = str(index).encode()
        res = pickle.loads(self.txn.get(index))
        res = torch.from_numpy(res.values)
        return res


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, labels):
        super().__init__(data_source=None)
        self.dataset = dataset
        self.indicies = list(range(len(self.dataset)))
        self.num_samples = len(self.dataset)

        df = pd.DataFrame()
        df["label"] = labels
        counts = df["label"].value_counts()
        self.weights = torch.from_numpy(1.0 / counts[df["label"]].values)

    def __iter__(self):
        return (self.indicies[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return len(self.dataset)


def compute_dataloader():
    dataset = LmdbDataset()
    dl_params = {"dataset": dataset, "batch_size": 16000, "num_workers": 16}

    dl_label = iter(DataLoader(**dl_params))
    labels = torch.cat([batch[:, -1] for batch in dl_label]).numpy()

    sampler = ImbalancedDatasetSampler(dataset=dataset, labels=labels)
    return DataLoader(dataset, batch_size=16000, num_workers=8, sampler=sampler)
