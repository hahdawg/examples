from collections import namedtuple
import pickle
from typing import Tuple

import time


DbOutput = namedtuple("DbOutput", ["colname", "colvals"])
OUTPUT_PATH = "/tmp/examples_testing_main.p"


def query_db(colname: str) -> DbOutput:
    time.sleep(10)
    if colname.startswith("a"):
        return {colname: ["a", "b", "c"]}
    return DbOutput(colname=colname, colvals=["x", "y", "z"])


def process_data(data: DbOutput) -> Tuple[str, str]:
    return (data.colname, "".join(data.colvals))


def main():
    raw = query_db("foo")
    proc = process_data(raw)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(proc, f)
