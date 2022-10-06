"""
This module represents app code that interacts with the outside world.
"""
import pickle
import time

import examples.testing.config as cfg
import examples.testing.lib as lib


def query_db(colname: str) -> lib.DbOutput:
    """
    Simulate long-running db query.
    """
    time.sleep(10)
    return lib.DbOutput(colname=colname, colvals=["x", "y", "z"])


def main() -> None:
    """
    Main function for app:
    * load data from db
    * process data
    * save it to cfg.output_path
    """
    raw = query_db("foo")
    proc = lib.process_data(raw)
    with open(cfg.output_path, "wb") as f:
        pickle.dump(proc, f)
