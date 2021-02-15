import logging
import time

import dask
import numpy as np

logger = logging.getLogger(__name__)


def expensive_fcn(x):
    logger.info("Running expensive_fcn.")
    time.sleep(1)
    logger.info("expensive_fcn done.")
    return x


def main():
    xs = np.arange(10)
    ys = [dask.delayed(expensive_fcn)(x) for x in xs]
    ys = dask.compute(ys, scheduler="threads")[0]
    return ys
