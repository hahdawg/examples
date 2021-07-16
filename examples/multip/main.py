import multiprocessing as mp
import time

# see: https://pythonspeed.com/articles/python-multiprocessing/


def expensive_fcn(x):
    time.sleep(1)
    return x


def main():
    num_reps = 10
    with mp.get_context("spawn").Pool(processes=mp.cpu_count()) as pool:
        res = [pool.apply_async(expensive_fcn, args=(i,)) for i in range(num_reps)]
        res = [p.get() for p in res]
    return res


if __name__ == "__main__":
    print(main())
