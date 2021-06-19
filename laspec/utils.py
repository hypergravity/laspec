import datetime
import time
import glob
from copy import deepcopy


def ezcount(fmt="", delta_t=3):
    print("---------- time ---------- \t count \t incr")
    c0 = 0
    while True:
        c1 = len(glob.glob(fmt))
        print(datetime.datetime.now(), "\t", c1, "\t", c1-c0)
        c0 = deepcopy(c1)
        time.sleep(delta_t)
    return

