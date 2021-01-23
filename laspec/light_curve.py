import numpy as np


def read_lightcurve(fp="/Users/cham/projects/sb2/lightcurve/lc/lc488994.dat"):
    with open(fp, "r") as f:
        s = f.readlines()
    s.pop(1)
    s.pop(1)
    s[0] = "HJD\tCam\tmag\tmag_err\tflux\tflux_err\n"
    return
