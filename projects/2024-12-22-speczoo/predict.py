import json
import joblib
import os
import sys
from laspec import MrsSpec
import numpy as np

from astropy.utils.iers import conf

conf.auto_max_age = None  # 禁用自动更新
conf.auto_download = False  # 关闭自动下载

SP_PATH = "/slam/sp.joblib"
VERBOSE = True
WAVE = np.arange(3950, 5851, 1.0)
INPUT_PATH = "/slam/input.fits"

# Read spectrum
if VERBOSE:
    print("[1/3] Read Spectrum")


def read_spectrum(fp):
    assert os.path.exists(fp), fp
    # read FITS spectrum
    spec = MrsSpec.from_lrs(fp)
    # correct redshift, normalize and interpolate spectrum
    flux, flux_err = spec.interp_then_norm(WAVE, rv=None)
    return flux, flux_err


flux, flux_err = read_spectrum(INPUT_PATH)
flux[flux > 2] = 1
flux[flux < 0] = 1

# Load model (/slam/weight.joblib)
if VERBOSE:
    print("[2/3] Load Model")
assert os.path.exists(SP_PATH), SP_PATH
sp = joblib.load(SP_PATH)

# Predict labels
if VERBOSE:
    print("[3/3] Predict Labels")
x_pred = sp.least_square(
    flux,
    flux_err,
    p0=[6000, 4, -0.1],
    method="trf",
    bounds=(np.ones(3) * -0.6, np.ones(3) * 0.6),
)
LALEL_NAMES = ["Teff", "logg", "[Fe/H]"]
x_pred = dict(zip(LALEL_NAMES, x_pred))
print(json.dumps(x_pred))
