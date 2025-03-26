import numpy as np
from laspec.mrs import MrsSpec
from astropy.table import Table, vstack

# cut sample
dr11v1 = Table.read(
    "/nfsdata/share/lamost/dr11-v1.0/catalog/dr11_v1.0_LRS_stellar.fits.gz"
)
# dr12v0 = Table.read(
#     "/nfsdata/share/lamost/dr12-v0/catalog/dr12_v0_LRS_stellar_q1q2q3.fits.gz"
# )
# it seems the columns are inconsistent, use DR11 only

# criteria
idx_selection = (
    (dr11v1["teff"] > 3000)
    & (dr11v1["teff"] < 10000)
    & (dr11v1["teff_err"] < 100)
    & (dr11v1["logg"] > 0)
    & (dr11v1["logg"] < 6)
    & (dr11v1["logg_err"] < 0.2)
    & (dr11v1["feh"] < 1)
    & (dr11v1["feh"] > -3)
    & (dr11v1["feh_err"] < 0.1)
    & (dr11v1["alpha_m_err"] < 0.1)
    & (dr11v1["snrg"] > 50)
    & (dr11v1["snrr"] > 30)
)
print(len(idx_selection), sum(idx_selection))

# determine sample
WAVE = np.arange(3950, 5851, 1.0)
N_PIX = len(WAVE)
N_SAMPLES = 100000
np.random.seed(seed=1)
idx_sample = np.random.choice(np.where(idx_selection)[0], size=N_SAMPLES, replace=False)

# get parameters
p = dr11v1[idx_sample][
    "teff",
    # "teff_err",
    "logg",
    # "logg_err",
    "feh",
    # "feh_err",
    # "rv",
    # "rv_err",
    # "alpha_m",
    # "alpha_m_err",
]
p_arr = np.array([p["teff"], p["logg"], p["feh"]]).T


from laspec.lamost import lamost_filepath

fps = lamost_filepath(
    dr11v1["obsdate"][idx_sample],
    dr11v1["planid"][idx_sample],
    dr11v1["lmjd"][idx_sample],
    dr11v1["spid"][idx_sample],
    dr11v1["fiberid"][idx_sample],
    rootdir="/nfsdata/share/lamost/dr11-v1.0/fits",
    extname="fits.gz",
)

import os

os.path.exists(fps[0])


def read_spectrum(fp):
    assert os.path.exists(fp), fp
    # read FITS spectrum
    spec = MrsSpec.from_lrs(fp)
    # correct redshift, normalize and interpolate spectrum
    flux, flux_err = spec.interp_then_norm(WAVE, rv=None)
    return flux, flux_err


import joblib

spec_data = joblib.Parallel(n_jobs=10, verbose=100)(
    joblib.delayed(read_spectrum)(fp) for fp in fps
)
flux = np.zeros((N_SAMPLES, len(WAVE)))
flux_err = np.zeros((N_SAMPLES, len(WAVE)))
for i in range(N_SAMPLES):
    flux[i], flux_err[i] = spec_data[i]

data = dict(p_arr=p_arr, flux=flux, flux_err=flux_err)
joblib.dump(data, "projects/2024-12-22-speczoo/training_set_speczoo.joblib")


import joblib

data = joblib.load("projects/2024-12-22-speczoo/training_set_speczoo.joblib")
idx_invalid = (data["flux"] > 2) | (data["flux"] < 0)
data["flux"][idx_invalid] = 1

# import model
from laspec.slam import NNModel

# initialize model
nnm = NNModel(
    n_label=3,
    n_pixel=N_PIX,
    n_hidden=200,
    n_layer=3,
    drop_rate=1e-5,
    activation="elu",
    lastrelu=False,
)
# train model
# you need to specify: labels, flux, batch_size, fraction of training set, learning_rate, number of epochs, step for verbose and device
# if you have GPUs inside, you can specify the serial number of the device, e.g., 0, 1, etc.
# it takes ~10 min to run on MacBook16(i9)
nnm.fit(
    data["p_arr"],
    data["flux"],
    batch_size=1000,
    f_train=0.9,
    lr=1e-5,
    n_epoch=10000,
    gain_loss=1e4,
    step_verbose=100,
    device=0,
    loss="L1",
)
