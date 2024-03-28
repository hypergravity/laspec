"""
Aim:
    DR11 v1.0 RV measurements.

Last modified:
    2024-03-28


cd /nfsdata/users/cham/projects/lamost/dr11-v1.0/reduced_catalog
echo 3 > /proc/sys/vm/drop_caches
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install git+https://github.com/hypergravity/laspec.git --force-reinstall
pip install git+https://gitee.com/hypergravity/laspec.git --force-reinstall

"""

import os
import joblib
from laspec.ccf import RVM
from astropy import table
import numpy as np
from laspec.lamost_kits import CodeKit


os.chdir("/nfsdata/users/cham/projects/lamost/dr11-v1.0/reduced_catalog")

# rvmdata = joblib.load("RVMDATA_R7500.joblib")
# rvm = RVM(**rvmdata)
# rvm.make_cache(cache_name="B", wave_range=(5000, 5300), rv_grid=(-1000, 1000, 10))
# rvm.make_cache(cache_name="R", wave_range=(6350, 6750), rv_grid=(-1000, 1000, 10))
# # joblib.dump(rvm, "RVM_FOR_PARALLEL.joblib")


t = table.Table.read("dr11v1.0-BR-snr5-im.fits")
n_spec = len(t)
spec_root = "/nfsdata/users/cham/projects/lamost/dr11-v1.0/medfits"
idx_list = CodeKit.ezscatter(n_spec, chunksize=1000)
for i_idx, idx in enumerate(idx_list):
    print(f"Prepare for {i_idx}")
    fp_input = f"rvr/input_{i_idx:05d}.joblib"
    fp_output = f"rvr/output_{i_idx:05d}.joblib"
    d = dict(
        fp_list=[],
        lmjm_list=[],
        snr_B_list=[],
        snr_R_list=[],
        snr_threshold=5.0,
        fpout=fp_output,
    )
    for i in idx:
        this_date = t[i]["obsdate"].replace("-", "")
        this_lmjd = t[i]["lmjd"]
        this_planid = t[i]["planid"]
        this_spid = t[i]["spid"]
        this_fiberid = t[i]["fiberid"]
        this_lmjm = t[i]["lmjm"]
        this_snrB = t[i]["snr_B"]
        this_snrR = t[i]["snr_R"]

        # this_snrR = t[i]["rv0_lamost_B"]
        this_fp = os.path.join(
            spec_root,
            this_date,
            this_planid,
            f"med-{this_lmjd}-{this_planid}_sp{this_spid:02d}-{this_fiberid:03d}.fits.gz",
        )
        d["fp_list"].append(this_fp)
        d["lmjm_list"].append(this_lmjm)
        d["snr_B_list"].append(this_snrB)
        d["snr_R_list"].append(this_snrR)
    joblib.dump(d, fp_input)


def batch_processing(i_idx=0):
    print(f"[{i_idx:05d}] Prepare for {i_idx}")
    os.chdir("/nfsdata/users/cham/projects/lamost/dr11-v1.0/reduced_catalog")
    # make RVM
    print(f"[{i_idx:05d}] Load RVM")
    # rvm = joblib.load("RVM_FOR_PARALLEL.joblib")
    rvmdata = joblib.load("../../rvm/RVMDATA_R7500_M8.joblib")
    rvm = RVM(**rvmdata)
    rvm.make_cache(cache_name="B", wave_range=(5000, 5300), rv_grid=(-1000, 1000, 10))
    rvm.make_cache(cache_name="R", wave_range=(6350, 6750), rv_grid=(-1000, 1000, 10))
    # load input data
    print(f"[{i_idx:05d}] Process spectra")
    fp_input = f"rvr/input_{i_idx:05d}.joblib"
    kwargs = joblib.load(fp_input)
    rvm.mrsbatch(**kwargs)
    print(f"[{i_idx:05d}] Finished")


# 7GB / rvm object
# alpha: 1000GB -> 84
# beta:  153GB -> 18
# gamma: 256GB -> 30

n_task = 12365
# alpha
results = joblib.Parallel(n_jobs=84, verbose=99)(
    joblib.delayed(batch_processing)(i_idx) for i_idx in range(0, 12365, 1)
)

# beta
results = joblib.Parallel(n_jobs=18, verbose=99)(
    joblib.delayed(batch_processing)(i_idx) for i_idx in range(8000, 12365, 2)
)

# gamma
results = joblib.Parallel(n_jobs=36, verbose=99)(
    joblib.delayed(batch_processing)(i_idx) for i_idx in range(7501, 12365, 2)
)

# raku
results = joblib.Parallel(n_jobs=32, verbose=99)(
    joblib.delayed(batch_processing)(i_idx) for i_idx in range(5000, 7501, 1)
)
