"""
Aim:
    DR11 v1.0 multi-epoch RV measurements.

Last modified:
    2024-03-29
"""

import os
import joblib
from laspec.ccf import RVM
from astropy import table
import numpy as np
from laspec.lamost_kits import CodeKit

hostname = os.uname()[1]
if hostname in ["alpha", "beta", "gamma"]:
    WORKDIR = "/nfsdata/users/cham/projects/lamost/dr11-v1.0/reduced_catalog"
    SPECDIR = "/nfsdata/users/cham/projects/lamost/dr11-v1.0/medfits"
elif hostname == "Mac-Studio.local":
    WORKDIR = "/Users/cham/nfsdata/users/cham/projects/lamost/dr11-v1.0/reduced_catalog"
    SPECDIR = "/Users/cham/nfsdata/users/cham/projects/lamost/dr11-v1.0/medfits"
elif hostname.startswith("MBP"):
    WORKDIR = "/Users/cham/projects/lamost/dr11-v1.0"
    SPECDIR = "/nfsdata/users/cham/projects/lamost/dr11-v1.0/medfits"
else:
    raise ValueError(f"Invalid hostname {hostname}")
os.chdir(WORKDIR)

# =====================================
# Load catalogs
# =====================================

m: table.Table = table.Table.read("dr11v1.0-BR-snr5-im-gdr3-2mass.fits")
m_by_gid = m.group_by("gid")

# =====================================
# Try some cases
# =====================================


# for grp in


# written to merv_test_obsid_list.txt
ugid, ugsize = np.unique(m["gid"].data, return_counts=True)

gid_low = 0
n_found = 0
for gid, gsize in zip(ugid, ugsize):
    if gid <= gid_low:
        continue
    if gsize > 100:
        index = m["gid"] == gid
        mean_snr_B = np.mean(m["snr_B"][index])
        mean_snr_R = np.mean(m["snr_R"][index])
        if mean_snr_B > 30:  # and mean_snr_R > 50:
            print(m["gid"][index][0], m["obsid"][index][0])
            # print(
            #     f"[{t['obsid'][index][0]}]: {gid:07d}, {gsize:03d}, {mean_snr_B:.1f}, {mean_snr_R:.1f}"
            # )
            n_found += 1
    if n_found > 100:
        break

# =====================================
# Group by gid
# =====================================


def proc_merv(t: table.Table):
    pass


# t["obsid", "ra", "dec", "gaia_source_id"][index].show_in_browser()
# np.unique(t["obsid"][index])
from laspec.lamost_kits import MrsKit

gid = 12  # normal star
gid = 25174  # binary
gid = 125558  # KIC6525196

for gid in [12, 25174, 125558]:
    index = m["gid"] == gid
    print(f"mkdir -p gid-{gid:07d}")
    for i in np.where(index)[0]:
        fp = MrsKit.generate_filepath(
            m[i]["lmjd"],
            m[i]["planid"],
            m[i]["spid"],
            m[i]["fiberid"],
            m[i]["obsdate"],
            SPECDIR,
        )
        print(f"cp -v {fp} gid-{gid:07d}/")


from laspec.mrs import MrsSource
import glob

ms = MrsSource.read(glob.glob("multiepoch_rvr/gid-0000012/*.gz"), norm_type="spline")

c = 0
for spec in ms:
    spec.plot_norm()
    print(spec.snr)


n_spec = len(t)
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
            SPECDIR,
            this_date,
            this_planid,
            f"med-{this_lmjd}-{this_planid}_sp{this_spid:02d}-{this_fiberid:03d}.fits.gz",
        )
        d["fp_list"].append(this_fp)
        d["lmjm_list"].append(this_lmjm)
        d["snr_B_list"].append(this_snrB)
        d["snr_R_list"].append(this_snrR)
    joblib.dump(d, fp_input)


def batch_processing(i_idx=0, debug=False):
    print(f"[{i_idx:05d}] Prepare for {i_idx}")
    os.chdir(WORKDIR)
    # load input data
    print(f"[{i_idx:05d}] Process spectra")
    fp_input = f"rvr/input_{i_idx:05d}.joblib"
    kwargs = joblib.load(fp_input)
    # adjust for mac-studio
    hostname = os.uname()[1]
    if hostname == "Mac-Studio.local":
        kwargs["fp_list"] = [
            _.replace("/nfsdata", "/Users/cham/nfsdata") for _ in kwargs["fp_list"]
        ]
        kwargs["fpout"] = "../../" + kwargs["fpout"]
    # debug purpose
    if debug:
        for k in ["fp_list", "lmjm_list", "snr_B_list", "snr_R_list"]:
            kwargs[k] = kwargs[k][:5]
    # skip if output exists
    if os.path.exists(kwargs["fpout"]):
        return
    # make RVM
    print(f"[{i_idx:05d}] Load RVM")
    # rvm = joblib.load("RVM_FOR_PARALLEL.joblib")
    rvmdata = joblib.load("../../rvm/RVMDATA_R7500_M8.joblib")
    rvm = RVM(**rvmdata)
    rvm.make_cache(cache_name="B", wave_range=(5000, 5300), rv_grid=(-1000, 1000, 10))
    rvm.make_cache(cache_name="R", wave_range=(6350, 6750), rv_grid=(-1000, 1000, 10))
    # process spectra
    rvm.mrsbatch(**kwargs)
    # %%timeit -r 10
    # rvm.measure_binary_mrsbatch(
    #     kwargs["fp_list"][0],
    #     kwargs["lmjm_list"][0],
    #     kwargs["snr_B_list"][0],
    #     kwargs["snr_R_list"][0],
    # )
    if os.path.exists(kwargs["fpout"]):
        print(f"[{i_idx:05d}] Finished -> {kwargs['fpout']}")
    else:
        print(f"[{i_idx:05d}] Failed -> {kwargs['fpout']}")


# 7GB / rvm object
# alpha: 1000GB -> 84
# beta:  153GB -> 18
# gamma: 256GB -> 30

n_task = 12365
# alpha
results = joblib.Parallel(n_jobs=72, verbose=99)(
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

# mac-studio
results = joblib.Parallel(n_jobs=6, verbose=99)(
    joblib.delayed(batch_processing)(i_idx) for i_idx in range(5000, 7501, 1)
)
