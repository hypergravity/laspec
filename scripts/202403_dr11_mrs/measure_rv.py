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
from laspec.mrs import MrsFits

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

# rvmdata = joblib.load("RVMDATA_R7500.joblib")
# rvm = RVM(**rvmdata)
# rvm.make_cache(cache_name="B", wave_range=(5000, 5300), rv_grid=(-1000, 1000, 10))
# rvm.make_cache(cache_name="R", wave_range=(6350, 6750), rv_grid=(-1000, 1000, 10))
# # joblib.dump(rvm, "RVM_FOR_PARALLEL.joblib")


t = table.Table.read("dr11v1.0-BR-snr5-im.fits")
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


# =====================================
# process meta
# =====================================
def batch_processing_meta(i_idx=0):
    # print(f"[{i_idx:05d}] Prepare for {i_idx}")
    os.chdir(WORKDIR)
    print(f"[{i_idx:05d}] Process spectra")
    fp_input = f"rvr/input_{i_idx:05d}.joblib"
    kwargs = joblib.load(fp_input)
    # fpout = kwargs["fpout"].replace("output", "meta")
    dlist = []
    for i in range(len(kwargs["fp_list"])):
        this_fp = kwargs["fp_list"][i].strip()
        this_lmjm = kwargs["lmjm_list"][i]

        mf = MrsFits(this_fp)

        try:
            ms = mf.get_one_spec(lmjm=this_lmjm, band="B")
            ind_wave = (ms.wave > 5000) & (ms.wave < 5300)
            npix = np.sum(ind_wave)
            npix_bad = np.sum(ms.mask[ind_wave] > 0)
            d_B = dict(
                exptime_B=ms.exptime,
                lamp_B=ms.lamplist,
                jd_B=ms.jdmid,
                bjdmid_B=ms.bjdmid,
                npix_B=npix,
                npix_bad_B=npix_bad,
            )
        except BaseException:
            d_B = dict(
                exptime_B=np.nan,
                lamp_B="",
                jd_B=np.nan,
                bjdmid_B=np.nan,
                npix_B=-1,
                npix_bad_B=-1,
            )
        try:
            ms = mf.get_one_spec(lmjm=this_lmjm, band="R")
            ind_wave = (ms.wave > 6350) & (ms.wave < 6750)
            npix = np.sum(ind_wave)
            npix_bad = np.sum(ms.mask[ind_wave] > 0)
            d_R = dict(
                exptime_R=ms.exptime,
                lamp_R=ms.lamplist,
                jd_R=ms.jdmid,
                bjdmid_R=ms.bjdmid,
                npix_R=npix,
                npix_bad_R=npix_bad,
            )
        except BaseException:
            d_R = dict(
                exptime_R=np.nan,
                lamp_R="",
                jd_R=np.nan,
                bjdmid_R=np.nan,
                npix_R=-1,
                npix_bad_R=-1,
            )
        d_B.update(d_R)
        dlist.append(d_B)
    return dlist


meta_list = joblib.Parallel(n_jobs=40, verbose=99)(
    joblib.delayed(batch_processing_meta)(i_idx) for i_idx in range(12366)
)
meta = []
for _ in meta_list:
    meta.extend(_)
tmeta = table.Table(meta)
tmeta.write("tmeta.fits")


# =====================================
# gather serv results
# =====================================
import glob
import joblib
from astropy.table import Table

fplist = glob.glob("rvr/out*")
fplist.sort()

rvr_list = joblib.Parallel(n_jobs=30, verbose=99)(
    joblib.delayed(joblib.load)(fp) for fp in fplist
)
rvr_contents = []
for rvr in rvr_list:
    rvr_contents.extend(rvr)
trvr = Table(rvr_contents)
trvr.write("serv.fits", overwrite=True)


# =====================================
# Calculate RVZP
# =====================================
from astropy import table
from laspec.rvzp import calibrate_rvzp
import numpy as np

m11 = table.Table.read("dr11v1.0-BR-snr5-im-gdr3-2mass.fits")
serv = table.Table.read("serv.fits")

tseu, ind_map = calibrate_rvzp(
    np.array(serv["rv1_B", "rv1_R"].to_pandas()),
    np.array(serv["rv1_err_B", "rv1_err_R"].to_pandas()),
    np.where((m11["RV"].mask) | (m11["e_RV"] > 5), np.nan, m11["RV"].data),
    np.where((m11["RV"].mask) | (m11["e_RV"] > 5), np.nan, m11["RV"].data),
    m11["spid"],
    m11["lmjm"],
    rvlabels=["B", "R"],
    ncommon_min=5,
    rv_internal_error=1.0,
    verbose=True,
    debug=False,
)
joblib.dump(tseu, "tseu.joblib")
joblib.dump(ind_map, "ind_map.joblib")
tseu.write("tseu.fits", overwrite=True)
tseu[ind_map].write("rvzp.fits", overwrite=True)

# =====================================
# Plot RVZPs
# =====================================
import joblib
from astropy import table
import matplotlib.pyplot as plt

m = table.Table.read("dr11v1.0-BR-snr5-im.fits")
tseu = joblib.load("tseu.joblib")
rvzp = table.Table.read("rvzp.fits")
meta = table.Table.read("tmeta.fits")
meta.add_columns((m["lmjm"], m["spid"], m["fiberid"]))
meta_good = meta[(meta["exptime_B"] > 0) & (meta["exptime_R"] > 0)]
serv = table.Table.read("serv.fits")

"""
make YJGuo's catalog
"""
print(m.colnames, rvzp.colnames, meta.colnames, serv.colnames)
tgyj = table.hstack(
    (
        m[
            "snr_B",
            "snr_R",
            "obsid",
            "obsdate",
            "lmjd",
            "mjd",
            "planid",
            "spid",
            "fiberid",
            "lmjm",
            "gaia_source_id",
            "gaia_g_mean_mag",
            "gaia_bp_mean_mag",
            "gaia_rp_mean_mag",
            "ra",
            "dec",
            "fibermask",
            "subproject",
            "id",
            "gid",
            "gsize",
        ],
        rvzp[
            "nobs_B",
            "ncommon_B",
            "rvzp_B",
            "rvzp_err_B",
            "nobs_R",
            "ncommon_R",
            "rvzp_R",
            "rvzp_err_R",
        ],
        meta[
            "exptime_B",
            "lamp_B",
            "jd_B",
            "bjdmid_B",
            "exptime_R",
            "lamp_R",
            "jd_R",
            "bjdmid_R",
            "lmjm",
            "spid",
            "fiberid",
        ],
        serv[
            "rv1_B",
            "rv1_err_B",
            "ccfmax1_B",
            "pmod1_B",
            "pmod2_B",
            "ccfmax2_B",
            "rv1_rv2_eta_B",
            "rv1_rv2_eta_err_B",
            "rv1_R",
            "rv1_err_R",
            "ccfmax1_R",
            "pmod1_R",
            "pmod2_R",
            "ccfmax2_R",
            "rv1_rv2_eta_R",
            "rv1_rv2_eta_err_R",
        ],
    )
)

tgyj.write("guo.fits")

u_lmjm_spid, u_idx = np.unique(
    np.array(meta_good["lmjm", "spid"].to_pandas()), axis=0, return_index=True
)
tseu_meta = table.join(
    tseu,
    meta_good[u_idx],
    keys_left=["u_lmjm", "u_spid"],
    keys_right=["lmjm", "spid"],
    join_type="left",
)
print(len(tseu), len(tseu_meta))

# =====================================
# Plot
# =====================================
# %%
idx_good_rvzp_B = (
    (tseu_meta["rvzp_err_B"] < 2)
    & (np.abs(tseu_meta["rvzp_B"]) < 25)
    & ~tseu_meta["rvzp_B"].mask
)
idx_good_rvzp_R = (
    (tseu_meta["rvzp_err_R"] < 2)
    & (np.abs(tseu_meta["rvzp_R"]) < 25)
    & ~tseu_meta["rvzp_R"].mask
)
mean_rvzp_error_B = np.median(tseu_meta[idx_good_rvzp_B]["rvzp_err_B"])
mean_rvzp_error_R = np.median(tseu_meta[idx_good_rvzp_R]["rvzp_err_R"])
n_seu = len(tseu_meta)

n_seu_solved_B = np.sum(idx_good_rvzp_B)
frac_rvzp_B = n_seu_solved_B / n_seu
n_seu_solved_R = np.sum(idx_good_rvzp_R)
frac_rvzp_R = n_seu_solved_R / n_seu


lamp_type_list = [
    "Sc",
    "Ne",
    "ThAr",
    "FeAr",
    # "ThAr2020",
]
color_list = [
    "tab:gray",
    "tab:olive",
    "tab:cyan",
    "tab:pink",
    # "tab:red",
]
lamp_B = dict(
    sc=tseu_meta["lamp_B"] == b"lampsc_med.dat",
    ne=tseu_meta["lamp_B"] == b"lampne_med.dat",
    thar=(tseu_meta["lamp_B"] == b"lampthar.dat")
    | (tseu_meta["lamp_B"] == b"lampthar_20201221.dat"),
    thar2020=tseu_meta["lamp_B"] == b"lampthar_20201221.dat",
    fear=tseu_meta["lamp_B"] == b"lampfear.dat",
)
lamp_R = dict(
    sc=tseu_meta["lamp_R"] == b"lampsc_med.dat",
    ne=tseu_meta["lamp_R"] == b"lampne_med.dat",
    thar=(tseu_meta["lamp_R"] == b"lampthar.dat")
    | (tseu_meta["lamp_R"] == b"lampthar_20201221.dat"),
    thar2020=tseu_meta["lamp_R"] == b"lampthar_20201221.dat",
    fear=tseu_meta["lamp_R"] == b"lampfear.dat",
)

from laspec.mpl import set_cham
from astropy.time import Time

# ['sans-serif']

set_cham(12, latex=False, xminor=False)
plt.rcParams["font.family"] = "Arial"

kwargs = dict(ms=5, mec="k", mew=0.08, alpha=0.7)
fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)
for lamp_type, color in zip(lamp_type_list, color_list):
    axs[0].plot(
        tseu_meta["u_lmjm"][lamp_B[lamp_type.lower()] & idx_good_rvzp_B],
        tseu_meta["rvzp_B"][lamp_B[lamp_type.lower()] & idx_good_rvzp_B],
        "s",
        c=color,
        label=lamp_type,
        **kwargs,
    )
for lamp_type, color in zip(lamp_type_list, color_list):
    axs[1].plot(
        tseu_meta["u_lmjm"][lamp_R[lamp_type.lower()] & idx_good_rvzp_R],
        tseu_meta["rvzp_R"][lamp_R[lamp_type.lower()] & idx_good_rvzp_R],
        "s",
        c=color,
        label=lamp_type,
        **kwargs,
    )

lmjm_tick_list = []
lmjm_label_list = []
for year in range(2017, 2024):
    for month in range(1, 13):
        date_str = f"{year:04d}-{month:02d}-01T00:00:00"
        lmjm_tick_list.append(Time(date_str, format="isot").mjd * 1440)
        lmjm_label_list.append(f"{year:04d}-{month:02d}")


axs[1].set_yticks(np.arange(-30, 30, 5))
axs[1].set_xticks(lmjm_tick_list)
axs[1].set_xticklabels(lmjm_label_list, rotation=60)
axs[1].set_ylim(-17, 17)
axs[1].set_xlim(83500000, 86600000)
axs[0].set_ylabel(r"$\Delta v_B$ [km s$^{-1}$]", fontsize=18)
axs[1].set_ylabel(r"$\Delta v_R$ [km s$^{-1}$]", fontsize=18)
axs[1].legend(loc="lower left", fontsize=10, framealpha=0.5)
axs[0].annotate(
    "Blue Arm",
    xy=(0.1, 0.7),
    xycoords="axes fraction",
    xytext=(0.2, 0.85),
    textcoords="axes fraction",
    color="tab:blue",
    fontsize=25,
    # fontweight="bold",
    # arrowprops=dict(facecolor="black", arrowstyle="->"),
    horizontalalignment="center",
)
axs[1].annotate(
    "Red Arm",
    xy=(0.1, 0.7),
    xycoords="axes fraction",
    xytext=(0.2, 0.85),
    textcoords="axes fraction",
    color="tab:red",
    fontsize=25,
    # fontweight="bold",
    # arrowprops=dict(facecolor="black", arrowstyle="->"),
    horizontalalignment="center",
)

axs[0].annotate(
    f"{n_seu_solved_B}/{n_seu} SEUs solved ({frac_rvzp_B*100:.1f}%)",
    xy=(0.1, 0.2),
    xycoords="axes fraction",
    xytext=(0.1, 0.15),
    textcoords="axes fraction",
    fontsize=12,
    horizontalalignment="left",
)
axs[0].annotate(
    f"median uncertainty = {mean_rvzp_error_B:.2f} km s$^{{-1}}$",
    xy=(0.1, 0.2),
    xycoords="axes fraction",
    xytext=(0.1, 0.05),
    textcoords="axes fraction",
    fontsize=12,
    horizontalalignment="left",
)
axs[1].annotate(
    f"{n_seu_solved_R}/{n_seu} SEUs solved ({frac_rvzp_R*100:.1f}%)",
    xy=(0.1, 0.2),
    xycoords="axes fraction",
    xytext=(0.1, 0.15),
    textcoords="axes fraction",
    fontsize=12,
    horizontalalignment="left",
)
axs[1].annotate(
    f"median uncertainty = {mean_rvzp_error_R:.2f} km s$^{{-1}}$",
    xy=(0.1, 0.2),
    xycoords="axes fraction",
    xytext=(0.1, 0.05),
    textcoords="axes fraction",
    fontsize=12,
    horizontalalignment="left",
)

axs[0].grid(True, linewidth=0.5, color="gray", linestyle="-", alpha=0.5)
axs[1].grid(True, linewidth=0.5, color="gray", linestyle="-", alpha=0.5)

fig.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0.05)
fig.savefig("figs/rvzp_date.pdf")
