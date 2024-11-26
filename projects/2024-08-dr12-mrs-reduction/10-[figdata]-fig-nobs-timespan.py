import os
import joblib
from laspec.ccf import RVM
from astropy import table
import numpy as np
from laspec.lamost_kits import CodeKit
from laspec.mrs import MrsFits

# ========================================
# Change working directory
# ========================================
hostname = os.uname()[1]
assert hostname in ["alpha", "beta", "gamma"]
if hostname in ["alpha", "beta", "gamma"]:
    WORKDIR = "/nfsdata/share/lamost/dr12v0-mrs-reduction"
    SPECDIR11 = "/nfsdata/share/lamost/dr11-v1.0/medfits"
    SPECDIR12 = "/nfsdata/share/lamost/dr12-v0/medfits"
else:
    raise ValueError(f"Invalid hostname {hostname}")
os.chdir(WORKDIR)

# rvmdata = joblib.load("RVMDATA_R7500.joblib")
# rvm = RVM(**rvmdata)
# rvm.make_cache(cache_name="B", wave_range=(5000, 5300), rv_grid=(-1000, 1000, 10))
# rvm.make_cache(cache_name="R", wave_range=(6350, 6750), rv_grid=(-1000, 1000, 10))
# # joblib.dump(rvm, "RVM_FOR_PARALLEL.joblib")
# %% Generate input data

t = table.Table.read("dr12v0-BR-snr5-im-fp.fits")

t = t["lmjm", "gid"]


def split_via_gid(t, n_split=10):

    # sort and split lmjm
    data = t["gid"].data
    data_sorted = np.sort(t["gid"].data)

    # determine edges
    split_size = int(len(data_sorted) / n_split)
    data_edges = [data_sorted[i * split_size] for i in range(n_split)]
    data_edges.append(np.inf)

    # split
    print("Split size:", split_size)
    chunks = []
    for i in range(n_split):
        if i % 10 == 0:
            print(f"{i} / {n_split}")
        chunks.append(t[(data >= data_edges[i]) & (data < data_edges[i + 1])])
    return chunks


def stat_nobs_timespan_chunk(t):
    gid = t["gid"].data
    ugid, ugid_nobs = np.unique(gid, return_counts=True)
    ugid_lmjm_ptp = np.zeros(len(ugid), np.float32)
    for i, this_ugid in enumerate(ugid):
        # if i % 1000 == 0:
        #     print(i)
        this_idx = gid == this_ugid
        ugid_lmjm_ptp[i] = t["lmjm"][this_idx].ptp()
    return table.Table(
        [ugid, ugid_nobs, ugid_lmjm_ptp], names=["ugid", "nobs", "lmjm_ptp"]
    )


chunks = split_via_gid(t, n_split=1000)
# stat_nobs_timespan_chunk(chunks[0])

import joblib

stats = joblib.Parallel(n_jobs=10, verbose=True)(
    joblib.delayed(stat_nobs_timespan_chunk)(_) for _ in chunks
)

tstats = table.vstack(stats)
joblib.dump(tstats, "figs/nobs-timespan.joblib")

# %%
from laspec.mpl import set_cham
import matplotlib.pyplot as plt

set_cham(fontsize=15, latex=True)

# %%
nobs_step = 10
span_step = 30

nobs_edges = np.linspace(0, 250, int(250 / nobs_step) + 1)
span_edges = np.linspace(0, 2460, int(2460 / span_step) + 1)
print(nobs_edges)
print(span_edges)

nobs_centers = (nobs_edges[1:] + nobs_edges[:-1]) / 2
span_centers = (span_edges[1:] + span_edges[:-1]) / 2

# %%
# plt.rcParams["font.weight"] = "bold"
fig = plt.figure(figsize=(6, 6))

h, _, _ = np.histogram2d(
    tstats["nobs"], tstats["lmjm_ptp"] / 1440, bins=(nobs_edges, span_edges)
)

ax = fig.add_axes([0.15, 0.14, 0.65, 0.65])
h_plot = np.where(np.isfinite(np.log10(h)), np.log10(h), -3)
handle = ax.imshow(
    h_plot,
    vmin=0,
    vmax=4,
    cmap=plt.cm.magma_r,
    origin="lower",
    aspect="auto",
    extent=(
        span_edges[0],
        span_edges[-1],
        nobs_edges[0],
        nobs_edges[-1],
    ),
)
ax.set_xlim(span_edges[[0, -1]])
ax.set_ylim(nobs_edges[[0, -1]])
ax.set_xlabel("Time span [day]", fontsize=20)
ax.set_ylabel("Number of exposures", fontsize=20)

cax = fig.add_axes([0.25, 0.75, 0.45, 0.02])
cbar = fig.colorbar(handle, cax=cax, orientation="horizontal")
cbar.set_label("log10(#)")

ax = fig.add_axes([0.15, 0.82, 0.65, 0.15])
ax.bar(
    span_centers,
    np.log10(h.sum(axis=0)),
    width=30,
    color="gray",
    edgecolor="gray",
    lw=2,
)
ax.set_ylim(0, 7)
ax.set_xlim(span_edges[[0, -1]])
ax.set_ylabel("log10(#)")
ax.xaxis.set_tick_params(labelleft=False, labelright=False)

ax = fig.add_axes([0.82, 0.14, 0.15, 0.65])
ax.barh(
    nobs_centers,
    np.log10(h.sum(axis=1)),
    height=10,
    color="gray",
    edgecolor="gray",
    lw=2,
)
ax.set_xlim(0, 7)
ax.set_ylim(nobs_edges[[0, -1]])
ax.set_xlabel("log10(#)")
ax.yaxis.set_tick_params(labelleft=False, labelright=False)

# fig.tight_layout()
fig.savefig("figs/nobs_timespan.pdf")
plt.show()
