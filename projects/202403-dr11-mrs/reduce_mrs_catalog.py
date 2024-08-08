import os
import joblib
import tempfile
import subprocess
from copy import deepcopy
import numpy as np
from astropy import table
from tqdm import trange

"""
Since DR11v1.0, the `obsid` column is no longer useful,
I synthesize a new column `sobsid` = f"{lmjm:08d}-{spid:02d}-{fiberid:03d}"
"""


def preprocessing(
    t: table.Table,
    colnames_reference="dr10v1.0-colnames.joblib",
):
    # if colnames_reference is specified, compare the colnames
    if os.path.exists(colnames_reference):
        colnames_reference: list = joblib.load(colnames_reference)
        print("compare columns with dr10 v1.0:")
        print("columns in dr10 v1.0 but not in this table:")
        print(set(colnames_reference).difference(set(t.colnames)))
        print("columns in this table but not in dr10 v1.0:")
        print(set(t.colnames).difference(set(colnames_reference)))
    else:
        print("skipping comparison ...")

    # select single exposure spectra
    t = t[t["coadd"] == 0]
    sobsid = []
    for _ in trange(len(t)):
        sobsid.append(f"{t[_]['lmjm']:08d}-{t[_]['spid']:02d}-{t[_]['fiberid']:03d}")
    t.add_column(table.Column(sobsid, name="sobsid"))

    # delete useless columns
    useless_columns = ["mobsid", "uid", "coadd"]
    for colname in useless_columns:
        if colname in t.colnames:
            print(f"remove {colname}")
            t.remove_column(colname)
    # map columns
    colname_mapping = {
        "rv_b0": "rv0_lamost_B",
        "rv_b0_err": "rv0_err_lamost_B",
        "rv_b1": "rv1_lamost_B",
        "rv_b1_err": "rv1_err_lamost_B",
        "rv_b_flag": "rv_flag_lamost_B",
        "rv_r0": "rv0_lamost_R",
        "rv_r0_err": "rv0_err_lamost_R",
        "rv_r1": "rv1_lamost_R",
        "rv_r1_err": "rv1_err_lamost_R",
        "rv_r_flag": "rv_flag_lamost_R",
        "bad_b": "bad_lamost_B",
        "bad_r": "bad_lamost_R",
        "rv_br0": "rv0_lamost_BR",
        "rv_br0_err": "rv0_err_lamost_BR",
        "rv_br1": "rv1_lamost_BR",
        "rv_br1_err": "rv1_err_lamost_BR",
        "rv_br_flag": "rv_flag_lamost_BR",
    }
    for colname_old, colname_new in colname_mapping.items():
        if colname_old in t.colnames:
            print(f"map {colname_old} to {colname_new}")
            t.rename_column(colname_old, colname_new)
        else:
            print(f"{colname_old} does not exist, skip mapping ...")
    # duplicate snr
    t.add_column(t["snr"], name="snr_B")
    t.add_column(t["snr"], name="snr_R")
    t.remove_column("snr")
    # check final columns
    print("Check columns:")
    print(t.colnames)
    print(len(t))
    return t


def internal_match(t: table.Table, n_split: int = 10):
    """
    shared columns: shared by B and R. maintain one
    respective columns: different for B and R. copy two
    """

    is_B = t["band"] == "B"
    is_R = t["band"] == "R"
    print(np.sum(is_B), np.sum(is_R), len(t))
    assert np.sum(is_B) + np.sum(is_R) == len(t)

    # sort and split lmjm
    lmjm = t["lmjm"].data
    lmjm_sorted = np.sort(t["lmjm"].data)

    # determine edges
    split_size = int(len(lmjm_sorted) / n_split)
    lmjm_edges = [lmjm_sorted[i * split_size] for i in range(n_split)]
    lmjm_edges.append(np.inf)

    colnames = t.colnames
    total_count = 0
    xt = []
    for i_split in range(n_split):
        print(f"{i_split}/{n_split}")

        lmjm_lo, lmjm_hi = lmjm_edges[i_split : i_split + 2]
        this_ind = (lmjm >= lmjm_lo) & (lmjm < lmjm_hi)
        print(f"{sum(this_ind)} rows selected within lmjm = [{lmjm_lo}, {lmjm_hi}]")

        this_t_B = t[this_ind & is_B]
        this_t_R = t[this_ind & is_R]

        for colname in this_t_B.colnames:
            if colname.endswith("_R"):
                print(f"this_t_B: remove column {colname}")
                this_t_B.remove_column(colname)
        for colname in this_t_R.colnames:
            if colname.endswith("_B"):
                print(f"this_t_R: remove column {colname}")
                this_t_R.remove_column(colname)
        this_t_BR = table.join(
            this_t_B,
            this_t_R,
            # keys=["lmjm", "spid", "fiberid"],
            keys=["sobsid"],
            join_type="outer",
            table_names=["1", "2"],
        )
        ind_B = ~this_t_BR["obsid_1"].mask
        ind_R = ~this_t_BR["obsid_2"].mask
        for colname in colnames:
            if (
                colname + "_1" in this_t_BR.colnames
                and colname + "_2" in this_t_BR.colnames
            ):

                print(f"combine column: {colname}")
                combined_col = table.Column(
                    data=np.where(
                        ind_B,
                        this_t_BR[colname + "_1"].data,
                        this_t_BR[colname + "_2"].data,
                    ),
                    name=colname,
                )
                this_t_BR.add_column(combined_col)
                this_t_BR.remove_columns([colname + "_1", colname + "_2"])
            else:
                print(f"skip column {colname}")

        xt.append(this_t_BR)

    return table.vstack(xt)


# after topcat / stilts
def group_obsid(t, radius=5, temp_folder=".", stilts=None):
    hostname = os.uname()[1]
    if stilts is None:
        if hostname == "alpha":
            stilts = "/home/cham/stilts.jar"
        else:
            stilts = "/Users/cham/stilts.jar"
    print(f"Path of stilts: {stilts}")
    print(f"Length of table: {len(t)}")
    # temp_folder = tempfile.gettempdir()
    file_obsid_ra_dec = os.path.join(temp_folder, "obsid_ra_dec.fits")
    file_group_results = os.path.join(temp_folder, "group_results.fits")

    command = (
        f'java -jar {stilts} tmatch1 matcher=sky params={radius} values="ra dec" '
        f"action=identify in={file_obsid_ra_dec} out={file_group_results}"
    )
    print("Group ra-dec with stilts...")
    print(command)
    # it seems not to be a good choice to use unique obsids
    # but can be fixed by some post-processing
    u_obsid, u_index, u_counts = np.unique(
        t["obsid"].data, return_index=True, return_counts=True
    )
    t_uinfo = t["ra", "dec", "obsid"][u_index]
    t_uinfo.write(file_obsid_ra_dec, overwrite=True)

    # use all obsids
    # too slow
    # t["ra", "dec", "obsid", "sobsid"].write(file_obsid_ra_dec, overwrite=True)

    if os.path.exists(file_group_results):
        print(f"Remove {file_group_results}")
        os.remove(file_group_results)
    print("Internal match ...")
    result = subprocess.run(command, capture_output=True, shell=True)
    print(
        f"Stilts returncode: {result.returncode}\n"
        f" - stdout: {result.stdout.decode()}\n"
        f" - stderr: {result.stderr.decode()}"
    )
    assert os.path.exists(file_group_results)

    # read results obsid <-> GroupID
    print("Reading stilts results ...")
    t_uresult = table.Table.read(file_group_results)["obsid", "GroupID"]
    n_pseudosingle_observation = t_uresult["GroupID"].mask.sum()
    print(f"- n_pseudosingle_observation={n_pseudosingle_observation}")
    gid = np.copy(t_uresult["GroupID"].data)
    gid[t_uresult["GroupID"].mask] = -1 - np.arange(
        n_pseudosingle_observation, dtype=int
    )
    t_uresult.remove_column("GroupID")
    t_uresult.add_column(table.Column(table.Column(data=gid, name="gid")))

    # add id column
    t_obsid = t["id", "obsid"]
    t_obsid_gid = table.join(t_obsid, t_uresult, keys=["obsid"], join_type="left")
    # find unique gid
    u_gid, u_gsize = np.unique(t_obsid_gid["gid"].data, return_counts=True)
    t_gid_gsize = table.Table(
        [
            table.Column(u_gid, name="gid"),
            table.Column(u_gsize, name="gsize"),
        ]
    )
    # map negative gid with gsize>1 to positive
    u_gid_to_map = np.unique(
        t_gid_gsize["gid"][(t_gid_gsize["gsize"] > 1) & (t_gid_gsize["gid"] < 0)].data
    )
    n_map = len(u_gid_to_map)
    n_gid_start = np.max(t_gid_gsize["gid"]) + 1
    u_gid_mapped = np.arange(
        n_gid_start, n_gid_start + len(u_gid_to_map), dtype=u_gid_to_map.dtype
    )
    print("Construct mapping dict ...")
    map_dict = dict()
    for i in trange(n_map):
        map_dict[u_gid_to_map[i]] = u_gid_mapped[i]
    print("Fix pseudo-single observation to positive gids ...")

    # make mapping dict
    mapped_gid = np.empty_like(t_uresult["gid"].data)
    for i in trange(len(t_uresult)):
        if t_uresult[i]["gid"] in map_dict.keys():
            mapped_gid[i] = map_dict[t_uresult[i]["gid"]]
        else:
            mapped_gid[i] = t_uresult[i]["gid"]

    # redefine negative gids
    n_single_observation = np.sum(mapped_gid < 0)
    print(f"- n_single_observation={n_single_observation}")
    mapped_gid[mapped_gid < 0] = -1 - np.arange(n_single_observation, dtype=int)

    # use mapped gid in t_uresult
    t_uresult["gid"] = mapped_gid

    # find unique gid
    # u_gid, u_gsize = np.unique(t_uresult["gid"].data, return_counts=True)
    print("Add gsize column")
    t_im_obsid_gid = table.join(
        t_obsid,
        t_uresult,
        keys=["obsid"],
        join_type="left",
    )
    u_gid, u_gsize = np.unique(t_im_obsid_gid["gid"], return_counts=True)
    gsize_dict = dict()
    for _gid, _gsize in zip(u_gid, u_gsize):
        gsize_dict[_gid] = _gsize

    # set gsize
    print("Set gsize")
    t_im_obsid_gid.add_column(
        table.Column(np.zeros(len(t), dtype=np.int32), name="gsize")
    )
    for i in trange(len(t)):
        t_im_obsid_gid[i]["gsize"] = gsize_dict[t_im_obsid_gid[i]["gid"]]

    t_im_obsid_gid.sort("id")
    t_im_obsid_gid.remove_columns(["id", "obsid"])

    # t_im_obsid_gid.remove_column("obsid")
    t_imatch = table.hstack((t, t_im_obsid_gid))
    return t_imatch


# change working directory & read catalog (dr11-v1.0)
hostname = os.uname()[1]
if hostname == "alpha":
    os.chdir("/nfsdata/users/cham/projects/lamost/dr11-v1.0/reduced_catalog")
    t = table.Table.read("../catalog/dr11_v1.0_MRS_catalogue.fits.gz")
else:
    os.chdir("/Users/cham/projects/lamost/dr11-v1.0")
    t = table.Table.read("dr11_v1.0_MRS_catalogue.fits.gz")


# match catalog
t_BR = internal_match(preprocessing(t), n_split=3)
t_BR.write("dr11v1.0-BR.fits", overwrite=True)

SNR = 5
t = t_BR[(t_BR["snr_B"] > SNR) | (t_BR["snr_R"] > SNR)]
print(f"{len(t_BR)}->{len(t)}")
t.add_column(table.Column(np.arange(len(t), dtype=int), name="id"))
t_BR_im = group_obsid(t)
t_BR_im.write("dr11v1.0-BR-snr5-im.fits", overwrite=True)
print(f"{len(t_BR)}->{len(t)}->{len(t_BR_im)}")

t_upload = table.Table.read("dr11v1.0-BR-snr5-im[uploaded_v1].fits")
assert np.all(t_BR_im["sobsid"] == t_upload["sobsid"])
assert np.all(t_BR_im["sobsid"] == t["sobsid"])
print(t["sobsid"])
print(t_BR_im["sobsid"])
print(t_upload["sobsid"])

t_BR_im["id", "sobsid", "obsid", "ra", "dec"][t_BR_im["gid"] == 28734]

t_BR_im[
    "rv0_lamost_B",
    "rv0_err_lamost_B",
    "rv1_lamost_B",
    "rv1_err_lamost_B",
    "rv_flag_lamost_B",
    "bad_lamost_B",
    "sobsid",
    "snr_B",
    "rv0_lamost_R",
    "rv0_err_lamost_R",
    "rv1_lamost_R",
    "rv1_err_lamost_R",
    "rv_flag_lamost_R",
    "bad_lamost_R",
    "snr_R",
    "obsid",
    "gp_id",
    "designation",
    "obsdate",
    "lmjd",
    "mjd",
    "planid",
    "spid",
    "fiberid",
    "lmjm",
    "band",
    "ra_obs",
    "dec_obs",
    "gaia_source_id",
    "gaia_g_mean_mag",
    "gaia_bp_mean_mag",
    "gaia_rp_mean_mag",
    "fibertype",
    "offsets",
    "offsets_v",
    "ra",
    "dec",
    "rv0_lamost_BR",
    "rv0_err_lamost_BR",
    "rv1_lamost_BR",
    "rv1_err_lamost_BR",
    "rv_flag_lamost_BR",
    "rv_lasp0",
    "rv_lasp0_err",
    "rv_lasp1",
    "rv_lasp1_err",
    "fibermask",
    "moon_angle",
    "lunardate",
    "moon_flg",
    "subproject",
    "id",
    "gid",
    "gsize",
]
