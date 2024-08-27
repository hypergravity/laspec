#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 22:08:09 2023

@author: cham
"""
#%%
import os
os.chdir('/nfsdata/users/cham/projects/lamost/dr10v1+dr11v0-reduction')

#%% read dr10 catalog
from astropy import table

t10 = table.Table.read("../dr10-v1.0/catalog/dr10_v1.0_MRS_catalogue.fits.gz")
t11 = table.Table.read("../dr11-v0/catalog-mrs/dr11_v0_MRS_catalogue_q1q2q3.fits.gz")

# lmjm = t10["lmjm"]
#%%
from copy import deepcopy
t = deepcopy(t10)
#%% pre-processing
import os
import joblib
def preprocessing(t, colnames_reference="dr10v1.0-colnames.joblib"):
    if os.path.exists(colnames_reference):
        colnames_reference = joblib.load(colnames_reference)
        print("compare columns with dr10 v1.0:")
        print("columns in dr10 v1.0 but not in this table:")
        print(set(colnames_reference).difference(set(t.colnames)))
        print("columns in this table but not in dr10 v1.0:")
        print(set(t.colnames).difference(set(colnames_reference)))
    else:
        print("skipping comparison ...")
    
    # select single exposure spectra
    t = t[t["coadd"] == 0]
    
    # delete useless columns
    useless_columns = ["mobsid", "uid", "coadd"]
    for colname in useless_columns:
        if colname in t.colnames:
            print(f"remove {colname}")
            t.remove_column(colname)
    
    # map columns
    colname_mapping = {
        'rv_b0': 'rv0_lamost_B',
        'rv_b0_err': 'rv0_err_lamost_B', 
        'rv_b1':'rv1_lamost_B', 
        'rv_b1_err':"rv1_err_lamost_B", 
        'rv_b_flag':'rv_flag_lamost_B', 
        'rv_r0': 'rv0_lamost_R', 
        'rv_r0_err': "rv0_err_lamost_R", 
        'rv_r1': 'rv1_lamost_R',
        'rv_r1_err': 'rv1_err_lamost_R', 
        'rv_r_flag': 'rv_flag_lamost_R',
        'bad_b':'bad_lamost_B', 
        'bad_r': 'bad_lamost_R', 
        'rv_br0':'rv0_lamost_BR',
        'rv_br0_err':'rv0_err_lamost_BR',
        'rv_br1': 'rv1_lamost_BR',
        'rv_br1_err':'rv1_err_lamost_BR', 
        'rv_br_flag':'rv_flag_lamost_BR',
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


#%%
import numpy as np

def internal_match(t, n_split=10):
    
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
    split_size = int(len(lmjm_sorted)/n_split)
    lmjm_edges = [lmjm_sorted[i*split_size] for i in range(n_split)]
    lmjm_edges.append(np.inf)
    
    colnames = t.colnames
    total_count = 0
    xt = []
    for i_split in range(n_split):
        print(f"{i_split}/{n_split}")
        
        lmjm_lo, lmjm_hi = lmjm_edges[i_split:i_split+2]
        this_ind = (lmjm>=lmjm_lo) & (lmjm <lmjm_hi)
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
            this_t_B, this_t_R, 
            keys=["lmjm", "spid", "fiberid"], 
            join_type="outer",
            table_names=['1','2']
        )
        ind_B = ~this_t_BR["obsid_1"].mask
        ind_R = ~this_t_BR["obsid_2"].mask
        for colname in colnames:
            if colname + "_1" in this_t_BR.colnames and colname + "_2" in this_t_BR.colnames:
                
                print(f"combine column: {colname}")
                combined_col = table.Column(
                    data=np.where(ind_B, this_t_BR[colname+"_1"].data, this_t_BR[colname+"_2"].data),
                    name=colname
                )
                this_t_BR.add_column(combined_col)
                this_t_BR.remove_columns([colname + "_1", colname + "_2"])
            else:
                print(f"skip column {colname}")
        
        xt.append(this_t_BR)
        
    return table.vstack(xt)



#%% after topcat

def group_obsid(t, stilts="/home/cham/stilts.jar"):
    import tempfile
    temp_folder = tempfile.gettempdir()
    
    file_obsid_ra_dec = os.path.join(temp_folder, "obsid_ra_dec.fits")
    file_group_results = os.path.join(temp_folder, "group_results.fits")
    
    command = f'java -jar {stilts} tmatch1 matcher=sky params=3 values="ra dec" action=identify in={file_obsid_ra_dec} out={file_group_results}'
    print("Group ra-dec with stilts...")
    print(command)
    
    u_obsid, u_index, u_counts = np.unique(t["obsid"].data, return_index=True, return_counts=True)
    t_uinfo = t["ra", "dec", "obsid"][u_index]
    t_uinfo.write(file_obsid_ra_dec, overwrite=True)

    
    import subprocess
    result = subprocess.run(command, capture_output=True, shell=True)
    assert os.path.exists(file_group_results)
    
    # read results
    t_uresult = table.Table.read(file_group_results)["obsid", "GroupID"]
    n_single_observation = t_uresult["GroupID"].mask.sum()
    t_uresult
    gid = np.copy(t_uresult["GroupID"].data)
    gid[t_uresult["GroupID"].mask] = -1-np.arange(n_single_observation, dtype=np.int32)
    t_uresult.remove_column("GroupID")
    t_uresult.add_column(table.Column(table.Column(data=gid, name="gid")))
    
    # join gid to table
    t_im_gid = table.join(t, t_uresult, keys=["obsid"], join_type="left")
    # find unique gid
    u_gid, u_gsize = np.unique(t_im_gid["gid"].data, return_counts=True)
    t_im_gid_gsize = table.join(t_im_gid, 
        table.Table([table.Column(u_gid, name="gid"), 
                     table.Column(u_gsize, name="gsize"),]),
        keys=["gid"], join_type="left")
    # remove temp file
    os.remove(file_obsid_ra_dec)
    os.remove(file_group_results)
    # add 
    if "id" not in t_im_gid_gsize:
        t_im_gid_gsize.add_column(table.Column(np.arange((t_im_gid_gsize)), dtype=np.int32, name="id"))
    
    return t_im_gid_gsize
    

#%%
t10_BR = internal_match(preprocessing(t10), n_split=10)
t11_BR = internal_match(preprocessing(t11), n_split=1)
t10_BR.write("dr10v1.0-BR.fits", overwrite=True)
t11_BR.write("dr11v0-BR.fits", overwrite=True)

t10_11_BR = table.vstack([t10_BR, t11_BR])
t10_11_BR = t10_11_BR[(t10_11_BR["snr_B"]>5) | (t10_11_BR["snr_R"]>5)]

t10_11_BR_im = group_obsid(t10_11_BR)
t10_11_BR_im.write("dr10-11-BR-snr5-im.fits", overwrite=True)
