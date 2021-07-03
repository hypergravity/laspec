import numpy as np
from astropy import table
from laspec.snstat import eval_xi_1
from scipy.optimize import minimize


def calibrate_rvzp(rvobs, rvobs_err, rvref, rvref_err, spid, lmjm, rvlabels=["B", "R", "Rm"], ncommon_min=5,
                   rvobs_err_min=1., verbose=True):
    """ A simplified version of Zhang et al. (2021)

    Parameters
    ----------
    rvobs:
        observed RVs, (nobs,) or (nobs, nrv), where nrv denotes the number of RV sources.
        e.g., for RV from B R and Rm nrv=3.
    rvobs_err:
        error of observed RVs, (nobs,) or (nobs, nrv)
    rvref:
        reference RVs, (nobs,) or (nobs, nrv)
    rvref_err:
        error of reference RVs, (nobs,) or (nobs, nrv)
    spid:
        spectrograph ID
    lmjm:
        Local Modified Julian Minute
    rvlabels: list
        labels of RVs
    ncommon_min:
        if common stars more than ncommon_min, evaluate RVZPs
    rvobs_err_min:
        a noise floor due to wavelength calibration and instrumentation defects. defaults to 1. in unit of km/s
    verbose:
        if True, print details

    Returns
    -------

    """
    if rvobs.ndim == 2:
        nrv = rvobs.shape[1]
    else:
        nrv = 1
        rvobs = rvobs[:, None]
        rvobs_err = rvobs_err[:, None]

    # extract unique SEUs
    spid_lmjm = np.array([spid, lmjm]).T
    u_spid_lmjm, c_spid_lmjm = np.unique(spid_lmjm, return_counts=True, axis=0)
    print("@RVZP: N_SEU = {}, NSTAR_MED = {}".format(len(u_spid_lmjm), np.median(c_spid_lmjm)))
    # construct SEU table
    tseu = table.Table(data=[u_spid_lmjm, c_spid_lmjm, u_spid_lmjm[:, 0], u_spid_lmjm[:, 1]],
                       names=["u_spid_lmjm", "c_spid_lmjm", "u_spid", "u_lmjm"])
    nseu = len(tseu)
    # evaluate RVZP, loop over SEUs
    rvzp_list = []
    for iseu in range(nseu):
        print("@RVZP: processing SEU {}/{} ".format(iseu, nseu), end="")
        this_rvzp = dict()
        # iseu = 0
        this_spid = tseu["u_spid"][iseu]
        this_lmjm = tseu["u_lmjm"][iseu]
        this_ind = np.where((spid == this_spid) & (lmjm == this_lmjm))[0]
        this_nobs = len(this_ind)
        # rvref
        this_rvref = rvref[this_ind]
        this_rvref_err = rvref_err[this_ind]
        this_nref = np.sum(np.isfinite(this_rvref))
        # counts
        this_rvzp["nobs"] = this_nobs
        this_rvzp["nref"] = this_nref

        for irv in range(nrv):
            print(rvlabels[irv], end=" ")
            # rvobs
            this_rvobs = rvobs[this_ind, irv]
            this_rvobs_err = rvobs_err[this_ind, irv]

            this_nobs_fnt = np.sum(np.isfinite(this_rvobs) & np.isfinite(this_rvobs_err))
            ind_common = np.isfinite(this_rvref) & np.isfinite(this_rvobs) & np.isfinite(this_rvobs_err)
            this_ncommon = np.sum(ind_common)
            this_rvzp["nobs_{}".format(rvlabels[irv])] = this_nobs_fnt
            this_rvzp["ncommon_{}".format(rvlabels[irv])] = this_ncommon
            if this_ncommon >= ncommon_min:
                # do the rvzp calculation
                res = minimize(rvzp0cost, x0=np.nanmedian(this_rvref - this_rvobs), method="Nelder-Mead",
                               args=(this_rvref[ind_common], this_rvref_err[ind_common], this_rvobs[ind_common],
                                     this_rvobs_err[ind_common], rvobs_err_min))

                this_rvzp["rvzp_{}".format(rvlabels[irv])] = res["x"][0]
                this_rvzp["rvzp_err_{}".format(rvlabels[irv])] = \
                    np.ptp(np.nanpercentile(this_rvobs[ind_common] + res["x"][0] - this_rvref[ind_common], [16, 84])) \
                    / (2 * eval_xi_1(this_ncommon) * np.sqrt(this_ncommon))
        if verbose:
            print(this_rvzp, end="\n")
        else:
            print("", end="\n")

        rvzp_list.append(this_rvzp)
    trvzp = table.Table(rvzp_list)
    # append rvzp columns
    tseu = table.hstack([tseu, trvzp])
    # reconstruct rvzp
    print("@RVZP: reconstruct mapping index from original catalog to tseu ...")
    ind_map = np.zeros(len(rvobs), dtype=int)
    for iseu in range(len(tseu)):
        ind_map[(spid == tseu["u_spid"][iseu]) & (lmjm == tseu["u_lmjm"][iseu])] = iseu
    return tseu, ind_map


def rvzp0cost(rvzp, this_rvref, this_rvref_err, this_rvobs, this_rvobs_err, rvobs_err_min):
    return np.nansum(np.abs(this_rvobs + rvzp - this_rvref) / np.sqrt(
        rvobs_err_min ** 2 + this_rvobs_err ** 2 + this_rvref_err ** 2))

