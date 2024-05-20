"""
A simplified version of RVZP calculation algorithm (Zhang et al. 2021).
ADS Link: https://ui.adsabs.harvard.edu/abs/2021ApJS..256...14Z/abstract
"""

__all__ = ["calibrate_rvzp"]

import numpy as np
import numpy.typing as npt
from astropy import table
from scipy.optimize import minimize

from .snstat import eval_xi_1  # to scale small number statistics


def calibrate_rvzp(
    rvobs: npt.NDArray,
    rvobs_err: npt.NDArray,
    rvref: npt.NDArray,
    rvref_err: npt.NDArray,
    spid: npt.NDArray,
    lmjm: npt.NDArray,
    rvlabels=["B", "R", "Rm"],
    ncommon_min: int = 5,
    rv_internal_error: float = 1.0,
    verbose: bool = True,
    debug: bool = False,
) -> tuple[table.Table, npt.NDArray]:
    """A simplified version of RVZP calculation algorithm (Zhang et al. 2021).

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
    rv_internal_error:
        a noise floor due to wavelength calibration and instrumentation defects. defaults to 1. in unit of km/s
    verbose:
        if True, print details
    debug : bool
        if True, return tseu and exit

    Returns
    -------
    tuple:
        tseu, ind_map

    Notes
    -----
    make sure that the input arrays are not masked

    References
    ----------
    https://ui.adsabs.harvard.edu/abs/2021ApJS..256...14Z/abstract

    Examples
    --------
    >>> tseu, ind_map = calibrate_rvzp(
    >>>     np.array(t["rv1_B", "rv1_R"].to_pandas()),
    >>>     np.array(t["rv1_err_B", "rv1_err_R", ].to_pandas()),
    >>>     np.where(m9["dr2_radial_velocity_gedr3"].mask, np.nan, m9["dr2_radial_velocity_gedr3"].data),
    >>>     np.where(m9["dr2_radial_velocity_error_gedr3"].mask, np.nan, m9["dr2_radial_velocity_error_gedr3"].data),
    >>>     m9["spid"],
    >>>     m9["lmjm"],
    >>>     rvlabels=["B", "R"],
    >>>     ncommon_min=5,
    >>>     rv_internal_error=1.,
    >>>     verbose=True,
    >>>     debug=False
    >>> )

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
    print(
        "@RVZP: N_SEU = {}, NSTAR_MED = {}".format(
            len(u_spid_lmjm), np.median(c_spid_lmjm)
        )
    )
    # construct SEU table
    tseu = table.Table(
        data=[u_spid_lmjm, c_spid_lmjm, u_spid_lmjm[:, 0], u_spid_lmjm[:, 1]],
        names=["u_spid_lmjm", "c_spid_lmjm", "u_spid", "u_lmjm"],
    )
    if debug:
        return tseu

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

            this_nobs_fnt = np.sum(
                np.isfinite(this_rvobs) & np.isfinite(this_rvobs_err)
            )
            ind_common = (
                np.isfinite(this_rvref)
                & np.isfinite(this_rvobs)
                & np.isfinite(this_rvobs_err)
            )
            this_ncommon = np.sum(ind_common)
            this_rvzp["nobs_{}".format(rvlabels[irv])] = this_nobs_fnt
            this_rvzp["ncommon_{}".format(rvlabels[irv])] = this_ncommon
            if this_ncommon >= ncommon_min:
                # do the rvzp calculation
                res = minimize(
                    rvzp0cost,
                    x0=np.nanmedian(this_rvref - this_rvobs),
                    method="Nelder-Mead",
                    args=(
                        this_rvref[ind_common],
                        this_rvref_err[ind_common],
                        this_rvobs[ind_common],
                        this_rvobs_err[ind_common],
                        rv_internal_error,
                    ),
                )

                this_rvzp["rvzp_{}".format(rvlabels[irv])] = res["x"][0]
                this_rvzp["rvzp_err_{}".format(rvlabels[irv])] = np.ptp(
                    np.nanpercentile(
                        this_rvobs[ind_common] + res["x"][0] - this_rvref[ind_common],
                        [16, 84],
                    )
                ) / (2 * eval_xi_1(this_ncommon) * np.sqrt(this_ncommon))
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
        print("@RVZP: map SEU [{}/{}] ...".format(iseu, len(tseu)))
        ind_map[(spid == tseu["u_spid"][iseu]) & (lmjm == tseu["u_lmjm"][iseu])] = iseu
    return tseu, ind_map


def rvzp0cost(
    rvzp: npt.NDArray,
    this_rvref: npt.NDArray,
    this_rvref_err: npt.NDArray,
    this_rvobs: npt.NDArray,
    this_rvobs_err: npt.NDArray,
    rv_internal_error: float,
) -> float:
    """Cost function for RVZP calculation."""
    return np.nansum(
        np.abs(this_rvobs + rvzp - this_rvref)
        / np.sqrt(rv_internal_error**2 + this_rvobs_err**2 + this_rvref_err**2)
    )
