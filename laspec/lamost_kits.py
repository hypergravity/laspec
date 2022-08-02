"""
This module aims to integrate some useful kits to cope with LAMOST data.

"""

import numpy as np
import joblib

from .mrs import MrsFits
from . import PACKAGE_PATH


PATH_M9WAVEBR = PACKAGE_PATH + "/data/lamost/m9waveBR.dump"


class MrsKit:

    @staticmethod
    def read_multi_spec(fp_list, lmjm_list, rvzp_B_list, rvzp_R_list, wave_interp=None):
        """ read multiple spectra, interpolate to a wavelength grid """
        n_spec = len(lmjm_list)

        wave_obs_list = []
        flux_obs_list = []
        flux_err_obs_list = []
        mask_list = []

        for i_spec in range(n_spec):
            print("reading spectrum [{}/{}] ...".format(i_spec, n_spec))
            this_fp = fp_list[i_spec]
            this_lmjm = lmjm_list[i_spec]
            # read this epoch
            this_flux, this_flux_err, this_mask = MrsKit.read_single_epoch(
                this_fp, this_lmjm, rvzp_B_list, rvzp_R_list,
                wave_interp=wave_interp if wave_interp is not None else MrsKit.load_wave())
            # append data
            flux_obs_list.append(this_flux)
            flux_err_obs_list.append(this_flux_err)
            mask_list.append(this_mask)

        # if interpolated to a wavelength grid, return a regular ndarray
        if wave_interp is None:
            return wave_obs_list, flux_obs_list, flux_err_obs_list
        else:
            return np.array(flux_obs_list), np.array(flux_err_obs_list), np.array(mask_list)

    @staticmethod
    def read_single_epoch(this_fp, this_lmjm, this_rvzp_B=0., this_rvzp_R=0., wave_interp=None):
        """ read a single epoch """
        # read spectra and reduce
        mf = MrsFits(this_fp)
        msB = mf.get_one_spec(lmjm=this_lmjm, band="B")
        msrB = msB.reduce(npix_cushion=70, norm_type="spline", niter=2)
        msR = mf.get_one_spec(lmjm=this_lmjm, band="R")
        msrR = msR.reduce(npix_cushion=70, norm_type="spline", niter=2)
        maskB = (msrB.mask > 0) | (msrB.indcr > 0) | (msrB.flux_norm <= 0) | (msrB.flux_norm >= 3)
        maskR = (msrR.mask > 0) | (msrR.indcr > 0) | (msrR.flux_norm <= 0) | (msrR.flux_norm >= 3)
        # shift rvzp
        wave_B = msrB.wave_rv(rv=-this_rvzp_B)
        wave_R = msrR.wave_rv(rv=-this_rvzp_R)

        # use default wavelength grid
        if wave_interp is None:
            wave_interp = MrsKit.load_wave()

        # interpolate spectrum
        wave_BR = np.hstack((wave_B, wave_R))
        this_flux = np.interp(wave_interp, wave_BR, np.hstack((msrB.flux_norm, msrR.flux_norm)))
        this_flux_err = np.interp(wave_interp, wave_BR, np.hstack((msrB.flux_norm_err, msrR.flux_norm_err)))
        this_mask = np.interp(wave_interp, wave_BR, np.hstack((maskB, maskR))) > 0
        return this_flux, this_flux_err, this_mask

    @staticmethod
    def load_wave():
        """ load MRS wavelength (BR) """
        return joblib.load(PATH_M9WAVEBR)

    @staticmethod
    def ezscatter(a, chunksize=1000, n_jobs=None):
        """ scatter array a to several jobs """
        if isinstance(a, int):
            a = np.arange(a, dtype=int)
        n_el = len(a)
        if n_jobs is not None:
            chunksize = np.int(np.ceil(n_el / n_jobs))
        n_chunks = np.int(np.ceil(n_el / chunksize))
        a_scattered = [a[chunksize * i_chunk:np.min((chunksize * (i_chunk + 1), n_el))] for i_chunk in range(n_chunks)]
        return a_scattered

