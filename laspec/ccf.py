#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""


import numpy as np
from astropy import constants
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator, interp1d


def sinebell(n=1000, index=0.5):
    """ sine bell to left & right end of spectra """
    return np.sin(np.linspace(0,np.pi,n))**index


def sinebell_like(x, index=0.5):
    return sinebell(len(x), index=index)


def test_sinebell():
    plt.figure();
    plt.plot(sinebell(4000,1))
    plt.plot(sinebell(4000,1/2))
    plt.plot(sinebell(4000,1/3))
    return


def test_sinebell2():
    """ load data """
    wave, flux, flux_err = np.loadtxt('/hydrogen/projects/song/delCep_order20.dat').T
    # flux_sine = flux - flux.mean()
    flux_sine = 1-flux
    flux_sine = flux_sine * sinebell_like(flux, 1.0)
    
    plt.figure()
    plt.plot(wave, (flux-1))
    plt.plot(wave, flux_sine)
    # plot(wave, flux_err)
    return wave, flux_sine


def xcorr_rvgrid(wave_obs, flux_obs, wave_mod, flux_mod, mask_obs=None, rv_grid=np.arange(-500, 510, 10), sinebell_idx=0):
    """ a naive cross-correlation method
    Interpolate a model spectrum with different RV and cross-correlate with
    the observed spectrum, return the CCF on the RV grid.
    
    wave_obs: array
        wavelength of observed spectrum (normalized)
    flux_obs: array
        flux of observed spectrum
    wave_mod: array
        wavelength of model spectrum (normalized)
    flux_mod:
        flux of model spectrum
    mask_obs:
        True for bad pixels
    rv_grid:
        km/s RV grid
    
    """
    wave_obs = np.asarray(wave_obs)
    flux_obs = np.asarray(flux_obs)
    wave_mod = np.asarray(wave_mod)
    flux_mod = np.asarray(flux_mod)
    
    if mask_obs is None:
        mask_obs = np.logical_not(np.isfinite(flux_obs))
    else:
        mask_obs = np.asarray(mask_obs, dtype=bool) | np.logical_not(np.isfinite(flux_obs))

    rv_grid = np.asarray(rv_grid)
    z_grid = rv_grid / constants.c.value * 1000
    nz = len(z_grid)
    
    # p = PchipInterpolator(wave_mod, flux_mod, extrapolate=False)
    p = interp1d(wave_mod, flux_mod, kind="linear", bounds_error=False, fill_value=np.nan)
    # p = Interp1q(wave_mod, flux_mod)
    
    wave_mod_interp = wave_obs.reshape(1, -1) / (1+z_grid.reshape(-1, 1))
    flux_mod_interp = p(wave_mod_interp)
    mask_bad = np.logical_not(np.isfinite(flux_mod_interp)) | mask_obs
    
    # use masked array
    xmod = np.ma.MaskedArray(flux_mod_interp, mask_bad)
    xobs = np.ma.MaskedArray(np.repeat(flux_obs.reshape(1, -1), nz, axis=0), mask_bad)
    xmod -= np.ma.mean(xmod)
    xobs -= np.ma.mean(xobs)
    
    ccf_grid = np.ma.sum(xmod*xobs, axis=1) / np.ma.sqrt(np.ma.sum(xmod*xmod, axis=1)*np.ma.sum(xobs*xobs, axis=1))
    #chi2_grid = 0.5*np.ma.sum((xmod-xobs)**2, axis=1) 
    return rv_grid, ccf_grid.data#, chi2_grid.data
    

def test_xcorr_rvgrid():
    """ load data """
    wave, flux, flux_err = np.loadtxt('/hydrogen/projects/song/delCep_order20.dat').T
    flux_sine = 1-flux
    flux_sine = flux_sine * sinebell_like(flux, 1.0)
    
    
    flux_obs = flux_sine+np.random.randn(*flux_sine.shape)*0.5
    wave_mod = wave
    wave_obs = wave
    flux_mod = flux_sine
    rv_grid=np.linspace(-500, 500, 1000)
    # z_grid = rv_grid / constants.c.value * 1000

    ccfv = xcorr_rvgrid(wave_obs, flux_obs,
                        wave_mod, flux_mod, mask_obs=None, 
                        rv_grid=rv_grid, 
                        sinebell_idx=1)
    plt.figure();
    plt.subplot(211)
    plt.plot(ccfv[0], ccfv[1], 's-')
    #plt.plot(ccfv[0], np.exp(-ccfv[2]+np.max(ccfv[2])), 's-')
    #plt.plot(ccfv[0], (-ccfv[2]+np.mean(ccfv[2]))/np.std(ccfv[2]), 's-')
    plt.hlines(ccfv[1].std()*np.array([0, 3, 5]), -500, 500)
    plt.subplot(212)
    plt.plot(wave, flux_obs)
    plt.plot(wave, flux_sine)
    
    return


def xcorr(x1, x2, w1=None):
    """ Pearson correlation coef """
    x1n = x1 - np.mean(x1)
    x2n = x2 - np.mean(x2)
    return np.dot(x1n, x2n) / np.sqrt(np.dot(x1n, x1n) * np.dot(x2n, x2n))
    

def test_lmfit():
    """ load data """
    wave, flux, flux_err = np.loadtxt('/hydrogen/projects/song/delCep_order20.dat').T
    flux_sine = 1-flux
    flux_sine = flux_sine * sinebell_like(flux, 1.0)

    flux_obs = flux_sine+np.random.randn(*flux_sine.shape)*0.1
    wave_mod = wave
    wave_obs = wave
    flux_mod = flux_sine
    rv_grid=np.linspace(-500, 500, 1000)
    # z_grid = rv_grid / constants.c.value * 1000

    ccfv = xcorr_rvgrid(wave_obs, flux_obs,
                        wave_mod, flux_mod, mask_obs=None, 
                        rv_grid=rv_grid, 
                        sinebell_idx=1)
    
    # Gaussian fit using LMFIT
    from lmfit import Model
    from lmfit.models import GaussianModel

    mod = GaussianModel()
    x, y = ccfv[0], ccfv[1]
    #pars = mod.guess(y, x=x)
    out = mod.fit(y, None, x=x, method="least_squares")
    #out = mod.fit(y, pars, x=x, method="leastsq")

    plt.figure()
    plt.plot(x, y)
    plt.plot(x, out.best_fit)
    print(out.fit_report())


if __name__ == "__main__":

    test_xcorr_rvgrid()
