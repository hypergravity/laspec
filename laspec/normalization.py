# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from scipy.optimize import minimize

from .extern.interpolate import SmoothSpline


def normalize_spectrum_null(wave):
    return np.ones_like(wave) * np.nan, np.ones_like(wave) * np.nan


def normalize_spectrum(wave, flux, norm_range, dwave,
                       p=(1E-6, 1E-6), q=0.5, ivar=None, eps=1e-10,
                       rsv_frac=1.):
    """ A double smooth normalization of a spectrum

    Converted from Chao Liu's normSpectrum.m
    Updated by Bo Zhang

    Parameters
    ----------
    wave: ndarray (n_pix, )
        wavelegnth array
    flux: ndarray (n_pix, )
        flux array
    norm_range: tuple
        a tuple consisting (wave_start, wave_stop)
    dwave: float
        binning width
    p: tuple of 2 ps
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    q: float in range of [0, 100]
        percentile, between 0 and 1
    ivar: ndarray (n_pix, ) | None
        ivar array, default is None
    eps: float
        the ivar threshold
    rsv_frac: float
        the fraction of pixels reserved in terms of std. default is 3.

    Returns
    -------
    flux_norm: ndarray
        normalized flux
    flux_cont: ndarray
        continuum flux

    Example
    -------
    >>> flux_norm, flux_cont = normalize_spectrum(
    >>>     wave, flux, (4000., 8000.), 100., p=(1E-8, 1E-7), q=0.5,
    >>>     rsv_frac=2.0)

    """
    print("This function is DEPRECATED, use *normalize_spectrum_general* instead!")
    if np.sum(np.logical_and(np.isfinite(flux), flux > 0)) <= 100:
        return normalize_spectrum_null(wave)

    if ivar is not None:
        # ivar is set
        ivar = np.where(np.logical_or(wave < norm_range[0],
                                      wave > norm_range[1]), 0, ivar)
        ivar = np.where(ivar <= eps, eps, ivar)
        # mask = ivar <= eps
        var = 1. / ivar
    else:
        # default config is even weight
        var = np.ones_like(flux)

    # wave = wave[~mask]
    # flux = flux[~mask]

    # check q region
    assert 0. < q < 1.

    # n_iter = len(p)
    n_bin = np.int(np.fix(np.diff(norm_range) / dwave) + 1)
    wave1 = norm_range[0]

    # SMOOTH 1
    # print(wave.shape, flux.shape, var.shape)
    if ivar is not None:
        ind_good_init = 1. * (ivar > 0.) * (flux > 0.)
    else:
        ind_good_init = 1. * (flux > 0.)
    ind_good_init = ind_good_init.astype(np.bool)
    # print("@Cham: sum(ind_good_init)", np.sum(ind_good_init))

    flux_smoothed1 = SmoothSpline(wave[ind_good_init], flux[ind_good_init],
                                  p=p[0], var=var[ind_good_init])(wave)
    dflux = flux - flux_smoothed1

    # collecting continuum pixels --> ITERATION 1
    ind_good = np.zeros(wave.shape, dtype=np.bool)
    for i_bin in range(n_bin):
        ind_bin = np.logical_and(wave > wave1 + (i_bin - 0.5) * dwave,
                                 wave <= wave1 + (i_bin + 0.5) * dwave)
        if np.sum(ind_bin > 0):
            # median & sigma
            bin_median = np.median(dflux[ind_bin])
            bin_std = np.median(np.abs(dflux - bin_median))
            # within 1 sigma with q-percentile
            ind_good_ = ind_bin * (
                    np.abs(dflux - np.nanpercentile(dflux[ind_bin], q * 100.)) < (
                    rsv_frac * bin_std))
            ind_good = np.logical_or(ind_good, ind_good_)

    ind_good = np.logical_and(ind_good, ind_good_init)
    # assert there is continuum pixels
    try:
        assert np.sum(ind_good) > 0
    except AssertionError:
        Warning("@Keenan.normalize_spectrum(): unable to find continuum!")
        ind_good = np.ones(wave.shape, dtype=np.bool)

    # SMOOTH 2
    # continuum flux
    flux_smoothed2 = SmoothSpline(
        wave[ind_good], flux[ind_good], p=p[1], var=var[ind_good])(wave)
    # normalized flux
    flux_norm = flux / flux_smoothed2

    return flux_norm, flux_smoothed2


def normalize_spectrum_spline(wave, flux, p=1E-6, q=0.5, lu=(-1, 3), binwidth=30, niter=3):
    """ A double smooth normalization of a spectrum

    Converted from Chao Liu's normSpectrum.m
    Updated by Bo Zhang

    Parameters
    ----------
    wave: ndarray (n_pix, )
        wavelegnth array
    flux: ndarray (n_pix, )
        flux array
    p: float
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    q: float in range of [0, 1]
        percentile, between 0 and 1
    lu: float tuple
        the lower & upper exclusion limits
    binwidth: float
        width of each bin
    niter: int
        number of iterations
    Returns
    -------
    flux_norm: ndarray
        normalized flux
    flux_cont: ndarray
        continuum flux

    Example
    -------
    >>> fnorm, fcont=normalize_spectrum_spline(
    >>>     wave, flux, p=1e-6, q=0.6, binwidth=200, lu=(-1,5), niter=niter)

    """
    if np.sum(np.logical_and(np.isfinite(flux), flux > 0)) <= 10:
        return normalize_spectrum_null(wave)

    _wave = np.copy(wave)
    _flux = np.copy(flux)
    ind_finite = np.isfinite(flux)
    wave = _wave[ind_finite]
    flux = _flux[ind_finite]
    _flux_norm = np.copy(_flux)
    _flux_cont = np.copy(_flux)

    # default config is even weight
    var = np.ones_like(flux)

    # check q region
    # assert 0. <= q <= 1.

    nbins = np.int(np.ceil((wave[-1] - wave[0]) / binwidth) + 1)
    bincenters = np.linspace(wave[0], wave[-1], nbins)

    # iteratively smoothing
    ind_good = np.isfinite(flux)
    for _ in range(niter):

        flux_smoothed1 = SmoothSpline(wave[ind_good], flux[ind_good], p=p, var=var[ind_good])(wave)
        # residual
        res = flux - flux_smoothed1

        # determine sigma
        stdres = np.zeros(nbins)
        for ibin in range(nbins):
            ind_this_bin = ind_good & (np.abs(wave - bincenters[ibin]) <= binwidth)
            if 0 <= q <= 0:
                stdres[ibin] = np.std(
                    res[ind_this_bin] - np.percentile(res[ind_this_bin], 100 * q))
            else:
                stdres[ibin] = np.std(res[ind_this_bin])
        stdres_interp = interp1d(bincenters, stdres, kind="linear")(wave)
        if 0 <= q <= 1:
            res1 = (res - np.percentile(res, 100 * q)) / stdres_interp
        else:
            res1 = res / stdres_interp
        ind_good = ind_good & (res1 > lu[0]) & (res1 < lu[1])

        # assert there is continuum pixels
        try:
            assert np.sum(ind_good) > 0
        except AssertionError:
            Warning("@normalize_spectrum_iter: unable to find continuum!")
            ind_good = np.ones(wave.shape, dtype=np.bool)

    # final smoothing
    flux_smoothed2 = SmoothSpline(
        wave[ind_good], flux[ind_good], p=p, var=var[ind_good])(wave)
    # normalized flux
    flux_norm = flux / flux_smoothed2

    _flux_norm[ind_finite] = flux_norm
    _flux_cont[ind_finite] = flux_smoothed2
    return _flux_norm, _flux_cont


def normalize_spectra_block(wave, flux_block, norm_range, dwave,
                            p=(1E-6, 1E-6), q=0.5, ivar_block=None, eps=1e-10,
                            rsv_frac=3., n_jobs=1, verbose=10):
    """ normalize multiple spectra using the same configuration
    This is specially designed for TheKeenan

    Parameters
    ----------
    wave: ndarray (n_pix, )
        wavelegnth array
    flux_block: ndarray (n_obs, n_pix)
        flux array
    norm_range: tuple
        a tuple consisting (wave_start, wave_stop)
    dwave: float
        binning width
    p: tuple of 2 ps
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    q: float in range of [0, 100]
        percentile, between 0 and 1
    ivar_block: ndarray (n_pix, ) | None
        ivar array, default is None
    eps: float
        the ivar threshold
    rsv_frac: float
        the fraction of pixels reserved in terms of std. default is 3.
    n_jobs: int
        number of processes launched by joblib
    verbose: int / bool
        verbose level

    Returns
    -------
    flux_norm_block: ndarray
        normalized flux

    flux_cont_block: ndarray
        continuum flux

    """
    print("This function is DEPRECATED, use *normalize_spectrum_general* instead!")
    if ivar_block is None:
        ivar_block = np.ones_like(flux_block)

    if flux_block.ndim == 1:
        flux_block.reshape(1, -1)
    n_spec = flux_block.shape[0]

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(normalize_spectrum)(
            wave, flux_block[i], norm_range, dwave, p=p, q=q,
            ivar=ivar_block[i], eps=eps, rsv_frac=rsv_frac)
        for i in range(n_spec))

    # unpack results
    flux_norm_block = []
    flux_cont_block = []
    for result in results:
        flux_norm_block.append(result[0])
        flux_cont_block.append(result[1])

    return np.array(flux_norm_block), np.array(flux_cont_block)


def normalize_spectrum_general(wave, flux, norm_type="spline",
                               deg=4, lu=(-1, 4), q=0.5, binwidth=100., niter=3, pw=1., p=1e-6):
    """ poly / spline normalization
    spline --> normalize_spectrum_spline: dict(p=1e-6, q=0.5, lu=(-2, 3), binwidth=100., niter=3)
    poly   --> normalize_spectrum_poly: (deg=4, lu=(-2, 3), q=0.5, binwidth=100., niter=3, pw=1.)

    Parameters
    ----------
    wave: array
        wavelength
    flux: array
        flux
    norm_type: str
        "spline" / "poly"
    deg: int
        poly deg
    lu: tuple
        defaults to (-1, 4), the data below 1 sigma and above 4 sigma will be excluded
    q: float
        percentile, default is 0.5,
    binwidth:
        bin width, detault to 100.
    niter:
        number of iterations, detaults to 3
    pw:
        power of residuals, defaults to 1, only used when norm_type=="poly"
    p:
        spline smoothness, defaults to 1e-6
    """

    if norm_type == "poly":
        flux_norm, flux_cont = normalize_spectrum_poly(
            wave, flux, deg=deg, lu=lu, q=q, binwidth=binwidth, niter=niter, pw=pw)
    elif norm_type == "spline":
        flux_norm, flux_cont = normalize_spectrum_spline(
            wave, flux, p=p, lu=lu, q=q, binwidth=binwidth, niter=niter)
    else:
        assert norm_type in ("poly", "spline")
    return flux_norm, flux_cont


def normalize_spectrum_poly(wave, flux, deg=10, pw=1., lu=(-1, 4), q=0.5, binwidth=100., niter=3):
    """ normalize spectrum using polynomial """
    if np.sum(np.logical_and(np.isfinite(flux), flux > 0)) <= 10:
        return normalize_spectrum_null(wave)

    # check q region
    assert 0. <= q <= 1.

    nbins = np.int(np.ceil((wave[-1] - wave[0]) / binwidth) + 1)
    bincenters = np.linspace(wave[0], wave[-1], nbins)

    # iteratively smoothing
    ind_good = np.ones_like(flux, dtype=bool)

    for _ in range(niter):
        # poly smooth
        flux_smoothed1 = PolySmooth(wave[ind_good], flux[ind_good], deg=deg, pw=pw)(wave)
        # residual
        res = flux - flux_smoothed1
        # determine sigma
        stdres = np.zeros(nbins)
        for ibin in range(nbins):
            ind_this_bin = ind_good & (
                    np.abs(wave - bincenters[ibin]) <= binwidth)
            if 0 <= q <= 0:
                stdres[ibin] = np.std(
                    res[ind_this_bin] - np.percentile(res[ind_this_bin], 100 * q))
            else:
                stdres[ibin] = np.std(res[ind_this_bin])
        stdres_interp = interp1d(bincenters, stdres, kind="linear")(wave)
        if 0 <= q <= 1:
            res1 = (res - np.percentile(res, 100 * q)) / stdres_interp
        else:
            res1 = res / stdres_interp
        ind_good = ind_good & (res1 > lu[0]) & (res1 < lu[1])

        # assert there is continuum pixels
        try:
            assert np.sum(ind_good) > 0
        except AssertionError:
            Warning("@normalize_spectrum_iter: unable to find continuum!")
            ind_good = np.ones(wave.shape, dtype=np.bool)

    # final smoothing
    flux_smoothed2 = PolySmooth(wave[ind_good], flux[ind_good], deg=deg, pw=pw)(wave)

    # normalized flux
    flux_norm = flux / flux_smoothed2

    return flux_norm, flux_smoothed2


class PolySmooth:
    def __init__(self, x, y, deg=4, pw=2.):
        # determine scales
        x_pct = np.percentile(x, q=[16, 50, 84])
        y_pct = np.percentile(y, q=[16, 50, 84])
        self.x_mean = x_pct[1]
        self.y_mean = y_pct[1]
        self.x_scale = (x_pct[2] - x_pct[0]) / 2
        self.y_scale = (y_pct[2] - y_pct[0]) / 2
        # scale data
        x_scaled = (x - self.x_mean) / self.x_scale
        y_scaled = (y - self.y_mean) / self.y_scale
        # optimization
        result = minimize(cost_poly, x0=np.zeros(deg+1), args=(x_scaled, y_scaled, pw), method="powell")
        self.p = result["x"]

    def __call__(self, x):
        """ prediction """
        return np.polyval(self.p, (x - self.x_mean) / self.x_scale) * self.y_scale + self.y_mean


def cost_poly(p, x, y, pw=2.):
    res = np.square(np.polyval(p, x) - y)
    return np.sum(res[np.isfinite(res)] ** (pw / 2))


if __name__ == '__main__':
    pass
    # test_normaliza_spectra_block()
