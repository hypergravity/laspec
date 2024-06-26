"""
RVM without weights.


Vectorized functions
--------------------
xcorr_spec_vectorized(rv_grid, wave_obs, flux_obs, wave_mod, flux_mod)


Un-vectorized functions
-----------------------
cov(x1, x2)
├── xcorr(x1, x2)
|    ├── xcorr_spec(rv, wave_obs, flux_obs, wave_mod, flux_mod) -> float
|    |   ├── xcorr_spec_cost(rv, wave_obs, flux_obs, wave_mod, flux_mod)-> float
|    |   ├── xcorr_spec_rvgrid
|    |       ├── RVM.ccf_1mod
|    |       ├── RVM.measure
|    |       ├── RVM.measure_pw
|    ├── xcorr_spec_twin(rv1, drv, eta, wave_obs, flux_obs, wave_mod, flux_mod) -> float
|    ├── xcorr_spec_twin_gamma
|    ├── nxcorr_spec_twin_gamma
|    ├── nxcorr_spec_twin_gamma_BR
|    ├── xcorr_spec_twin_gamma_rv1grid
|        ├── No usage found
|    ├── xcorr_spec_binary(rv1, rv2, eta, wave_obs, flux_obs, wave_mod1, flux_mod1, wave_mod2, flux_mod2) -> float
|        ├── xcorr_spec_binary_cost(rv1_rv2_eta, wave_obs, flux_obs, wave_mod1, flux_mod1, wave_mod2, flux_mod2, eta_lim, drvmax) -> float
|            ├── xcorr_spec_binary_rv2grid

"""

__all__ = ["RVM", "SOL_kms"]

import datetime
import os
import warnings
from collections import OrderedDict
from typing import Optional, Union, Iterable

import joblib
import numpy as np
import numpy.typing as npt
from astropy import constants
from matplotlib import pyplot as plt
from scipy.optimize import minimize, OptimizeResult

from .mrs import MrsFits

SOL_kms = constants.c.value / 1000.0


# ================================================
# sine bell curves, from Tonry & Davis (1979)
# ================================================
def sinebell(n: int = 1000, index: float = 0.5) -> npt.NDArray:
    """sine bell to left & right end of spectra"""
    return np.sin(np.linspace(0, np.pi, n)) ** index


def sinebell_like(x, index: float = 0.5) -> npt.NDArray:
    return sinebell(len(x), index=index)


def test_sinebell() -> None:
    plt.figure()
    plt.plot(sinebell(4000, 1))
    plt.plot(sinebell(4000, 1 / 2))
    plt.plot(sinebell(4000, 1 / 3))
    return


def test_sinebell2() -> tuple[npt.NDArray, npt.NDArray]:
    """load data"""
    wave, flux, flux_err = np.loadtxt("/hydrogen/projects/song/delCep_order20.dat").T
    # flux_sine = flux - flux.mean()
    flux_sine = 1 - flux
    flux_sine = flux_sine * sinebell_like(flux, 1.0)

    plt.figure()
    plt.plot(wave, (flux - 1))
    plt.plot(wave, flux_sine)
    # plot(wave, flux_err)
    return wave, flux_sine


# ================================================
# CCF related functions
# ================================================
def cov(x1: npt.NDArray, x2: npt.NDArray) -> npt.NDArray:
    """Evaluate covariance."""
    return np.mean((x1 - np.mean(x1)) * (x2 - np.mean(x2)))


def xcorr(x1: npt.NDArray, x2: npt.NDArray) -> float:
    """Evaluate cross-correlation."""
    return cov(x1, x2) / np.sqrt(cov(x1, x1) * cov(x2, x2))


def xcorr_spec(
    rv: float,
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
) -> float:
    """Evaluate cross-correlation of two spectra."""
    flux_mod_interp = np.interp(wave_obs, wave_mod * (1 + rv / SOL_kms), flux_mod)
    return xcorr(flux_obs, flux_mod_interp)


def xcorr_spec_cost(
    rv: float,
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
) -> float:
    """Negative `xcorr_spec`, used as cost function for minimization"""
    return -xcorr_spec(rv, wave_obs, flux_obs, wave_mod, flux_mod)


def xcorr_spec_vectorized(
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
    rv_grid: npt.NDArray,
) -> npt.NDArray:
    """Vectorized cross-correlation at an `rv_grid`."""
    # determine shape
    npix = len(wave_obs)
    nmod = flux_mod.shape[0]
    nrv = len(rv_grid)
    # make model cube
    flux_mod_interp = np.zeros((nmod, nrv, npix), dtype=float)
    for imod in range(nmod):
        for irv in range(nrv):
            flux_mod_interp[imod, irv, :] = np.interp(
                wave_obs, wave_mod * (1 + rv_grid[irv] / SOL_kms), flux_mod[imod]
            )
    mean_obs = np.mean(flux_obs)
    mean_mod = np.mean(flux_mod_interp, axis=2)
    res0 = flux_obs - mean_obs
    res1 = flux_mod_interp - mean_mod[:, :, None]
    # underscore means it is not normalized
    _cov00 = np.sum(res0**2.0)  # var(F, F) float
    _cov11 = np.sum(res1**2.0, axis=2)  # var(G, G)  (nmod, nrv)
    _cov01 = np.sum(res0.reshape(1, 1, -1) * res1, axis=2)  # cov(F, G) (nmod, nrv)
    ccf_grid = _cov01 / np.sqrt(_cov00) / np.sqrt(_cov11)
    return ccf_grid


def xcorr_spec_twin(
    rv1: float,
    drv: float,
    eta: float,
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
) -> float:
    """Evaluate cross-correlation of two spectra (twin case) with luminosity ratio `eta`."""
    flux_mod_interp = np.interp(
        wave_obs, wave_mod * (1 + rv1 / SOL_kms), flux_mod
    ) + eta * np.interp(wave_obs, wave_mod * (1 + (rv1 + drv) / SOL_kms), flux_mod)
    return xcorr(flux_obs, flux_mod_interp)


def xcorr_spec_binary(
    rv1: float,
    rv2: float,
    eta: float,
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod1: npt.NDArray,
    flux_mod1: npt.NDArray,
    wave_mod2: npt.NDArray,
    flux_mod2: npt.NDArray,
) -> float:
    """Evaluate cross-correlation of two spectra (binary case)."""
    # combine two templates
    flux_mod_interp = np.interp(
        wave_obs, wave_mod1 * (1 + rv1 / SOL_kms), flux_mod1
    ) + eta * np.interp(wave_obs, wave_mod2 * (1 + rv2 / SOL_kms), flux_mod2)
    return xcorr(flux_obs, flux_mod_interp)


def derive_rv2(
    gamma: Union[float, npt.NDArray],
    q: Union[float, npt.NDArray],
    rv1: Union[float, npt.NDArray],
) -> Union[float, npt.NDArray]:
    """
    Derive `rv2` from `gamma`, `q`, and `rv1` for a binary system.

    Parameters
    ----------
    gamma : Union[float, npt.NDArray]
        RVs of center of mass
    q : Union[float, npt.NDArray]
        mass ratio = m2/m1
    rv1 : Union[float, npt.NDArray]
        RVs of primary star

    Returns
    -------
    rv2
        RVs of secondary star
    """
    return gamma * (1 + 1 / q) - rv1 / q


def xcorr_spec_twin_gamma(
    rv1: float,
    gamma: float,
    eta: float,
    q: float,
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
) -> float:
    """Evaluate cross-correlation of two spectra (twin case)"""
    rv2 = derive_rv2(gamma, q, rv1)
    # combine two templates
    flux_mod_interp = np.interp(
        wave_obs, wave_mod * (1 + rv1 / SOL_kms), flux_mod
    ) + eta * np.interp(wave_obs, wave_mod * (1 + rv2 / SOL_kms), flux_mod)
    return xcorr(flux_obs, flux_mod_interp)


def nxcorr_spec_twin_gamma(
    rv1: float,
    gamma: float,
    eta: float,
    q: float,
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
) -> float:
    """Negative cross-correlation of two spectra (twin case)."""
    rv2 = derive_rv2(gamma, q, rv1)
    # combine two templates
    flux_mod_interp = np.interp(
        wave_obs, wave_mod * (1 + rv1 / SOL_kms), flux_mod
    ) + eta * np.interp(wave_obs, wave_mod * (1 + rv2 / SOL_kms), flux_mod)
    return -xcorr(flux_obs, flux_mod_interp)


def nxcorr_spec_twin_gamma_BR(
    rv1: float,
    gamma: float,
    eta: float,
    q: float,
    wave_B: npt.NDArray,
    flux_B: npt.NDArray,
    wave_R: npt.NDArray,
    flux_R: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
) -> float:
    """Sum of negative cross-correlation of two spectra (twin case) for B & R."""
    rv2 = derive_rv2(gamma, q, rv1)
    # combine two templates
    flux_mod_B = np.interp(
        wave_B, wave_mod * (1 + rv1 / SOL_kms), flux_mod
    ) + eta * np.interp(wave_B, wave_mod * (1 + rv2 / SOL_kms), flux_mod)
    flux_mod_R = np.interp(
        wave_R, wave_mod * (1 + rv1 / SOL_kms), flux_mod
    ) + eta * np.interp(wave_R, wave_mod * (1 + rv2 / SOL_kms), flux_mod)
    return -np.nansum((xcorr(flux_B, flux_mod_B), xcorr(flux_R, flux_mod_R)))


def xcorr_spec_twin_gamma_rv1grid(
    rv1_grid: npt.NDArray,
    gamma: float,
    eta: float,
    q: float,
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
) -> float:
    """cross-correlation of two spectra (twin case)"""
    rv2 = derive_rv2(gamma, q, rv1_grid)
    # combine two templates
    xcorr_results = np.zeros_like(rv1_grid, dtype=float)
    for irv in range(len(rv1_grid)):
        flux_mod_interp = np.interp(
            wave_obs, wave_mod * (1 + rv1_grid[irv] / SOL_kms), flux_mod
        ) + eta * np.interp(wave_obs, wave_mod * (1 + rv2[irv] / SOL_kms), flux_mod)
        xcorr_results[irv] = xcorr(flux_obs, flux_mod_interp)
    return -np.max(xcorr_results)


def xcorr_spec_binary_cost(
    rv1_rv2_eta: tuple[float, float, float],
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod1: npt.NDArray,
    flux_mod1: npt.NDArray,
    wave_mod2: npt.NDArray,
    flux_mod2: npt.NDArray,
    eta_lim: tuple[float, float] = (0.1, 1.2),
    drvmax: float = 500.0,
) -> float:
    """Negative of xcorr_spec_binary, used as cost function for minimization."""
    rv1, rv2, eta = rv1_rv2_eta
    if not eta_lim[0] < eta <= eta_lim[1] or np.abs(rv2 - rv1) > drvmax:
        return np.inf
    return -xcorr_spec_binary(
        rv1, rv2, eta, wave_obs, flux_obs, wave_mod1, flux_mod1, wave_mod2, flux_mod2
    )


def xcorr_spec_binary_rv2grid(
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod1: npt.NDArray,
    flux_mod1: npt.NDArray,
    wave_mod2: npt.NDArray,
    flux_mod2: npt.NDArray,
    flux_err: Optional[npt.NDArray] = None,
    rv1_init: float = 0.0,
    eta_init: float = 0.3,
    eta_lim: tuple[float, float] = (0.1, 1.2),
    drvmax: float = 500.0,
    drvstep: float = 5.0,
    method: str = "Powell",
    nmc: int = 100,
) -> Union[OptimizeResult | tuple]:
    """Evaluate cross-correlation of two spectra (binary case) based on RV2 grid."""
    # make grid for star 2
    rv2_grid = construct_rv_grid((-drvmax, drvmax, drvstep))
    # ccf2_grid
    ccf2_grid = np.zeros_like(rv2_grid, float)
    for irv2, rv2 in enumerate(rv2_grid):
        ccf2_grid[irv2] = xcorr_spec_binary(
            rv1_init,
            rv2,
            eta_init,
            wave_obs,
            flux_obs,
            wave_mod1,
            flux_mod1,
            wave_mod2,
            flux_mod2,
        )

    # find grid best
    ind_ccfmax = np.argmax(ccf2_grid)
    rv2_best = rv2_grid[ind_ccfmax]
    ccfmax = ccf2_grid[ind_ccfmax]

    if method is None:
        return rv2_best, ccfmax
    else:
        # optimization
        x0 = np.array([rv1_init, rv2_best, eta_init])
        opt = minimize(
            xcorr_spec_binary_cost,
            x0,
            method=method,
            args=(
                wave_obs,
                flux_obs,
                wave_mod1,
                flux_mod1,
                wave_mod2,
                flux_mod2,
                eta_lim,
            ),
        )
        opt["x0"] = x0
        opt["ccfmax2"] = -opt["fun"]

    if flux_err is not None:
        # add random noise to flux_obs
        xs = []
        for i in range(nmc):
            flux_mc = flux_obs + np.random.normal(0, 1, flux_obs.shape) * flux_err
            this_opt = minimize(
                xcorr_spec_binary_cost,
                opt.x,
                method=method,
                args=(
                    wave_obs,
                    flux_mc,
                    wave_mod1,
                    flux_mod1,
                    wave_mod2,
                    flux_mod2,
                    eta_lim,
                ),
            )
            xs.append(this_opt.x)
        opt["x_pct"] = np.percentile(np.array(xs), q=[16, 50, 84], axis=0)
    return opt


def respw_cost(
    rv: float,
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
    pw: float = 1,
) -> float:
    """Cost function of residual powered."""
    flux_mod_interp = np.interp(wave_obs, wave_mod * (1 + rv / SOL_kms), flux_mod)
    cost = np.sum(np.abs(flux_obs - flux_mod_interp) ** pw)
    return cost


def respw_rvgrid(
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
    rv_grid: npt.NDArray,
    pw: float = 1,
) -> npt.NDArray:
    """Cost function of residuals based on rv grid."""
    respw_grid = np.array(
        [
            respw_cost(rv, wave_obs, flux_obs, wave_mod, flux_mod, pw=pw)
            for rv in rv_grid
        ]
    )
    return respw_grid


def xcorr_spec_rvgrid(
    wave_obs: npt.NDArray,
    flux_obs: npt.NDArray,
    wave_mod: npt.NDArray,
    flux_mod: npt.NDArray,
    rv_grid: npt.NDArray,
) -> npt.NDArray:
    """Evaluate cross-correlation via Tonry & Davis (1979).

    Interpolate a model spectrum with different RV and cross-correlate with
    the observed spectrum, return the CCF at the RV grid.

    wave_obs: array
        Wavelength of observed spectrum (normalized).
    flux_obs: array
        Flux of observed spectrum.
    wave_mod: array
        Wavelength of model spectrum (normalized).
    flux_mod:
        Flux of model spectrum.
    rv_grid:
        RV grid, a tuple of (start, stop, step).

    """
    wave_obs = np.asarray(wave_obs)
    flux_obs = np.asarray(flux_obs)
    wave_mod = np.asarray(wave_mod)
    flux_mod = np.asarray(flux_mod)

    # RV grid --> CCF grid
    rv_grid = construct_rv_grid(rv_grid)
    # nz = len(z_grid)
    ccf_grid = np.ones_like(rv_grid, float)

    # calculate CCF
    for i_rv, this_rv in enumerate(rv_grid):
        ccf_grid[i_rv] = xcorr_spec(this_rv, wave_obs, flux_obs, wave_mod, flux_mod)

    return ccf_grid


class RVM:
    def __init__(self, pmod: npt.NDArray, wave_mod: npt.NDArray, flux_mod: npt.NDArray):
        """
        Parameters:
        -----------
        pmod: (n_model, *)
            parameters of model spectra
        wave_mod: (n_pixel,)
            wavelength of model spectra
        flux_mod: (n_model, n_pixel)
            normalized flux of model spectra
        """
        print("@RVM2: initializing Radial Velocity Machine (RVM)...")
        # wavelength
        self.wave_mod = wave_mod

        # parameters
        if pmod.ndim == 2:
            self.pmod = pmod
        else:
            self.pmod = pmod.reshape(1, -1)

        # flux
        if flux_mod.ndim == 2:
            self.flux_mod = flux_mod
        else:
            self.flux_mod = flux_mod.reshape(1, -1)

        # dimensions
        self.nparam = self.pmod.shape[1]
        self.nmod, self.npix = self.flux_mod.shape

        # cache names
        self.cache_names = []

    def __repr__(self):
        return "<RVM [nmod={}] [{:.1f}<lambda<{:.1f}]>".format(
            self.nmod, self.wave_mod[0], self.wave_mod[-1]
        )

    def make_cache(
        self,
        cache_name: str = "B",
        wave_range: Union[list[tuple], tuple] = (5000, 5300),
        rv_grid: Union[tuple, npt.NDArray] = (-1000, 1000, 10),
    ) -> None:
        """Make cache for fast RV measurements, for single exposures only.

        Parameters
        ----------
        cache_name:
            Suffix of cached data.
        wave_range:
            Wavelength range(s).
        rv_grid:
            A tuple of (rv_start, rv_stop, rv_step) or a list of tuples.
        """
        print("@RVM: making cache ...")
        if isinstance(wave_range, list):
            # multiple ranges
            ind_wave_cache = np.zeros_like(self.wave_mod, dtype=bool)
            for _wave_range in wave_range:
                this_ind = (self.wave_mod > _wave_range[0]) & (
                    self.wave_mod < _wave_range[1]
                )
                print(
                    "@RVM: adding {} pixels in range [{}, {}]".format(
                        np.sum(this_ind), *_wave_range
                    )
                )
                ind_wave_cache |= this_ind
        else:
            # single range
            ind_wave_cache = (self.wave_mod > wave_range[0]) & (
                self.wave_mod < wave_range[1]
            )
            print(
                "@RVM: adding {} pixels in range [{}, {}]".format(
                    np.sum(ind_wave_cache), *wave_range
                )
            )
        # cache wavelength
        wave_cache = self.wave_mod[ind_wave_cache]
        # cache rv_grid
        rv_grid_cache = construct_rv_grid(rv_grid)
        # cache model flux
        npix = len(wave_cache)
        nmod = self.nmod
        nrv = len(rv_grid_cache)
        flux_mod_cache = np.zeros((nmod, nrv, npix), dtype=float)
        for imod in range(nmod):
            for irv in range(nrv):
                flux_mod_cache[imod, irv, :] = np.interp(
                    wave_cache,
                    self.wave_mod * (1 + rv_grid_cache[irv] / SOL_kms),
                    self.flux_mod[imod],
                )
        self.__setattr__("wave_mod_cache_{}".format(cache_name), wave_cache)
        self.__setattr__("rv_grid_cache_{}".format(cache_name), rv_grid_cache)
        self.__setattr__("flux_mod_cache_{}".format(cache_name), flux_mod_cache)
        # statistics
        _mean1 = np.mean(flux_mod_cache, axis=2)
        _res1 = flux_mod_cache - _mean1[:, :, None]
        _cov11 = np.sum(_res1**2.0, axis=2)  # var(G, G)  (nmod, nrv)
        self.__setattr__("_mean1_cache_{}".format(cache_name), _mean1)
        self.__setattr__("_res1_cache_{}".format(cache_name), _res1)
        self.__setattr__("_cov11_cache_{}".format(cache_name), _cov11)
        # update cache names
        self.cache_names.append(cache_name)

    def delete_cache(self, cache_name: str) -> None:
        """delete cache"""
        assert cache_name in self.cache_names
        print("@RVM: deleting cache [cache_name={}]...".format(cache_name))
        self.__delattr__("wave_mod_cache_{}".format(cache_name))
        self.__delattr__("rv_grid_cache_{}".format(cache_name))
        self.__delattr__("flux_mod_cache_{}".format(cache_name))
        self.__delattr__("_mean1_cache_{}".format(cache_name))
        self.__delattr__("_res1_cache_{}".format(cache_name))
        self.__delattr__("_cov11_cache_{}".format(cache_name))
        self.cache_names.remove(cache_name)

    def shrink(self, nmod: Union[float, int] = 0.5, method: str = "top"):
        # determine number of models
        if 0 < nmod < 1:
            assert self.nmod * nmod >= 1
            nmod = int(self.nmod * nmod)
        elif nmod > 1:
            nmod = int(nmod)
        else:
            raise ValueError("Invalid nmod value: {}".format(nmod))
        # determine ind of new models
        assert method in ["top", "bottom", "random"]
        if method == "top":
            ind = np.arange(self.nmod)[:nmod]
        elif method == "bottom":
            ind = np.arange(self.nmod)[-nmod:]
        elif method == "random":
            ind = np.random.choice(np.arange(self.nmod), size=nmod, replace=False)
        else:
            raise ValueError(f"Invalid method {method}")
        # construct new RVM
        return RVM(self.pmod[ind, :], self.wave_mod, self.flux_mod[ind, :])

    def measure(
        self,
        wave_obs: npt.NDArray,
        flux_obs: npt.NDArray,
        rv_grid: Union[tuple, list, npt.NDArray] = (-1000, 1000, 10),
        flux_err: Optional[npt.NDArray] = None,
        flux_bounds: tuple[float, float] = (0, 3.0),
        nmc: int = 100,
        method: str = "BFGS",
        cache_name: Optional[str] = None,
        return_ccfgrid: bool = False,
    ) -> dict:
        """Measure RV for a single spectrum.

        Parameters
        ----------
        wave_obs:
            observed wavelength
        flux_obs:
            observed flux (normalized)
        flux_err:
            observed flux error
        rv_grid: Union[tuple, list, npt.NDArray]
            if cache, use the cached rv_grid
            else, use this rv_grid
        flux_bounds:
            flux bounds
        nmc:
            number of MC repeats
        method:
            optimization method
        cache_name:
            cache name.
            if None: no acceleration;
            if "vector": partial acceleration.
        return_ccfgrid:
            if True, return ccfgrid

        Returns
        -------
        dict
            RV measurement.
        """
        # clip extreme values
        ind3 = (flux_obs > flux_bounds[0]) & (flux_obs < flux_bounds[1])
        flux_obs = np.interp(wave_obs, wave_obs[ind3], flux_obs[ind3])
        # CCF grid
        if cache_name in self.cache_names:
            # if cache exists
            flux0 = np.interp(
                self.__getattribute__("wave_mod_cache_{}".format(cache_name)),
                wave_obs[ind3],
                flux_obs[ind3],
                left=1,
                right=1,
            )
            mean0 = np.mean(flux0)
            res0 = flux0 - mean0
            # underscore means it is not normalized
            _cov00 = np.sum(res0**2.0)  # var(F, F) float
            _cov01 = np.sum(
                res0.reshape(1, 1, -1)
                * self.__getattribute__("_res1_cache_{}".format(cache_name)),
                axis=2,
            )  # cov(F, G) (nmod, nrv)
            ccf_grid = (
                _cov01
                / np.sqrt(_cov00)
                / np.sqrt(self.__getattribute__("_cov11_cache_{}".format(cache_name)))
            )
        elif cache_name == "matrix":
            # vectorize data to accelerate
            # e.g., 100 templates x 200 rv values (-1000 to 1000, a step of 10) x 3347 pixels takes 450MB memory
            rv_grid = construct_rv_grid(rv_grid)
            ccf_grid = xcorr_spec_vectorized(
                wave_obs, flux_obs, self.wave_mod, self.flux_mod, rv_grid
            )
        elif cache_name is None or cache_name is False:
            # no cache
            rv_grid = construct_rv_grid(rv_grid)
            ccf_grid = np.zeros((self.flux_mod.shape[0], rv_grid.shape[0]))
            for imod in range(self.nmod):
                ccf_grid[imod] = xcorr_spec_rvgrid(
                    wave_obs=wave_obs,
                    flux_obs=flux_obs,
                    wave_mod=self.wave_mod,
                    flux_mod=self.flux_mod[imod],
                    rv_grid=rv_grid,
                )
        else:
            raise ValueError(
                "@RVM: invalid cache_name: {}. valid options:{} or matrix".format(
                    cache_name, self.cache_names
                )
            )

        # CCF max
        ccf_max = np.max(ccf_grid)
        imod, irv_best = np.unravel_index(np.argmax(ccf_grid), ccf_grid.shape)
        if cache_name in self.cache_names:
            rv_best = self.__getattribute__("rv_grid_cache_{}".format(cache_name))[
                irv_best
            ]
        else:
            rv_best = rv_grid[irv_best]
        # CCF opt
        opt = minimize(
            xcorr_spec_cost,
            x0=rv_best,
            args=(wave_obs, flux_obs, self.wave_mod, self.flux_mod[imod]),
            method=method,
        )  # Powell
        result = dict(
            rv_opt=float(opt.x),
            rv_err=float(opt.hess_inv) if method == "BFGS" else np.nan,
            rv_best=rv_best,
            ccfmax=-opt["fun"],
            success=opt.success,
            imod=imod,
            pmod=self.pmod[imod],
            status=opt["status"],
        )
        if flux_err is not None:
            x_mc = np.zeros(nmc, dtype=float)
            for i in range(nmc):
                # CCF opt
                flux_mc = flux_obs + np.random.normal(0, 1, flux_obs.shape) * flux_err
                opt = minimize(
                    xcorr_spec_cost,
                    x0=rv_best,
                    args=(wave_obs, flux_mc, self.wave_mod, self.flux_mod[imod]),
                    method=method,
                )
                x_mc[i] = opt.x
            result["rv_pct"] = np.percentile(x_mc, [16, 50, 84])

        if return_ccfgrid:
            result["ccf_grid"] = ccf_grid

        return result

    def measure_multiepoch(
        self,
        wave_obs_list: list,
        flux_obs_list: list,
        flux_err_obs_list,
        cache_name: str = "BR",
        flux_bounds: tuple = (0.0, 3.0),
        method: str = "BFGS",
        eta_init: float = 0.4,
        eta_lim: tuple = (0.1, 2.0),
        drvmax: float = 500.0,
        drvstep: float = 10,
        nmc: int = 100,
        verbose: bool = False,
    ):
        """Determine the RVs of multi-epoch spectra.

        For best performance, all input spectra are assumed to have the same wavelength coverage.
        Those with only one good arm.

        Parameters
        ----------
        wave_obs_list:
            list of wavelength
        flux_obs_list
            list of flux
        flux_err_obs_list:
            list flux_err
        cache_name:
            Cache name.
        flux_bounds:
            Flux bounds.

        Returns
        -------
        list[dict]
            RV measurements.

        """
        if verbose:
            print(datetime.datetime.now(), "starting")

        # cache algorithm needed
        assert cache_name in self.cache_names

        # assert cache is ready
        assert len(wave_obs_list) == len(flux_obs_list)
        n_spec = len(wave_obs_list)

        # initialize ccf arrays
        ccf_max_grid = np.zeros(n_spec, int)
        imod_best_grid = np.zeros(n_spec, int)
        irv_best_grid = np.zeros(n_spec, int)
        if verbose:
            print(datetime.datetime.now(), "cache grid for best template")
        # loop spectra
        for i_spec in range(n_spec):
            # get current spectrum
            wave_obs = wave_obs_list[i_spec]
            flux_obs = flux_obs_list[i_spec]
            # clip extreme values
            ind_inbounds = (flux_obs > flux_bounds[0]) & (flux_obs < flux_bounds[1])
            flux_obs = np.interp(
                wave_obs, wave_obs[ind_inbounds], flux_obs[ind_inbounds]
            )
            # interpolate to cache wavelength grid
            flux0 = np.interp(
                self.__getattribute__("wave_mod_cache_{}".format(cache_name)),
                wave_obs[ind_inbounds],
                flux_obs[ind_inbounds],
                left=1,
                right=1,
            )
            # calculate ccf_grid
            mean0 = np.mean(flux0)
            res0 = flux0 - mean0
            # underscore means it is not normalized
            _cov00 = np.sum(res0**2.0)  # var(F, F) float
            _cov01 = np.sum(
                res0.reshape(1, 1, -1)
                * self.__getattribute__("_res1_cache_{}".format(cache_name)),
                axis=2,
            )  # cov(F, G) (nmod, nrv)
            ccf_grid = (
                _cov01
                / np.sqrt(_cov00)
                / np.sqrt(self.__getattribute__("_cov11_cache_{}".format(cache_name)))
            )

            # CCF max
            ccf_max_grid[i_spec] = np.max(ccf_grid)
            imod_best_grid[i_spec], irv_best_grid[i_spec] = np.unravel_index(
                np.argmax(ccf_grid), ccf_grid.shape
            )

        # select the best template
        imod_selected = imod_best_grid[np.argmax(ccf_max_grid)]

        # initialize ccf arrays
        ccf_max_grid = np.zeros(n_spec, float)
        irv_best_grid = np.zeros(n_spec, int)
        if verbose:
            print(datetime.datetime.now(), "template selected, calculating rv ...")
        res_list = []
        # loop spectra
        for i_spec in range(n_spec):
            # get current spectrum
            wave_obs = wave_obs_list[i_spec]
            flux_obs = flux_obs_list[i_spec]

            # clip extreme values
            ind_inbounds = (flux_obs > flux_bounds[0]) & (flux_obs < flux_bounds[1])
            flux_obs = np.interp(
                wave_obs, wave_obs[ind_inbounds], flux_obs[ind_inbounds]
            )
            # interpolate to cache wavelength grid
            flux0 = np.interp(
                self.__getattribute__("wave_mod_cache_{}".format(cache_name)),
                wave_obs[ind_inbounds],
                flux_obs[ind_inbounds],
                left=1,
                right=1,
            )
            # calculate ccf_grid
            mean0 = np.mean(flux0)
            res0 = flux0 - mean0
            # underscore means it is not normalized
            _cov00 = np.sum(res0**2.0)  # var(F, F) float
            _cov01 = np.sum(
                res0.reshape(1, -1)
                * self.__getattribute__("_res1_cache_{}".format(cache_name))[
                    imod_selected
                ],
                axis=1,
            )  # cov(F, G) (nrv,)
            ccf_grid = (
                _cov01
                / np.sqrt(_cov00)
                / np.sqrt(
                    self.__getattribute__("_cov11_cache_{}".format(cache_name))[
                        imod_selected
                    ]
                )
            )

            # CCF max
            ccf_max_grid[i_spec] = np.max(ccf_grid)
            irv_best_grid[i_spec] = np.argmax(ccf_grid)
            rv_best_grid = self.__getattribute__("rv_grid_cache_{}".format(cache_name))[
                irv_best_grid[i_spec]
            ]

            # CCF opt
            opt = minimize(
                xcorr_spec_cost,
                x0=self.__getattribute__("rv_grid_cache_{}".format(cache_name))[
                    irv_best_grid[i_spec]
                ],
                args=(wave_obs, flux_obs, self.wave_mod, self.flux_mod[imod_selected]),
                method=method,
            )  # Powell
            # store single star result
            this_res = dict(n_spec=n_spec)
            this_res["rv1_opt_{}".format(cache_name)] = float(opt.x)
            this_res["rv1_best_{}".format(cache_name)] = rv_best_grid
            this_res["ccfmax1_{}".format(cache_name)] = -opt["fun"]
            this_res["imod_{}".format(cache_name)] = imod_selected
            this_res["pmod_{}".format(cache_name)] = self.pmod[imod_selected]
            this_res["status1_{}".format(cache_name)] = opt["status"]
            this_res["success1_{}".format(cache_name)] = opt.success

            # Monte Carlo for error
            if flux_err_obs_list is not None:
                flux_err_obs = flux_err_obs_list[i_spec]
                x_mc = np.zeros(nmc, dtype=float)
                for i in range(nmc):
                    # CCF opt
                    flux_mc = (
                        flux_obs + np.random.normal(0, 1, flux_obs.shape) * flux_err_obs
                    )
                    opt = minimize(
                        xcorr_spec_cost,
                        x0=self.__getattribute__("rv_grid_cache_{}".format(cache_name))[
                            irv_best_grid[i_spec]
                        ],
                        args=(
                            wave_obs,
                            flux_mc,
                            self.wave_mod,
                            self.flux_mod[imod_selected],
                        ),
                        method=method,
                    )
                    x_mc[i] = opt.x
                this_res["rv1_pct_{}".format(cache_name)] = np.percentile(
                    x_mc, [16, 50, 84]
                )
                this_res["rv1_err_{}".format(cache_name)] = (
                    float(opt.hess_inv)
                    if method == "BFGS"
                    else np.mean(np.diff(this_res["rv1_pct_{}".format(cache_name)]))
                )
            else:
                flux_err_obs = None
                this_res["rv1_err_{}".format(cache_name)] = (
                    float(opt.hess_inv) if method == "BFGS" else np.nan
                )

            # measure double components
            """ given a template, optimize (rv1, drv, eta) """
            rvr2 = xcorr_spec_binary_rv2grid(
                wave_obs,
                flux_obs,
                self.wave_mod,
                self.flux_mod[imod_selected],
                self.wave_mod,
                self.flux_mod[imod_selected],
                flux_err=flux_err_obs,
                rv1_init=this_res["rv1_opt_{}".format(cache_name)],
                eta_init=eta_init,
                eta_lim=eta_lim,
                drvmax=drvmax,
                drvstep=drvstep,
                method=method,
                nmc=nmc,
            )

            (
                this_res["rv1_{}".format(cache_name)],
                this_res["rv2_{}".format(cache_name)],
                this_res["eta_{}".format(cache_name)],
            ) = rvr2["x"]
            this_res["dccfmax_{}".format(cache_name)] = (
                rvr2["ccfmax2"] - this_res["ccfmax1_{}".format(cache_name)]
            )
            this_res["ccfmax2_{}".format(cache_name)] = rvr2["ccfmax2"]

            this_res["rv1_rv2_eta0_{}".format(cache_name)] = rvr2["x0"]
            this_res["rv1_rv2_eta_{}".format(cache_name)] = rvr2["x"]
            this_res["rv1_rv2_eta_pct_{}".format(cache_name)] = rvr2["x_pct"]
            this_res["rv1_rv2_eta_err_{}".format(cache_name)] = (
                rvr2["x_pct"][2] - rvr2["x_pct"][0]
            ) / 2
            this_res["success_2_{}".format(cache_name)] = rvr2["success"]
            this_res["status2_{}".format(cache_name)] = rvr2["status"]
            if verbose:
                print(datetime.datetime.now(), "finished")
            res_list.append(this_res)

        return res_list

    def measure2(
        self,
        wave_obs: npt.NDArray,
        flux_obs: npt.NDArray,
        flux_err: npt.NDArray,
        wave_mod1: npt.NDArray,
        flux_mod1: npt.NDArray,
        wave_mod2: npt.NDArray,
        flux_mod2: npt.NDArray,
        rv1_init: float = 0.0,
        eta_init: float = 0.3,
        eta_lim=(0.1, 1.0),
        drvmax: float = 500.0,
        drvstep: float = 5.0,
        method: str = "Powell",
        nmc: int = 100,
    ):
        """given a template, optimize (rv1, drv, eta)"""
        opt = xcorr_spec_binary_rv2grid(
            wave_obs,
            flux_obs,
            wave_mod1,
            flux_mod1,
            wave_mod2,
            flux_mod2,
            flux_err,
            rv1_init=rv1_init,
            eta_init=eta_init,
            eta_lim=eta_lim,
            drvmax=drvmax,
            drvstep=drvstep,
            method=method,
            nmc=nmc,
        )
        return opt

    def mock_binary_spectrum(self, imod1, imod2, rv1, drv, eta):
        """make mock binary spectrum"""
        flux_mod_interp = np.interp(
            self.wave_mod, self.wave_mod * (1 + rv1 / SOL_kms), self.flux_mod[imod1]
        ) + eta * np.interp(
            self.wave_mod,
            self.wave_mod * (1 + (rv1 + drv) / SOL_kms),
            self.flux_mod[imod2],
        )
        return flux_mod_interp

    def reproduce_spectrum_single(self, rvr: dict):
        """reproduce the spectrum"""
        imod1 = rvr["imod1"]
        rv1 = rvr["rv1"]
        flux_mod_interp = np.interp(
            self.wave_mod, self.wave_mod * (1 + rv1 / SOL_kms), self.flux_mod[imod1]
        )
        return flux_mod_interp

    def reproduce_spectrum_binary(self, rvr):
        """reproduce the binary spectrum"""
        imod1 = rvr["imod1"]
        imod2 = rvr["imod2"]
        rv1, drv, eta = rvr["rv1_drv_eta"]

        flux_mod_interp = np.interp(
            self.wave_mod, self.wave_mod * (1 + rv1 / SOL_kms), self.flux_mod[imod1]
        ) + eta * np.interp(
            self.wave_mod,
            self.wave_mod * (1 + (rv1 + drv) / SOL_kms),
            self.flux_mod[imod2],
        )
        return flux_mod_interp / (1 + eta)

    def measure_binary(
        self,
        wave_obs: npt.NDArray,
        flux_obs: npt.NDArray,
        rv_grid: Union[tuple, list, npt.NDArray] = (-1000, 1000, 10),
        flux_err: Optional[npt.NDArray] = None,
        cache_name: str = "B",
        flux_bounds: tuple = (0.0, 3.0),
        twin: bool = True,
        eta_init: float = 0.3,
        eta_lim: tuple = (0.01, 3.0),
        drvmax: float = 500.0,
        drvstep: float = 5.0,
        method: str = "Powell",
        nmc: int = 100,
        suffix: str = "",
        return_ccfgrid: bool = False,
        return_spec: bool = False,
    ):
        # clip extreme values
        ind3 = (flux_obs > flux_bounds[0]) & (flux_obs < flux_bounds[1])
        flux_obs = np.interp(wave_obs, wave_obs[ind3], flux_obs[ind3])
        # RV1
        rv_grid = construct_rv_grid(rv_grid)
        rvr1 = self.measure(
            wave_obs,
            flux_obs,
            rv_grid=rv_grid,
            flux_err=flux_err,
            nmc=nmc,
            cache_name=cache_name,
            return_ccfgrid=return_ccfgrid,
        )

        # determine the secondary template if necessary
        if twin:
            imod2 = rvr1["imod"]
        else:
            # fix one template and calculate RV2
            drv_best = np.zeros((self.nmod,), float)
            ccfmax = np.zeros((self.nmod,), float)
            for i in range(self.nmod):
                drv_best[i], ccfmax[i] = self.measure2(
                    wave_obs,
                    flux_obs,
                    flux_err=flux_err,
                    wave_mod1=self.wave_mod,
                    flux_mod1=self.flux_mod[rvr1["imod"]],
                    wave_mod2=self.wave_mod,
                    flux_mod2=self.flux_mod[i],
                    rv1_init=rvr1["rv_opt"],
                    eta_init=eta_init,
                    eta_lim=eta_lim,
                    drvmax=drvmax,
                    drvstep=drvstep,
                    method=None,
                )
            # best secondary
            imod2 = np.argmax(ccfmax)

        rvr2 = self.measure2(
            wave_obs,
            flux_obs,
            flux_err,
            wave_mod1=self.wave_mod,
            flux_mod1=self.flux_mod[rvr1["imod"]],
            wave_mod2=self.wave_mod,
            flux_mod2=self.flux_mod[imod2],
            rv1_init=rvr1["rv_opt"],
            eta_init=eta_init,
            eta_lim=eta_lim,
            drvmax=drvmax,
            drvstep=drvstep,
            method=method,
            nmc=nmc,
        )

        if suffix == "" or suffix is None:
            suffix = ""
        else:
            suffix = "_{}".format(suffix)

        if flux_err is None:
            rvr = OrderedDict()
            rvr["rv1{}".format(suffix)] = rvr1["rv_opt"]
            rvr["ccfmax1{}".format(suffix)] = rvr1["ccfmax"]
            rvr["rv1_best{}".format(suffix)] = rvr1["rv_best"]
            rvr["imod1{}".format(suffix)] = rvr1["imod"]
            rvr["pmod1{}".format(suffix)] = rvr1["pmod"]
            rvr["imod2{}".format(suffix)] = imod2
            rvr["pmod2{}".format(suffix)] = self.pmod[imod2]
            rvr["success1{}".format(suffix)] = rvr1["success"]
            rvr["ccfmax2{}".format(suffix)] = rvr2["ccfmax2"]
            rvr["success2{}".format(suffix)] = rvr2["success"]
            rvr["rv1_rv2_eta0{}".format(suffix)] = rvr2["x0"]
            rvr["rv1_rv2_eta{}".format(suffix)] = rvr2["x"]
            rvr["status1{}".format(suffix)] = rvr1["status"]
            rvr["status2{}".format(suffix)] = rvr2["status"]
        else:
            rvr = OrderedDict()
            rvr["rv1{}".format(suffix)] = rvr1["rv_opt"]
            rvr["rv1_pct{}".format(suffix)] = rvr1["rv_pct"]
            rvr["rv1_err{}".format(suffix)] = (
                rvr1["rv_pct"][2] - rvr1["rv_pct"][0]
            ) / 2
            rvr["ccfmax1{}".format(suffix)] = rvr1["ccfmax"]
            rvr["rv1_best{}".format(suffix)] = rvr1["rv_best"]
            rvr["imod1{}".format(suffix)] = rvr1["imod"]
            rvr["pmod1{}".format(suffix)] = rvr1["pmod"]
            rvr["imod2{}".format(suffix)] = imod2
            rvr["pmod2{}".format(suffix)] = self.pmod[imod2]
            rvr["success1{}".format(suffix)] = rvr1["success"]
            rvr["ccfmax2{}".format(suffix)] = rvr2["ccfmax2"]
            rvr["success2{}".format(suffix)] = rvr2["success"]
            rvr["rv1_rv2_eta0{}".format(suffix)] = rvr2["x0"]
            rvr["rv1_rv2_eta{}".format(suffix)] = rvr2["x"]
            rvr["rv1_rv2_eta_pct{}".format(suffix)] = rvr2["x_pct"]
            rvr["rv1_rv2_eta_err{}".format(suffix)] = (
                rvr2["x_pct"][2] - rvr2["x_pct"][0]
            ) / 2
            rvr["status1{}".format(suffix)] = rvr1["status"]
            rvr["status2{}".format(suffix)] = rvr2["status"]
            # rvr["b_rv1{}".format(suffix)] = rvr2["x_pct"][0]
            # rvr["b_rv2{}".format(suffix)] = rvr2["x_pct"][1]
            # rvr["b_eta{}".format(suffix)] = rvr2["x_pct"][2]
            # rvr["b_rv1_err{}".format(suffix)] = rvr2["x_pct"][0]
            # rvr["b_rv2_err{}".format(suffix)] = rvr2["x_pct"][1]
            # rvr["b_eta_err{}".format(suffix)] = rvr2["x_pct"][2]
        if return_ccfgrid:
            rvr["ccf_grid{}".format(suffix)] = rvr1["ccf_grid"]
        if return_spec:
            rvr["spec{}".format(suffix)] = wave_obs, flux_obs

        # if method is "BFGS":
        #     rvr["hess_inv"] = rvr2["hess_inv"]
        return rvr

    def ccf_1mod(
        self,
        wave_mod: npt.NDArray,
        flux_mod: npt.NDArray,
        wave_obs: npt.NDArray,
        flux_obs: npt.NDArray,
        rv_grid: Union[tuple, list, npt.NDArray] = (-1000, 1000, 10),
        flux_bounds: tuple = (0, 3.0),
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Calculate CCF for a given flux_mod and a flux_obs at `rv_grid`."""
        # clip extreme values
        ind3 = (flux_obs > flux_bounds[0]) & (flux_obs < flux_bounds[1])
        flux_obs = np.interp(wave_obs, wave_obs[ind3], flux_obs[ind3])
        # CCF grid
        rv_grid = construct_rv_grid(rv_grid)
        ccf_grid = xcorr_spec_rvgrid(wave_obs, flux_obs, wave_mod, flux_mod[1], rv_grid)
        return rv_grid, ccf_grid

    def chi2_1mod(
        self,
        imod: int,
        wave_obs: npt.NDArray,
        flux_obs: npt.NDArray,
        rv_grid: Union[tuple, list, npt.NDArray] = (-1000, 1000, 10),
        pw: float = 2,
        flux_bounds: tuple[float, float] = (0.0, 3.0),
    ):
        """measure RV"""
        # clip extreme values
        ind3 = (flux_obs > flux_bounds[0]) & (flux_obs < flux_bounds[1])
        flux_obs = np.interp(wave_obs, wave_obs[ind3], flux_obs[ind3])
        # respw grid
        rv_grid = construct_rv_grid(rv_grid)
        respw_grid = respw_rvgrid(
            wave_obs,
            flux_obs,
            self.wave_mod,
            self.flux_mod[imod],
            rv_grid,
            pw=pw,
        )
        return rv_grid, respw_grid

    def measure_pw(
        self,
        wave_obs: npt.NDArray,
        flux_obs: npt.NDArray,
        rv_grid: Union[tuple, list, npt.NDArray] = (-1000, 1000, 10),
        method: str = "BFGS",
        pw: float = 1.0,
    ):
        # clip extreme values
        ind3 = (flux_obs < 3) & (flux_obs > 0.0)
        flux_obs = np.interp(wave_obs, wave_obs[ind3], flux_obs[ind3])
        # CCF grid
        rv_grid = construct_rv_grid(rv_grid)
        ccf = np.zeros((self.flux_mod.shape[0], rv_grid.shape[0]))
        for j in range(self.flux_mod.shape[0]):
            ccf[j] = xcorr_spec_rvgrid(
                wave_obs, flux_obs, self.wave_mod, self.flux_mod[j][1], rv_grid
            )
        # CCF max
        ccfmax = np.max(ccf)
        ind_best = np.where(ccfmax == ccf)
        imod = ind_best[0][0]
        irv_best = ind_best[1][0]
        rv_best = rv_grid[irv_best]
        # CCF opt
        opt = minimize(
            respw_cost,
            x0=rv_best,
            args=(wave_obs, flux_obs, self.wave_mod, self.flux_mod[imod], pw),
            method=method,
        )
        # opt = minimize(ccf_cost_interp, x0=rv_best, args=(wave_obs, flux_obs, wave_mod, flux_mod[imod_best]),
        # method="Powell")
        # x = np.interp(wave, wave_obs/(1+opt.x/SOL_kms), flux_obs).reshape(1, -1)
        return dict(
            rv_opt=float(opt.x),
            rv_err=float(opt.hess_inv) if method == "BFGS" else np.nan,
            rv_best=rv_best,
            ccfmax=ccfmax,
            success=opt.success,
            imod=imod,
            pmod=self.pmod[imod],
            opt=opt,
        )

    def measure_binary_mrsbatch(
        self,
        fp: str,
        lmjm: int,
        snr_B: Optional[float] = None,
        snr_R: Optional[float] = None,
        snr_threshold: float = 5.0,
        raise_error: bool = False,
    ) -> OrderedDict:
        """this is the configuration used in"""
        # read spectrum
        mrv_kwargs = {
            "rv_grid": (-1000, 1000, 10),
            "eta_init": 0.5,
            "eta_lim": (0.01, 3.0),
        }
        mf = MrsFits(fp.strip())
        try:
            # blue arm
            ms = mf.get_one_spec(lmjm=lmjm, band="B")
            if snr_B is not None:
                assert snr_B > snr_threshold
            else:
                assert ms.snr > snr_threshold
            # cosmic ray removal
            msr = ms.reduce(npix_cushion=70, norm_type="spline", niter=2)
            rvr_B = self.measure_binary(
                msr.wave,
                msr.flux_norm,
                flux_err=msr.flux_norm_err,
                cache_name="B",
                nmc=50,
                **mrv_kwargs,
                suffix="B",
            )
            rvr_B["bjdmid"] = ms.bjdmid
        except Exception as e_:
            if raise_error:
                raise e_
            rvr_B = {}

        try:
            # red arm
            ms = mf.get_one_spec(lmjm=lmjm, band="R")
            if snr_B is not None:
                assert snr_R > snr_threshold
            else:
                assert ms.snr > snr_threshold
            # cosmic ray removal
            msr = ms.reduce(npix_cushion=70, norm_type="spline", niter=2)
            # cut 6800+A
            ind_use = msr.wave < 6800
            rvr_R = self.measure_binary(
                msr.wave[ind_use],
                msr.flux_norm[ind_use],
                flux_err=msr.flux_norm_err[ind_use],
                cache_name="R",
                nmc=50,
                **mrv_kwargs,
                suffix="R",
            )
            rvr_R["bjdmid"] = ms.bjdmid
            # ind_use = (msr.wave < 6800) & ((msr.wave < 6540) | (msr.wave > 6590))
            # rvr_Rm = self.measure_binary(msr.wave[ind_use], msr.flux_norm[ind_use],
            #                              flux_err=msr.flux_norm_err[ind_use], nmc=50, **mrv_kwargs, suffix="Rm")
        except Exception as e_:
            if raise_error:
                raise e_
            rvr_R = {}
            # rvr_Rm = {}

        rvr_B.update(rvr_R)
        # rvr_B.update(rvr_Rm)
        return rvr_B

    def mrsbatch(
        self,
        fp_list: list[str],
        lmjm_list: list[int],
        snr_B_list: list[float],
        snr_R_list: list[float],
        snr_threshold: float = 5.0,
        fpout: Optional[str] = None,
    ):
        if os.path.exists(fpout):
            return
        nspec = len(fp_list)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rvr = [
                self.measure_binary_mrsbatch(
                    fp_list[i],
                    lmjm_list[i],
                    snr_B_list[i],
                    snr_R_list[i],
                    snr_threshold=snr_threshold,
                )
                for i in range(nspec)
            ]
        if fpout is None:
            return rvr
        else:
            joblib.dump(rvr, fpout)


def construct_rv_grid(
    rv_grid: Union[tuple, list, npt.NDArray] = (-1000, 1000, 10)
) -> npt.NDArray:
    """Construct rv_grid form input.

    Parameters
    ----------
    rv_grid : Union[tuple, list, npt.NDArray]
        RV grid or parameters.
        - tuple/list: (rv_start, rv_stop, rv_step)
        - npt.NdArray: return grid

    Returns
    -------
    npt.NDArray
        RV grid.
    """
    if isinstance(rv_grid, (list, tuple)):
        # RV grid parameters
        assert (
            isinstance(rv_grid, tuple) and len(rv_grid) == 3
        ), "The input `rv_grid` should be a tuple of (rv_start, rv_stop, rv_step)!"
        return np.arange(rv_grid[0], rv_grid[1] + rv_grid[2], rv_grid[2], dtype=float)
    else:
        # RV grid already
        return np.asarray(rv_grid, dtype=float)


def test_rvm_20240401():
    import joblib
    import numpy as np
    from laspec import RVM

    rvmdata = joblib.load("/Users/cham/projects/lamost/rvm/RVMDATA_R7500_M8.joblib")
    rvm = RVM(**rvmdata)

    i_test = 1
    print(rvm.pmod[i_test])
    rvm.make_cache(cache_name="B", wave_range=(5000, 5300), rv_grid=(-1000, 1000, 10))

    index_B = (rvm.wave_mod > 5000) & (rvm.wave_mod < 5300)
    rv = 123.45
    wave_obs = rvm.wave_mod[index_B] * (1 + rv / SOL_kms)
    flux_obs = rvm.flux_mod[i_test][index_B] + np.random.normal(
        loc=0, scale=0.05, size=wave_obs.shape
    )
    # %%time
    rvm.measure(wave_obs, flux_obs, flux_err=None, cache_name=None)
    # %%time
    rvm.measure(wave_obs, flux_obs, flux_err=None, cache_name="matrix")
    # %%time
    rvm.measure(wave_obs, flux_obs, flux_err=None, cache_name="B")
    # %%time
    rvm.measure_binary(
        wave_obs, flux_obs, rv_grid=(-1000, 1000, 10), flux_err=None, cache_name="B"
    )


# def test_new_rvm():
#     import joblib
#
#     rvm = RVM(
#         joblib.load(
#             "/Users/cham/PycharmProjects/laspec/laspec/data/rvm/v8_rvm_pmod.dump"
#         ),
#         joblib.load(
#             "/Users/cham/PycharmProjects/laspec/laspec/data/rvm/v8_rvm_wave_mod.dump"
#         ),
#         joblib.load(
#             "/Users/cham/PycharmProjects/laspec/laspec/data/rvm/v8_rvm_flux_mod.dump"
#         ),
#     )
#
#     waveBR, spec_list = joblib.load(
#         "/Users/cham/projects/sb2/test_ccf/wave_flux_30.dump"
#     )
#     wave_obs = waveBR[waveBR < 5500]
#     npix = len(wave_obs)
#     # %%%% read spectra
#     import glob
#
#     fps = glob.glob("./*.fits.gz")
#     from laspec.mrs import MrsSource, debad
#
#     ms = MrsSource.read(fps)
#
#     # %%
#     fig, axs = plt.subplots(1, 2)
#     for i, me in enumerate(ms[:1]):
#         # wave_obs = me.wave_B
#         # flux_obs = me.flux_norm_B
#         wave_obs, flux_obs = (
#             me.wave_B[50:-50],
#             debad(me.wave_B, me.flux_norm_B, nsigma=(4, 8), maxiter=5)[50:-50],
#         )
#         axs[0].plot(wave_obs, flux_obs + i, "k")
#
#         # measure RV
#         rvr = rvm.measure(wave_obs, flux_obs)
#         print(rvr)
#         imod = rvr["imod"]
#         rv_grid, ccf_grid = rvm.ccf_1mod(
#             rvm.wave_mod,
#             rvm.flux_mod[imod],
#             wave_obs,
#             flux_obs,
#             rv_grid=(-2000, 2000, 5),
#         )
#         axs[1].plot(rv_grid, ccf_grid + i, "b")
#
#     # %%time
#     rvr = []
#
#     for i, me in enumerate(ms[:]):
#         print(i)
#
#         # wave_obs = me.wave_B
#         # flux_obs = me.flux_norm_B
#
#         # remove cosmic rays
#         wave_obs, flux_obs = (
#             me.wave_B[50:-50],
#             debad(me.wave_B, me.flux_norm_B, nsigma=(4, 8), maxiter=5)[50:-50],
#         )
#         # measure binary
#         this_rvr = rvm.measure_binary(
#             wave_obs,
#             flux_obs,
#             rv_grid=(-600, 600, 10),
#             flux_bounds=(0, 3.0),
#             eta_init=0.3,
#             drvmax=500,
#             drvstep=5,
#             method="Powell",
#         )
#         this_rvr["lmjm"] = me.epoch
#         this_rvr["snr"] = me.snr[0]
#         # append results
#         rvr.append(this_rvr)
#
#     from astropy.table import Table
#
#     trvr = Table(rvr)
#     trvr.write("./trvr.fits", overwrite=True)
#     trvr.show_in_browser()
#     # %%
#     plt.figure()
#     plt.plot(trvr["snr"], trvr["ccfmax1"], "bo")
#     plt.plot(trvr["snr"], trvr["ccfmax2"], "ro")
#     plt.ylim(0, 1)
#     # %%
#     plt.figure()
#     plt.plot(trvr["lmjm"], trvr["rv1_drv_eta"][:, 0], "ro", label="star 1")
#     plt.plot(
#         trvr["lmjm"],
#         trvr["rv1_drv_eta"][:, 0] + trvr["rv1_drv_eta"][:, 1],
#         "bo",
#         label="star 2",
#     )
#     plt.legend(loc="right")
#     plt.xlabel("lmjm")
#     plt.ylabel("RV[km/s]")
#
#
# if __name__ == "__main__":
#     pass
