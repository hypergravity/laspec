import numpy as np
from scipy.interpolate import interp1d

__all__ = ["mdwave", "wave_log10", "center2edge", "rebin_flux", "rebin_mask"]


def mdwave(wave):
    """ median delta wavelength """
    return np.median(np.diff(wave))


def wave_log10(wave, dwave=None):
    """ generate log10 wavelength array given wave array
    Parameters
    ----------
    wave:
        old wavelength array
    dwave:
        delta wavelength. if not specified, use median(dwave).

    Examples
    -------
    >>> wave_new = wave_log10(wave)
    """
    if dwave is None:
        dwave = mdwave(wave)
    npix = np.int(np.ptp(wave) / dwave) + 1
    return np.logspace(np.log10(np.min(wave)), np.log10(np.max(wave)), npix,
                       base=10.0)


def center2edge(x):
    x = np.asarray(x)
    dx = np.diff(x)
    return np.hstack((x[0] - .5 * dx[0], x[:-1] + .5 * dx, x[-1] + .5 * dx[-1]))


def rebin_flux(wave, flux, wave_new):
    """ Rebin spectrum to a new wavelength grid

    wave: array
        old wavelength
    flux: array
        old flux
    wave_new:
        new wavelength

    Return
    ------
    re-binned flux
    """
    wave = np.asarray(wave)
    wave_new = np.asarray(wave_new)

    wave_edge = center2edge(wave)
    wave_new_edge = center2edge(wave_new)

    I = interp1d(wave_edge[:-1], np.arange(len(wave)), kind="linear",
                 bounds_error=False)
    wave_new_edge_pos = I(wave_new_edge)  # accurate position projected to old
    wave_new_edge_pos2 = np.array(
        [wave_new_edge_pos[:-1], wave_new_edge_pos[1:]]).T  # slipt to lo & hi

    wave_new_ipix = np.floor(wave_new_edge_pos2).astype(int)  # integer part
    wave_new_frac = wave_new_edge_pos2 - wave_new_ipix  # fraction part
    flux_new = np.zeros_like(wave_new, dtype=float)
    for ipix, _flag in enumerate(np.any(np.isnan(wave_new_edge_pos2), axis=1)):
        if not _flag:
            flux_new[ipix] = \
                flux[wave_new_ipix[ipix, 0]:wave_new_ipix[ipix, 1]].sum() \
                - flux[wave_new_ipix[ipix, 0]] * wave_new_frac[ipix, 0] \
                + flux[wave_new_ipix[ipix, 1]] * wave_new_frac[ipix, 1]
        else:
            flux_new[ipix] = np.nan
    return flux_new


def rebin_mask(wave, mask, wave_new):
    """ Rebin spectrum to a new wavelength grid

    wave: array
        old wavelength
    mask: array
        old mask, True for bad!
    wave_new:
        new wavelength

    Return
    ------
    re-binned mask
    """
    wave = np.asarray(wave)
    wave_new = np.asarray(wave_new)

    wave_edge = center2edge(wave)
    wave_new_edge = center2edge(wave_new)

    I = interp1d(wave_edge[:-1], np.arange(len(wave)), kind="linear",
                 bounds_error=False)
    wave_new_edge_pos = I(wave_new_edge)  # accurate position projected to old
    wave_new_edge_pos2 = np.array(
        [wave_new_edge_pos[:-1], wave_new_edge_pos[1:]]).T  # slipt to lo & hi

    wave_new_ipix = np.floor(wave_new_edge_pos2).astype(int)  # integer part
    # wave_new_frac = wave_new_edge_pos2 - wave_new_ipix        # fraction part
    mask_new = np.zeros_like(wave_new, dtype=bool)
    for ipix, _flag in enumerate(np.any(np.isnan(wave_new_edge_pos2), axis=1)):
        if not _flag:
            mask_new[ipix] = np.any(
                mask[wave_new_ipix[ipix, 0]:wave_new_ipix[ipix, 1] + 1])
        # else: unnecessary
        #     mask_new[ipix] = False
    return mask_new


def _test():
    wave, flux, wave_new = np.arange(10), np.ones(10), np.arange(0, 10, 2) + 0.5
    print(wave, flux)
    print(wave_new, rebin_flux(wave, flux, wave_new))
    # figure()
    # plot(wave, flux, 'x-')
    # plot(wave_new, rebin(wave, flux, wave_new), 's-')
    return


if __name__ == "__main__":
    _test()
