"""
This module implements the conversion of wavelengths between vacuum and air
Reference: Donald Morton (2000, ApJ. Suppl., 130, 403)
VALD3 link: http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
"""

import numpy as np


def vac2air(wave_vac):
    """
    Parameters
    ----------
    wave_vac:
        wavelength (A) in vacuum

    Return
    ------
    wave_air:
        wavelength (A) in air
    """
    wave_vac = np.array(wave_vac)
    s = 1e4 / wave_vac
    n = 1 + 0.0000834254 + \
        0.02406147 / (130 - s**2) + \
        0.00015998 / (38.9 - s**2)
    return wave_vac / n


def air2vac(wave_air):
    """
    Parameters
    ----------
    wave_air:
        wavelength (A) in air

    Return
    ------
    wave_vac:
        wavelength (A) in vacuum
    """
    wave_air = np.array(wave_air)
    s = 1e4 / wave_air
    n = 1 + 0.00008336624212083 + \
        0.02408926869968 / (130.1065924522 - s ** 2) + \
        0.0001599740894897 / (38.92568793293 - s ** 2)
    return wave_air * n


def mdwave(wave):
    """ median delta wavelength """
    return np.median(np.diff(wave))


def wave_log10(wave, osr_ext=1., dwave=None):
    """ generate log10 wavelength array given wave array

    Parameters
    ----------
    wave:
        old wavelength array
    osr_ext:
        extra over-sampling rate
    dwave:
        delta wavelength. if not specified, use median(dwave).

    Example
    -------
    >>> import numpy as np
    >>> wave = np.arange(3000, 5000)
    >>> wave_new = wave_log10(wave, osr_ext=3)

    """
    if dwave is not None:
        npix = np.int(np.ptp(wave) / dwave * osr_ext + 1)
    else:
        npix = np.int(len(wave) * osr_ext + 1)
    return np.logspace(np.log10(np.min(wave)), np.log10(np.max(wave)), npix, base=10.0)