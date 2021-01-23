# -*- coding: utf-8 -*-
# ###########################################################################
# Enlightening example:
#
# for one pixel:
# R_hi = 2000 @5000A    FWHM_hi =  2.5A
# R_lo =  500 @5000A    FWHM_lo = 10.0A
# FWHM_GK = sqrt(10.0**2 - 2.5**2) = 9.68A
#
# for next pixel:
# R_hi = 1000 @5000A    FWHM_hi =  5.0A
# R_lo =  500 @5000A    FWHM_lo = 10.0A
# FWHM_GK = sqrt(10.0**2 - 5.0**2) = 8.66A
#
# Therefore, to keep the number of pixels of Gaussian Kernels the same,
# we have to adjust the FWHM_interp (and thus the R_interp).
#
# For pixels 2000->500, the FWHM_GK = 9.68 (wider GK),
# while for pixels 1000-> 500, FWHM_GK = 8.66 (narrower GK),
# thus the R_interp should be 9.68/8.66=1.12 times higher,
# i.e., the delta_lambda should be  1./1.12 times smaller.
# ###########################################################################

import datetime

import numpy as np
from astropy.table import Table
from scipy import signal
from scipy.interpolate import interp1d


# ########################################################################### #
# ################  transformation between R and FWHM  ###################### #
# ########################################################################### #


def resolution2fwhm(R, wave=5000.):
    # assert R is positive
    if np.isscalar(R):
        assert R > 0.
    else:
        assert np.all(R > 0.)

    # assert wave is correct in shape
    assert np.isscalar(wave) or len(wave) == len(R)

    return wave / R


def fwhm2resolution(fwhm, wave=5000.):
    # assert fwhm is positive
    if np.isscalar(fwhm):
        assert fwhm > 0.
    else:
        assert np.all(fwhm > 0.)

    # assert wave is correct in shape
    assert np.isscalar(wave) or len(wave) == len(fwhm)

    return wave / fwhm


# ########################################################################### #
# ######################  generate wave array given R  ###################### #
# ########################################################################### #


def _generate_wave_array_R_fixed(wave_start, wave_stop, R=2000.,
                                 over_sample=1.):
    """ generate wave array given a fixed R """
    R_ = over_sample * R - .5
    # determine wave_step_min
    wave_step_min = wave_start / R_
    # wave guess
    wave_step_guess = np.zeros(np.int((wave_stop-wave_start)/wave_step_min))
    wave_guess = np.zeros_like(wave_step_guess)
    wave_step_guess[0] = wave_step_min
    wave_guess[0] = wave_start
    # iterate for real
    for i in np.arange(1, len(wave_guess)):
        wave_guess[i] = wave_guess[i-1] + wave_step_guess[i-1]
        wave_step_guess[i] = wave_guess[i] / R_
    return wave_guess[
        np.logical_and(wave_guess >= wave_start, wave_guess <= wave_stop)]


def _generate_wave_array_R_func(wave_start, wave_stop, R=(lambda x: x),
                                over_sample=1., wave_test_step=1.):
    """ generate wave array given R as a function """
    # determine wave_step_min
    wave_test = np.arange(wave_start, wave_stop, wave_test_step)
    R_test = over_sample * R(wave_test)
    wave_step_min = np.min(wave_test / R_test)
    # wave guess
    wave_guess = np.zeros(np.int(np.ceil((wave_stop-wave_start)/wave_step_min)))
    wave_guess[0] = wave_start
    # iterate for real # single side R !!!
    for i in np.arange(1, len(wave_guess)):
        wave_guess[i] = wave_guess[i-1] + \
                        wave_guess[i-1] / (over_sample * R(wave_guess[i-1]))
    return wave_guess[
        np.logical_and(wave_guess >= wave_start, wave_guess <= wave_stop)]


def generate_wave_array_R(wave_start, wave_stop, R=2000.,
                          over_sample=1., wave_test_step=1.):
    """ generate a wavelength array matching the given R

    Parameters
    ----------
    wave_start: float
        start from this wavelength
    wave_stop: float
        stop at this wavelength
    R: float or function
        specify a fixed R or specify R as a function of wavelength
    over_sample: float
        over-sampling rate, default is 1.
    wave_test_step:
        used to determing the wave_step_min

    Returns
    -------
    wave: array
        an array matching the given R

    Example
    -------
    >>> def R(x): return 0.2*x
    >>> wave_array_R = generate_wave_array_R(4000., 5000., R)

    """
    if np.isscalar(R):
        # if R is scalar
        return _generate_wave_array_R_fixed(
            wave_start, wave_stop, R=R,
            over_sample=over_sample)
    else:
        # if R is a function / Interpolator
        return _generate_wave_array_R_func(
            wave_start, wave_stop, R=R,
            over_sample=over_sample, wave_test_step=wave_test_step)


# ########################################################################### #
# ################  generate wave array given delta_lambda  ################# #
# ########################################################################### #


def _generate_wave_array_delta_lambda_fixed(wave_start, wave_stop,
                                            delta_lambda,
                                            over_sample=1.):
    """ generate a wavelength array matching the given delta_lambda (fixed) """
    return np.arange(wave_start, wave_stop, delta_lambda / over_sample)


def _generate_wave_array_delta_lambda_func(wave_start, wave_stop,
                                           delta_lambda=(lambda x: 1.),
                                           over_sample=1.,
                                           wave_test_step=1.):
    """ generate a wavelength array matching the given delta_lambda
        (specified as a function of wavelength) """
    wave_test = np.arange(wave_start, wave_stop, wave_test_step)
    delta_lambda_min = np.min(delta_lambda(wave_test))
    wave_guess = np.arange(wave_start, wave_stop, delta_lambda_min)
    for i in range(1, len(wave_guess)):
        wave_guess[i] = \
            wave_guess[i-1] + delta_lambda(wave_guess[i-1]) / over_sample
    return wave_guess[
        np.logical_and(wave_guess >= wave_start, wave_guess <= wave_stop)]


def generate_wave_array_delta_lambda(wave_start, wave_stop,
                                     delta_lambda=(lambda x: 1.),
                                     over_sample=1.,
                                     wave_test_step=1.):
    """ generate a wavelength array matching the given delta_lambda a function of wavelength

    Parameters
    ----------
    wave_start: float
        where the wavelength starts
    wave_stop: float
        where the wavelength stops
    delta_lambda: float or function
        specifies the delta_lambda as a fixed number or a function of wavelength
    over_sample: float
        over-sampling
    wave_test_step: float
        tests for the smallest wave_guess step

    Returns
    -------
    wave_guess: array
        the guess of wavelength array

    Example
    -------
    >>> def dl(x): return 0.002*x
    >>> wave_array_dl = generate_wave_array_delta_lambda(4000., 5000., dl)

    """
    if np.isscalar(delta_lambda):
        # if delta_lambda is scalar
        return _generate_wave_array_delta_lambda_fixed(
            wave_start, wave_stop, delta_lambda=delta_lambda,
            over_sample=over_sample)
    else:
        # if delta_lambda is a function / Interpolator
        return _generate_wave_array_delta_lambda_func(
            wave_start, wave_stop, delta_lambda=delta_lambda,
            over_sample=over_sample, wave_test_step=wave_test_step)


# ########################################################################### #
# ############## find spectral R (FWHM) and R_max (FWHM_min) ################ #
# ########################################################################### #


def find_R_for_wave_array(wave):
    """ find the R of wavelength array (sampling resolution array) """
    wave = wave.flatten()
    wave_diff = np.diff(wave)
    wave_diff_ = (wave_diff[1:] + wave_diff[:-1]) / 2.
    return np.hstack((
        wave[0]/wave_diff[0], wave[1:-1]/wave_diff_, wave[-1]/wave_diff[-1]))


def find_R_max_for_wave_array(wave):
    """ find the maximum sampling resolution of a given wavelength array """
    return np.max(find_R_for_wave_array(wave))


def find_delta_lambda_for_wave_array(wave):
    """ find the delta_lambda of wavelength array (delta_lambda array) """
    wave = wave.flatten()
    wave_diff = np.diff(wave)
    wave_diff_ = (wave_diff[1:] + wave_diff[:-1]) / 2.
    return np.hstack((wave_diff[0], wave_diff_, wave[-1]))


def find_delta_lambda_min_for_wave_array(wave):
    """ find the minimum delta_lambda of a given wavelength array """
    return np.min(find_delta_lambda_for_wave_array(wave))


# ########################################################################### #
# ###############################  find Rgk  ################################ #
# ########################################################################### #


def find_Rgk(R_hi=2000., R_lo=500., over_sample=1.):
    """ find Rgk as a function of wavelength

    Parameters
    ----------
    R_hi: float or funtion
        higher resolution (as a function of wavelength)
    R_lo: float or funtion
        lower resolution (as a function of wavelength)
    over_sample: float
        over-sampled resolution, default is 1.

    Returns
    -------
    Rgk: function
        Gaussian Kernel resolution as a function of wavelength

    """
    if np.isscalar(R_hi):
        # if R_hi is a fixed number
        R_hi_ = lambda x: R_hi
    else:
        R_hi_ = R_hi

    if np.isscalar(R_lo):
        # if R_lo is a fixed number
        R_lo_ = lambda x: R_lo
    else:
        R_lo_ = R_lo

    Rgk = lambda x: \
        (over_sample * x / np.sqrt((x/R_lo_(x))**2. - (x/R_hi_(x))**2.))
    return Rgk


# ########################################################################### #
# #########################  Gaussian Kernel ################################ #
# ########################################################################### #


def fwhm2sigma(fwhm):
    return fwhm / (2. * np.sqrt(2. * np.log(2.)))


def sigma2fwhm(sigma):
    return 2. * np.sqrt(2. * np.log(2.)) * sigma


def normalized_gaussian_array(x, b=0., c=1.):
    # a = 1. / (np.sqrt(2*np.pi) * c)
    ngs_arr = np.exp(- (x-b)**2. / (2.*c**2.))
    return ngs_arr / np.sum(ngs_arr)


def generate_gaussian_kernel_array(over_sample_Rgk, sigma_num):
    """ generate gaussian kernel array according to over_sample_Rgk

    Parameters
    ----------
    over_sample_Rgk: float
        over_sample rate
    sigma_num: float
        1 sigma of the Gaussian = sigma_num pixels

    Returns
    -------
    normalized gaussian array

    """
    sigma_pixel_num = fwhm2sigma(over_sample_Rgk)

    array_length = np.fix(sigma_num * sigma_pixel_num)
    if array_length % 2 == 0:
        array_length += 1
    array_length_half = (array_length-1) / 2.

    xgs = np.arange(- array_length_half, array_length_half + 1)
    return normalized_gaussian_array(xgs, b=0., c=sigma_pixel_num)


# most general case
def conv_spec(wave, flux, R_hi=2000., R_lo=500., over_sample_additional=3.,
              gaussian_kernel_sigma_num=5., wave_new=None,
              wave_new_oversample=3., verbose=True, return_type='array'):
    """ to convolve high-R spectrum to low-R spectrum

    Parameters
    ----------
    wave: array
        wavelength
    flux: array
        flux array
    R_hi: float or function
        higher resolution
    R_lo: float or function
        lower resolution
    over_sample_additional: float
        additional over-sample rate
    gaussian_kernel_sigma_num: float
        the gaussian kernel width in terms of sigma
    wave_new: None or float or array
        if None: wave_new auto-generated using wave_new_oversample
        if float: this specifies the over-sample rate
        if voctor: this specifies the new wave_new array
    wave_new_oversample:
        if wave_new is None, use auto-generated wave_new_oversample
    verbose: bool
        if True, print the details on the screen
    return_type: string
        if 'array': return wave and flux as array
        if 'table': retrun spec object

    Returns
    -------
    wave_new, flux_new OR Table([wave, flux])
    
    """
    if verbose:
        start = datetime.datetime.now()
        print('===============================================================')
        print('@laspec.convolution.conv_spec: starting at {}'.format(start))

    # 1. re-format R_hi & R_lo
    assert R_hi is not None and R_lo is not None

    if np.isscalar(R_hi):
        R_hi_ = lambda x: R_hi
    else:
        R_hi_ = R_hi

    if np.isscalar(R_lo):
        R_lo_ = lambda x: R_lo
    else:
        R_lo_ = R_lo

    # 2. find Rgk
    Rgk = find_Rgk(R_hi_, R_lo_, over_sample=1.)

    # 3. find appropriate over_sample
    R_hi_specmax = find_R_max_for_wave_array(wave)
    R_hi_max = np.max(R_hi_(wave))
    over_sample = over_sample_additional * np.fix(np.max([
        R_hi_specmax/Rgk(wave), R_hi_max/Rgk(wave)]))

    # 4. find wave_interp & flux_interp
    if verbose:
        print('@laspec.convolution.conv_spec: interpolating orignal spectrum to wave_interp ...')
    wave_max = np.max(wave)
    wave_min = np.min(wave)
    wave_interp = generate_wave_array_R(wave_min, wave_max,
                                        Rgk, over_sample=over_sample)
    # P = PchipInterpolator(wave, flux, extrapolate=None)
    P = interp1d(wave, flux,
                 kind="linear", bounds_error=False, fill_value=np.nan)
    flux_interp = P(wave_interp)
    assert not np.any(np.isnan(flux_interp))

    # 5. generate Gaussian Kernel array
    if verbose:
        print('@laspec.convolution.conv_spec: generating gaussian kernel array ...')
    gk_array = generate_gaussian_kernel_array(over_sample,
                                              gaussian_kernel_sigma_num)
    # gk_len = len(gk_array)
    # gk_len_half = np.int((gk_len - 1) / 2.)

    # 6. convolution
    if verbose:
        print('@laspec.convolution.conv_spec: convolving ...')
    # convolved_flux = np.convolve(flux_interp, gk_array)[gk_len_half:-gk_len_half]
    # convolved_flux = np.convolve(flux_interp, gk_array, mode="same")
    convolved_flux = signal.fftconvolve(flux_interp, gk_array, mode="same")

    # 7. find new wave array
    if wave_new is None:
        # wave_new is None
        # default: 5 times over-sample
        if verbose:
            print('@laspec.convolution.conv_spec: using default 5 times over-sample wave array ...')
        wave_new = generate_wave_array_R(wave_interp[0], wave_interp[-1],
                                         R_lo, wave_new_oversample)
    elif np.isscalar(wave_new):
        # wave_new specifies the new wave array over_sampling_lo rate
        # default is 5. times over-sample
        if verbose:
            print('@laspec.convolution.conv_spec: using user-specified {:.2f} times over-sample wave array ...'.format(wave_new))
        wave_new = generate_wave_array_R(wave_interp[0], wave_interp[-1],
                                         R_lo, wave_new)
    else:
        # wave_new specified
        if verbose:
            print('@laspec.convolution.conv_spec: using user-specified wave array ...')

    # 8. interpolate convolved flux to new wave array
    if verbose:
        print('@laspec.convolution.conv_spec: interpolating convolved spectrum to new wave array ...')
    # P = PchipInterpolator(wave_interp, convolved_flux, extrapolate=False)
    P = interp1d(wave_interp, convolved_flux,
                 kind="linear", bounds_error=False, fill_value=np.nan)
    flux_new = P(wave_new)
    if verbose:
        stop = datetime.datetime.now()
        print('@laspec.convolution.conv_spec: total time spent: {:.2f} seconds'.format((stop-start).total_seconds()))
        print('===============================================================')

    if return_type == 'array':
        return wave_new, flux_new
    elif return_type == 'table':
        return Table([wave_new, flux_new], names=["wave", "flux"])


def read_phoenix_sun():
    """ read PHOENIX synthetic spectrum for the Sun """
    import laspec
    from astropy.io import fits
    flux_sun = fits.open(laspec.__path__[0] + "/data/phoenix/lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")[0].data
    wave_sun = fits.open(laspec.__path__[0] + "/data/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")[0].data
    return wave_sun, flux_sun


def test_conv_phoenix_sun():
    """ testing convolving PHOENIX synthetic spectrum for the Sun """
    import matplotlib.pyplot as plt
    wave_sun, flux_sun = read_phoenix_sun()
    ind_optical = (wave_sun > 4000) & (wave_sun < 7000)
    wave = wave_sun[ind_optical]
    flux = flux_sun[ind_optical]
    wave_conv, flux_conv = conv_spec(wave, flux, R_hi=3e5, R_lo=2000)

    plt.figure()
    plt.plot(wave, flux)
    plt.plot(wave_conv, flux_conv)
    return
    

if __name__ == '__main__':
    test_conv_phoenix_sun()
