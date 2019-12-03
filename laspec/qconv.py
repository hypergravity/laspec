import datetime

import numpy as np
from scipy import signal

from .wavelength import wave_log10


def Gaussian_kernel(dRV_sampling=0.1, dRV_Gk=2.3548200450309493, n_sigma_Gk=5.):
    """ Gaussian kernel

    Parameters
    ----------
    dRV_sampling:
         the sampling rate of the input spectrum (km/s)
    dRV_Gk:
        the sampling rate of the Gaussian kernel (km/s)
    n_sigma_Gk:
        the length of the Gaussian kernel, in terms of sigma

    Return
    ------
    Gaussian kernel

    """
    # FWHM --> sigma
    sigma = dRV_Gk/dRV_sampling/(2*np.sqrt(2*np.log(2)))
    
    # determine X
    npix_half = np.int(sigma*n_sigma_Gk)
    # npix = 2 * npix_half + 1
    x = np.arange(-npix_half, npix_half+1)
        
    # normalized Gaussian kernel
    kernel = np.exp(-.5*(x/sigma)**2.)
    kernel /= np.sum(kernel)
    return kernel


def Rotation_kernel(dRV_sampling=10, vsini=100, epsilon=0.6, osr_kernel=3):
    """ Rotation kernel

    Parameters
    ----------
    dRV_sampling:
        the sampling rate of the input spectrum (km/s)
    vsini:
        v*sin(i)  (km/s)
    epsilon:
        the limb-darkening coefficient (0, 1)
    osr_kernel:
        the over-sampling rate of the kernel

    Return
    ------
    rotation kernel

    """
    osr_kernel = np.int(np.floor(osr_kernel / 2)) * 2 + 1  # an odd number
    # determine X
    npix_half = np.int(np.floor(vsini / dRV_sampling))
    npix_half = np.int(npix_half * osr_kernel + 0.5 * (osr_kernel - 1))
    # npix = 2 * npix_half + 1
    vvl = np.arange(-npix_half, npix_half + 1) / osr_kernel * dRV_sampling / vsini

    # rotation kernel
    denominator = np.pi * vsini * (1.0 - epsilon / 3.0)
    c1 = 2.0 * (1.0 - epsilon) / denominator
    c2 = 0.5 * np.pi * epsilon / denominator
    kernel = np.where(np.abs(vvl) <= 1,
                      c1 * np.sqrt(1.0 - vvl ** 2) + c2 * (1.0 - vvl ** 2), 0)
    kernel = kernel.reshape(-1, osr_kernel).sum(axis=1)
    kernel /= np.sum(kernel)
    return kernel


def conv_spec_Gaussian(wave, flux, dRV_Gk=None,
                       R_hi=3e5, R_lo=2000., n_sigma_Gk=5., 
                       interp=True, osr_ext=3., wave_new=None):
    """ to convolve instrumental broadening (high-R spectrum to low-R spectrum)

    Parameters
    ----------
    wave: array
        wavelength
    flux: array
        flux array
    dRV_Gk: float
        the FWHM of the Gaussian kernel (km/s)
        if None, determined by R_hi and R_lo
    R_hi: float
        higher resolution
    R_lo: float
        lower resolution
    n_sigma_Gk: float
        the gaussian kernel width in terms of sigma
    interp: bool
        if True, interpolate to log10 wavelength
    osr_ext:
        the extra oversampling rate if interp is True.
    wave_new:
        if not None, return convolved spectrum at wave_new
        if None, return log10 spectrum

    Returns
    -------
    wave_new, flux_new

    """
    if interp:
        wave_interp = wave_log10(wave, osr_ext=osr_ext)
        flux_interp = np.interp(wave_interp, wave, flux) # 10 times faster
        # flux_interp = interp1d(wave, flux, kind="linear", bounds_error=False)(wave_interp)
    else:
        wave_interp = np.asarray(wave)
        flux_interp = np.asarray(flux)
    assert np.all(np.isfinite(flux_interp))
    
    # evaluate the RV sampling rate via log10(wave) sampling rate
    # d(log10(wave)) = z / ln(10) = d(RV)/(c*ln(10))
    # --> d(RV) = c*ln(10)*d(log10(wave))
    dRV_sampling = 299792.458 * np.log(10) * (np.log10(wave_interp[1])-np.log10(wave_interp[0]))
    if dRV_Gk is None:
        R_Gk = R_hi*R_lo / np.sqrt(R_hi**2-R_lo**2) # Gaussian kernel resolution
        dRV_Gk = 299792.458 / R_Gk # Gaussian kernel resolution (FWHM) --> RV resolution
    # generate Gaussian kernel
    Gk = Gaussian_kernel(dRV_sampling, dRV_Gk, n_sigma_Gk=n_sigma_Gk)
    
    # convolution
    # fftconvolve is the most efficient ...currently
    flux_conv = signal.fftconvolve(flux_interp, Gk, mode="same")
    # flux_conv = np.convolve(flux_interp, Gk, mode="same")
    # flux_conv = signal.convolve(flux_interp, Gk, mode="same")
    
    # interpolate to new wavelength grid if necessary
    if wave_new is None:
        return wave_interp, flux_conv
    else:
        flux_conv_interp = np.interp(wave_new, wave_interp, flux_conv)
        return wave_new, flux_conv_interp
    

def conv_spec_Rotation(wave, flux, vsini=100., epsilon=0.6,
                       interp=True, osr_ext=3., wave_new=None):
    """ to convolve instrumental broadening (high-R spectrum to low-R spectrum)

    Parameters
    ----------
    wave: array
        wavelength
    flux: array
        flux array
    vsini: float
        the projected stellar rotational velocity (km/s)
    epsilon: float
        0 to 1, the limb-darkening coefficient, default 0.6.
    interp: bool
        if True, interpolate to log10 wavelength
    osr_ext:
        the extra oversampling rate if interp is True.
    wave_new:
        if not None, return convolved spectrum at wave_new
        if None, return log10 spectrum

    Returns
    -------
    wave_new, flux_new OR Table([wave, flux])

    """
    assert vsini > 0.
    if interp:
        wave_interp = wave_log10(wave, osr_ext=osr_ext)
        flux_interp = np.interp(wave_interp, wave, flux) # 10 times faster
        # flux_interp = interp1d(wave, flux, kind="linear", bounds_error=False)(wave_interp)
    else:
        wave_interp = np.asarray(wave)
        flux_interp = np.asarray(flux)
    #assert np.all(np.isfinite(flux_interp))
    
    # evaluate the RV sampling rate via log10(wave) sampling rate
    # d(log10(wave)) = z / ln(10) = d(RV)/(c*ln(10))
    # --> d(RV) = c*ln(10)*d(log10(wave))
    dRV_sampling = 299792.458 * np.log(10) * (np.log10(wave_interp[1])-np.log10(wave_interp[0]))

    # generate rotation kernel
    Rk = Rotation_kernel(dRV_sampling, vsini, epsilon=epsilon, osr_kernel=15)

    # convolution
    # fftconvolve is the most efficient ...currently
    flux_conv = signal.fftconvolve(flux_interp, Rk, mode="same")
    # flux_conv = np.convolve(flux_interp, Gk, mode="same")
    # flux_conv = signal.convolve(flux_interp, Gk, mode="same")
    
    # interpolate to new wavelength grid if necessary
    if wave_new is None:
        return wave_interp, flux_conv
    else:
        flux_conv_interp = np.interp(wave_new, wave_interp, flux_conv)
        return wave_new, flux_conv_interp


def read_phoenix_sun():
    """ read PHOENIX synthetic spectrum for the Sun """
    import laspec
    from astropy.io import fits
    flux_sun = fits.open(laspec.__path__[0] + "/data/phoenix/lte05800-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")[0].data
    wave_sun = fits.open(laspec.__path__[0] + "/data/phoenix/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")[0].data
    return wave_sun, flux_sun


def test_convolution():
    """ testing convolving PHOENIX synthetic spectrum for the Sun """
    import matplotlib.pyplot as plt
    from laspec.convolution import conv_spec
    wave_sun, flux_sun = read_phoenix_sun()
    ind_optical = (wave_sun > 4000) & (wave_sun < 6000)
    wave = wave_sun[ind_optical]
    flux = flux_sun[ind_optical]
    
    print("testing laspec.convolution.conv_spec ...  ")
    t0 = datetime.datetime.now()    
    wave_conv1, flux_conv1 = conv_spec(wave, flux, R_hi=3e5, R_lo=2000, verbose=False)
    print("time spent: ", datetime.datetime.now() - t0, "npix = ", wave_conv1.shape[0])
    
    wave_interp = wave_log10(wave, osr_ext=3)
    flux_interp = np.interp(wave_interp, wave, flux) # 10 times faster
    
    print("testing laspec.qconv.conv_spec_Gaussian ...  ")
    t0 = datetime.datetime.now()
    wave_conv2, flux_conv2 = conv_spec_Gaussian(wave_interp, flux_interp, R_hi=3e5, R_lo=2000, interp=False)
    print("time spent: ", datetime.datetime.now() - t0, "npix = ", wave_conv2.shape[0])
    
    print("testing laspec.qconv.conv_spec_Rotation ...  ")
    t0 = datetime.datetime.now()    
    wave_conv3, flux_conv3 = conv_spec_Rotation(wave_conv2, flux_conv2, vsini=100, epsilon=0.6, interp=False)
    print("time spent: ", datetime.datetime.now() - t0, "npix = ", wave_conv3.shape[0])
    
    plt.figure()
    plt.plot(wave, flux)
    plt.plot(wave_conv1, flux_conv1, label="conv_spec")
    plt.plot(wave_conv2, flux_conv2, label="conv_spec_Gaussian")
    plt.plot(wave_conv3, flux_conv3, label="conv_spec")
    plt.legend(loc="upper right")
    return
    

if __name__ == '__main__':
    test_convolution()
