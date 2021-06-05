import numpy as np
import matplotlib.pyplot as plt
# this module is to produce artifacts


def make_flat(npix=1000, vmin=0, vmax=1):
    return np.ones(npix, dtype=float) * np.random.uniform(vmin, vmax)


def make_peak(xx, mu=0., sigma=1., peakmax=1, kind="gauss"):
    """ make a peak at position mu with equivalent width sigma """
    assert kind in ["gauss", "exp", "cauchy"]
    if kind == "gauss":
        # Gaussian
        return peakmax * np.exp(-(xx - mu) ** 2 / sigma ** 2 / 2)
    elif kind == "exp":
        # Exponential / Laplace
        delta = sigma / 1.414  # equivalent standard deviation
        return peakmax * np.exp(- np.abs((xx - mu) / delta))
    elif kind == "cauchy":
        # Cauchy / Lorentzian
        gamma = sigma / 1.485  # equivalent interquartile
        return peakmax * gamma ** 2 / (gamma ** 2. + (xx - mu) ** 2)


def make_random_signal(xx, yy=None,
                       peaklam=3, peaksigma_min=.1, peaksigma_scale=.5, peakmax_scale=5.,
                       lfwidth=200, lfamp=.1,
                       flat_min=-.3, flat_max=.3,
                       chunklam=2, chunkwidth_min=1, chunkwidth_scale=2, chunkvalue=0):
    """ make random signal """
    _signal = np.zeros_like(xx, dtype=float)
    # add peaks
    npeaks = np.random.poisson(peaklam)
    sigma = np.random.exponential(peaksigma_scale, size=npeaks) + peaksigma_min
    mu = np.random.uniform(xx[0], xx[-1], size=npeaks)
    peakmax = np.random.exponential(scale=peakmax_scale, size=npeaks)
    for ipeak in range(npeaks):
        _signal += make_peak(xx, mu[ipeak], sigma=sigma[ipeak], peakmax=peakmax[ipeak],
                             kind=np.random.choice(["cauchy", "exp", "gauss"]))
    # add low freq
    _signal += make_low_freq(xx,
                             halfperiod=np.random.exponential(lfwidth)+lfwidth,
                             halfamplitude=np.random.normal(0, scale=lfamp))
    if yy is None:
        yyn = _signal + np.random.uniform(flat_min, flat_max)
    else:
        # add flat
        yyn = yy + _signal + np.random.uniform(flat_min, flat_max)
        # yyn = make_random_mask(yyn, pct=pct, value=value)

    # mask chunks
    nchunks = np.random.poisson(chunklam)
    chunkwidths = np.random.exponential(chunkwidth_scale, size=nchunks) + chunkwidth_min
    chunkpos = np.random.uniform(xx[0], xx[-1], size=nchunks)
    for ichunk in range(nchunks):
        yyn[(xx > chunkpos[ichunk]) & (xx < chunkpos[ichunk] + chunkwidths[ichunk])] = chunkvalue
    return yyn


def make_random_mask(yy, pct=0.1, value=1):
    """ randomly set a fraction of yy elements to value """
    return np.where(np.random.uniform(0, 1, size=yy.shape) < pct, value, yy)


def test_make_random_signal():
    plt.figure()
    wave = np.linspace(5000, 5300, 3347)
    for i in range(10):
        plt.plot(wave, make_random_signal(wave, lam=3, sigma_min=.2, sigma_scale=.1))


def test_make_peak():
    fig = plt.figure();
    wave = np.arange(5000, 6000)
    mu = 5500.
    sigma = 2.
    peakmax = 1.
    plt.plot(make_peak(wave, mu, sigma, peakmax, kind="gauss"), label="gauss")
    plt.plot(make_peak(wave, mu, sigma, peakmax, kind="exp"), label="exp")
    plt.plot(make_peak(wave, mu, sigma, peakmax, kind="cauchy"), label="cauchy")
    plt.legend()
    plt.ylim(0, 1.5)


def make_low_freq(xx, halfperiod=10, halfamplitude=1.):
    """ make low frequncy signals """
    omega = np.pi / halfperiod
    return halfamplitude * np.sin(omega * xx + np.random.uniform(0, 2*np.pi))


def test_make_low_freq():
    fig = plt.figure()
    wave = np.arange(5000, 6000)
    plt.plot(wave, 1 + make_low_freq(wave, halfperiod=100, halfamplitude=0.1, ), label="100")
    plt.plot(wave, 1 + make_low_freq(wave, halfperiod=500, halfamplitude=0.1, ), label="500")
    plt.plot(wave, 1 + make_low_freq(wave, halfperiod=1000, halfamplitude=0.1, ), label="1000")
    plt.plot(wave, 1 + make_low_freq(wave, halfperiod=2000, halfamplitude=0.1, ), label="2000")
    plt.plot(wave, 1 + make_low_freq(wave, halfperiod=5000, halfamplitude=0.1, ), label="5000")
    plt.legend()
    plt.ylim(0, 1.5)


def make_bad_chunks(xx, lam=3, scale_width=10):
    npeaks = np.random.poisson(lam)
    return


if __name__ == "__main__":
    wave = np.linspace(5000, 5300, 3347)
    flux = np.ones_like(wave)
    flux_noise = make_random_signal(wave, flux,
                                    peaklam=3, peaksigma_min=.1, peaksigma_scale=.3, peakmax_scale=3.,
                                    lfwidth=300, lfamp=.1,
                                    flat_min=-.3, flat_max=.3,
                                    chunklam=3, chunkwidth_min=.3, chunkwidth_scale=1, chunkvalue=0)
    plt.figure()
    plt.plot(wave, flux)
    plt.plot(wave, flux_noise)

