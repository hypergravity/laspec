import numpy as np
from scipy.optimize import minimize


def gauss1d(p, x):
    """ 1D Gaussian function """
    A, mu, sigma = p
    return A / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def costfun(p, x, y, robust=False):
    if p[2] <= 0:
        return np.inf
    model = gauss1d(p, x)
    if not robust:
        return np.sum((model - y) ** 2) / 2.
    else:
        return np.sum(np.abs(model - y))


class GA:
    A = np.nan
    mu = np.nan
    sigma = np.nan

    p = np.ones(3, dtype=float) * np.nan

    bins = None

    def __init__(self, x, bins, initial_guess="auto", method="Powell", robust=False):
        """ to evaluate the Gaussian parameters of the residuals *x* """
        # record x & bins
        self.x = np.array(x)
        self.bins = np.array(bins)
        self.robust = robust
        self.method = method
        # counts
        self.N = np.nansum(np.isfinite(x))
        # histogram
        self.H, self.be = np.histogram(x, bins=bins)
        # bin centers
        self.bc = (self.be[:-1] + self.be[1:]) / 2
        # initial guess for Gaussian parameters
        if initial_guess == "std":
            # start from a standard Gaussian function
            self.p0 = np.array([len(x), 0., 1.])
        elif initial_guess == "auto":
            # start from mean & std
            self.p0 = np.array([len(x), np.nanmedian(x), np.nanstd(x)])
        else:
            raise ValueError("Invalid initial_guess parameter!")
        # search for Gaussian parameters
        self.opt = minimize(costfun, x0=self.p0, args=(self.bc, self.H, self.robust), method=self.method)
        if self.N > 0:
            self.p = self.opt.x
        else:
            self.p = self.opt.x * np.nan
        # unpack parameters
        self.A, self.mu, self.sigma = self.p

        return

    def return_dict(self):
        """ compatible with old code """
        return dict(N=self.N, p=self.p, bc=self.bc, H=self.H)

    def replicate(self, xx):
        """ replicate the 1d best-fitted Gaussian function """
        xx = np.array(xx)
        return xx, gauss1d([self.A, self.mu, self.sigma], xx)

    def __repr__(self):
        return "<GA N={:d}, A={:0.2f}, mu={:0.2f}, sigma={:0.2f}>".format(self.N, *self.p)