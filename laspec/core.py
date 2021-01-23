import numpy as np


def wmean(x, w=None):
    """ weighted mean """
    if w is None:
        return np.mean(x)
    else:
        return np.sum(x * w) / np.sum(w)


def wstd(x, w=None):
    """
    Weighted standard deviation

    Parameters
    ----------
    x : array
        data
    w : array
        weights

    Returns
    -------
    wstd

    Ref
    ---
    https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf

    """
    if w is None:
        w = np.ones_like(x)
    if len(x) < 1:
        return 0.
    else:
        _n_nonzero = np.sum(w>0)
        if _n_nonzero > 1:
            _wmean = wmean(x, w)
            return np.sqrt(np.sum(w*(x-_wmean)**2) / np.sum(w) /(_n_nonzero-1)*_n_nonzero)
        else:
            return 0.

