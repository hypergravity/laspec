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


def wpercentile(x, w, q, eps=0.):
    """ weighted percentile """
    x = np.asarray(x)
    w = np.asarray(w)
    q = np.asarray(q)
    indsort = np.argsort(x)
    x_sorted = x[indsort]
    w_sorted = w[indsort]
    x_sorted_appended = np.hstack((x_sorted[0]-eps, x_sorted, x[-1]+eps))
    w_sorted_cumsum = np.cumsum(w_sorted)
    w_sorted_cumsum_appended = np.hstack((0, w_sorted_cumsum, w_sorted_cumsum[-1]))
    return np.interp(q/100, w_sorted_cumsum_appended/w_sorted_cumsum_appended[-1], x_sorted_appended)


def test_wpercentile():
    x = [0,1,2,2,2,2,2,3,4,5]
    w = [1,1,1,1,1,1,1,1,1,1]
    q = [16, 50]
    print(wpercentile(x, w, q, eps=0))
    return
