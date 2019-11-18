from bisect import bisect

import numpy as np


class Interp1q:
    def __init__(self, x, y, method="linear", fill_value=np.nan, cushion=1e5,
                 issorted=True):

        if issorted:
            self.x = np.asarray(x)
            self.y = np.asarray(y)
        else:
            ind_sort = np.argsort(x)
            self.x = np.asarray(x)[ind_sort]
            self.y = np.asarray(y)[ind_sort]

        self.cusion = cushion

        assert self.x.ndim == 1
        assert self.y.shape == self.x.shape

        self.x_lo = self.x[0]
        self.x_hi = self.x[-1]
        self.y_lo = self.y[0]
        self.y_hi = self.y[-1]

        # self.ind_lo, self.ind_hi = self.x[[0, -1]]
        self.fill_value = fill_value

    def __call__(self, xi):
        xi = np.asarray(xi)

        x_lo = np.min((self.x_lo, np.min(xi))) - self.cusion
        x_hi = np.max((self.x_hi, np.max(xi))) + self.cusion
        x = np.hstack((x_lo, self.x, x_hi))
        y = np.hstack((self.y_lo, self.y, self.y_hi))

        # reshape xi
        xi_shape = xi.shape
        xi_flatten = xi.flatten()

        ind_bisect = np.asarray(
            [bisect(x, xi_flatten[i]) for i in range(len(xi_flatten))])
        w_l = x[ind_bisect] - xi_flatten
        w_r = xi_flatten - x[ind_bisect - 1]
        yi = (w_l * y[ind_bisect - 1] + w_r * y[ind_bisect]) / (w_l + w_r)

        # fill values for outbounds
        yi = np.where(
            np.logical_and(xi_flatten >= self.x_lo, xi_flatten <= self.x_hi),
            yi, self.fill_value)
        return yi.reshape(xi_shape)


if __name__ == "__main__":
    print(Interp1q([0, 1, 2], [0, 1, 2])([-1, 0, 1, 2, 3]))
