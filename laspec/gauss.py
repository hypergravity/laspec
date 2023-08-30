import numpy as np
from scipy.optimize import least_squares


def model_gauss1(p, x):
    amp, mean, sigma = p
    return amp * np.exp(- (x - mean) ** 2 / sigma ** 2 / 2)


def res_gauss1(p, x, y):
    return y - model_gauss1(p, x)


def res_gauss2(p, x, y):
    p1 = p[:3]
    p2 = p[3:]
    return y - model_gauss1(p1, x) - model_gauss1(p2, x)


def res_gauss2_eqsigma(p, x, y):
    p1 = p[:3]
    p2 = p[[3, 4, 2]]
    return y - model_gauss1(p1, x) - model_gauss1(p2, x)


class GaussianFitter:
    def __init__(self):
        pass

    def fit1(self, x, y, pmin=(1e-6, -500, 1), pmax=(np.inf, 500, 500), p1_init=(1, 0, 5)):
        p1_opt = least_squares(res_gauss1, p1_init, args=(x, y), bounds=(pmin, pmax))
        return p1_opt

    def fit2_eqsigma(self, x, y, p2min=(1e-3, -500, 5, 1e-3, -500),
                     p2max=(np.inf, 500, 100, np.inf, 500),
                     p2init=(1, -50, 20, 1, 50, 20)):
        # random initialization for p2
        p2init = np.max([p2min, p2init], axis=0)
        p2init = np.min([p2max, p2init], axis=0)
        p2opt = least_squares(res_gauss2_eqsigma, p2init, args=(x, y), bounds=(p2min, p2max))
        return p2opt

    def fit2(self, x, y, pmin=(1e-6, -500, 1), pmax=(np.inf, 500, 500), p1_init=(1, 0, 5), nmc=0, dposmax=500):
        p1_opt = self.fit1(x, y, pmin, pmax, p1_init=p1_init)
        if nmc <= 0:
            # random initialization for p2
            p2_init = [p1_opt.x[0], p1_opt.x[1], p1_opt.x[2] / 2, p1_opt.x[0],
                       p1_opt.x[1] + np.random.uniform(-1, 1) * dposmax, p1_opt.x[2] / 2, ]
            p2_min = tuple([*pmin, *pmin])
            p2_max = tuple([*pmax, *pmax])
            p2_init = np.max([p2_min, p2_init], axis=0)
            p2_init = np.min([p2_max, p2_init], axis=0)
            p2_opt = least_squares(res_gauss2, p2_init, args=(x, y), bounds=(p2_min, p2_max))
            return p2_opt
        else:
            # random initialization for p2 + Monte Carlo
            p2_mc = np.zeros((nmc, 5))
            cost_mc = np.zeros((nmc))
            p2_min = tuple([*pmin, *pmin])
            p2_max = tuple([*pmax, *pmax])
            for imc in range(nmc):
                p2_init = [p1_opt.x[0], p1_opt.x[1], p1_opt.x[2] / 2, p1_opt.x[0],
                           p1_opt.x[1] + np.random.uniform(-1, 1) * dposmax, p1_opt.x[2] / 2, ]
                p2_init = np.max(np.array([p2_min, p2_init]), axis=0)
                p2_init = np.min(np.array([p2_max, p2_init]), axis=0)
                p2_opt = least_squares(res_gauss2, p2_init, args=(x, y), bounds=(p2_min, p2_max))
                p2_mc[imc] = p2_opt.x
                cost_mc[imc] = p2_opt.cost
            p2_init = p2_mc[np.argmin(cost_mc)]
            p2_init = np.max(np.array([p2_min, p2_init]), axis=0)
            p2_init = np.min(np.array([p2_max, p2_init]), axis=0)
            p2_opt = least_squares(res_gauss2, p2_init, args=(x, y), bounds=(p2_min, p2_max))
            return p2_opt

    def replicate1(self, p1, x):
        return model_gauss1(p1, x)

    def replicate2(self, p2, x):
        return np.array([model_gauss1(p2[:3], x), model_gauss1(p2[3:], x)])
