import numpy as np
from astropy import table
import matplotlib.pyplot as plt


class RandomWalkMinimizer:
    """ Random Walk Minimizer """

    def __init__(self, fun, x0, dx, maxiter=20, args=[], kwargs={}, optind=None,
                 verbose=False, random="normal"):
        """ multiple loops over different dx

        Parameters
        ----------
        fun:
            objective function
        x0: array like
            initial guess of x
        dx:
            a list of different scales. e.g. []
        maxiter:
            number of max iterations
        args, kwargs:
            fun(x, *args, **kwargs)
        optind:
            a subset ind of parameters
        verbose: bool
            if True, print verbose
        random: [uniform, normal]
            type of random number generator of

        """
        self.fun = fun
        self.x0 = x0
        self.dx = dx
        self.maxiter = maxiter
        self.args = args
        self.kwargs = kwargs
        self.optind = optind
        self.verbose = verbose

        assert random in ["uniform", "normal"]
        self.random = random

        self.xhist = []

    def __call__(self, x):
        return np.float(self.fun(np.array(x), *self.args, **self.kwargs))

    def run(self, fun=None, x0=None, dx=None, maxiter=None, args=None,
            kwargs=None, optind=None, verbose=None, random=None):

        x0 = x0 if x0 is not None else self.x0
        xhist = []
        opthist = []
        cloop = 0
        for _dx in dx if dx is not None else self.dx:
            for _optind in optind if optind is not None else self.optind:
                cloop += 1
                info = cloop
                _result = self.minimize(
                    fun=fun if fun is not None else self.fun,
                    x0=x0,
                    dx=_dx,
                    maxiter=maxiter if maxiter is not None else self.maxiter,
                    args=args if args is not None else self.args,
                    kwargs=kwargs if kwargs is not None else self.kwargs,
                    optind=_optind,
                    verbose=verbose if verbose is not None else self.verbose,
                    info=info if info is not None else self.info,
                    random=random if random is not None else self.random,
                )
                opthist.append(_result)
                xhist.append(x0)

        return dict(x=_result["x"],
                    nfev=np.sum([_["nfev"]for _ in opthist]),
                    niter=np.sum([_["niter"] for _ in opthist]),
                    msg=table.vstack([table.Table(_["msg"]) for _ in opthist]))

    @staticmethod
    def minimize(fun, x0, dx, maxiter=10, args=None, kwargs={},
                 optind=None, verbose=False, info="", random="normal"):
        """ a single

        Parameters
        ----------
        fun:
            objective function
        x0: array like
            initial guess of x
        dx:
            a list of different scales. e.g. []
        maxiter:
            number of max iterations
        args:
            arguments
        kwargs:
            keyword arguments
        optind:
            a subset ind of parameters, e.g., [5, 7]
        verbose: bool
            if True, print verbose
        random: [uniform, normal]
            type of random number generator
        info:
            additional info appended in msg

        """
        if args is None:
            args = []

        # evaluate cost0
        x0 = np.asarray(x0, float)
        dx = np.asarray(dx, float)
        ndim = len(x0)
        cost0 = fun(x0, *args, **kwargs)

        # opt ind
        if optind is None:
            optmask = np.ones(ndim, dtype=bool)
        elif len(optind) == ndim:
            optmask = np.asarray(optind, dtype=bool)
        else:
            optmask = np.zeros(ndim, dtype=bool)
            optmask[optind] = True

        # max number of iterations --> maxiter
        niter = 0
        nfev = 0
        iiter = 0
        if verbose:
            print("X{} = {}, cost={}".format(niter, x0, cost0))
        # messages
        this_msg = dict(x=x0,
                        cost=cost0,
                        accept=True,
                        info=info,
                        nfev=0,
                        niter=0)
        msg = [this_msg]

        while iiter < maxiter:
            # random step
            if random == "normal":
                x1 = x0 + np.random.normal(loc=0, scale=dx,
                                           size=x0.shape) * optmask
            elif random == "uniform":
                x1 = x0 + np.random.uniform(low=-dx, high=dx,
                                            size=x0.shape) * optmask
            else:
                raise ValueError("Invalid random type! [{}]".format(random))

            cost1 = fun(x1, *args, **kwargs)
            nfev += 1

            if cost1 <= cost0:
                x0 = np.copy(x1)
                cost0 = cost1
                iiter = 0
                niter += 1
                if verbose:
                    print("X{} = {}, cost={}".format(niter, x0, cost0), iiter)

                # messages
                this_msg = dict(x=x1,
                                cost=cost1,
                                accept=True,
                                info=info,
                                nfev=nfev,
                                niter=niter)
                msg.append(this_msg)

            else:
                iiter += 1

                # messages
                this_msg = dict(x=x1,
                                cost=cost1,
                                accept=False,
                                info=info,
                                nfev=nfev,
                                niter=niter)
                msg.append(this_msg)

        return dict(x=x0, nfev=nfev, niter=niter, msg=table.Table(msg))


def test1():
    def fun(x, asin=1):
        return asin * np.sin(x) + x ** 2

    xx = np.linspace(-10, 10, 1000)
    plt.figure()
    plt.plot(xx, fun(xx, 10))

    res = RandomWalkMinimizer.minimize(fun, x0=[10], dx=10, maxiter=10,
                                       args=[], optind=None, verbose=True, )
    print("")
    print(res)


def test2():
    def fun(x, asin=1):
        return np.sum(asin * np.sin(x) + x ** 2)

    res = RandomWalkMinimizer.minimize(fun, x0=[-2, -1], dx=1, maxiter=20,
                                       args=[], optind=[0], verbose=True,
                                       random="uniform")
    print("")
    print(res)


def test3():
    def fun(x, asin=1):
        return np.sum(asin * np.sin(x) + x ** 2)

    rwm = RandomWalkMinimizer(fun, x0=[10, 10], dx=[[10, 10], [2, 2]],
                              maxiter=20, args=[], optind=[[0], [1]],
                              verbose=True, random="uniform")
    res = rwm.run()
    print("")
    print(res)


if __name__ == "__main__":
    test3()
