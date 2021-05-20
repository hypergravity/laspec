import numpy as np
import tensorflow as tf
from laspec.neural_network import NN
from scipy.optimize import minimize, curve_fit
import joblib


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def elu(x, alpha=0.01):
    return np.where(x > 0, x, alpha(np.exp(x) - 1.))


class SlamPlus:
    def __init__(self, tr_flux, tr_label, tr_weight=None, wave=None):
        # set training set
        self.tr_flux = np.asarray(tr_flux, dtype=float)
        self.tr_label = np.asarray(tr_label, dtype=float)
        self.tr_label_min = np.min(self.tr_label, axis=0)
        self.tr_label_max = np.max(self.tr_label, axis=0)
        self.tr_flux_min = np.min(self.tr_flux, axis=0)
        self.tr_flux_max = np.max(self.tr_flux, axis=0)
        self.tr_flux_scaled = (self.tr_flux - self.tr_flux_min)/(self.tr_flux_max - self.tr_flux_min) - 0.5
        self.tr_label_scaled = (self.tr_label - self.tr_label_min) / (self.tr_label_max - self.tr_label_min) - 0.5
        self.history = None
        self.wave = wave

        # set parameters
        assert self.tr_flux.shape[0] == self.tr_label.shape[0]
        assert self.tr_flux.ndim == 2 and self.tr_label.ndim == 2
        self.nstar, self.npix = self.tr_flux.shape
        self.ndim = self.tr_label.shape[1]

        # set weight
        if tr_weight is None:
            self.tr_weight = np.ones(self.nstar, dtype=float)
        else:
            self.tr_weight = np.asarray(tr_weight, dtype=float)
        self.nnweights = []

        # default parameters
        self.nlayer = 0
        self.activation = "leakyrelu"
        self.alpha = 0.
        self.w = 0
        self.b = 0
        self.model = None
        return

    def initialize(self, nhidden=(100, 300,), activation="leakyrelu", alpha=0.01):
        from collections.abc import Iterable
        if isinstance(nhidden, Iterable):
            self.nlayer = len(nhidden)
        else:
            assert isinstance(nhidden, int)
            self.nlayer = 1
        self.activation = activation
        self.alpha = alpha

        # initialize NN regressor
        self.model = NN(kind="slam", ninput=self.ndim, nhidden=nhidden, noutput=self.npix,
                        activation=activation, alpha=alpha)
        self.model.summary()

    @staticmethod
    def get_gpu():
        NN.get_gpu()
        return

    @staticmethod
    def set_gpu(device=0):
        NN.set_gpu(device=device)
        return

    def train(self, test_size=0.1, random_state=0, epochs=1000, batch_size=100,  # training parameters
              optimizer="adam", learning_rate=1e-4, loss="mae", metrics=['mse', "mae"],
              patience_earlystopping=5, patience_reducelronplateau=3, factor_reducelronplateau=0.5, filepath="",
              ):
        """ train all pixels """
        # set optimizer
        assert optimizer in ["adam", "sgd"]
        if optimizer == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        # set callbacks
        self.model.set_callbacks(monitor_earlystopping="val_loss",
                                 patience_earlystopping=patience_earlystopping,
                                 monitor_modelcheckpoint="val_loss",
                                 filepath=filepath,
                                 monitor_reducelronplateau="val_loss",
                                 patience_reducelronplateau=patience_reducelronplateau,
                                 factor_reducelronplateau=factor_reducelronplateau)
        # train pixels
        self.history = self.model.train(self.tr_label_scaled, self.tr_flux_scaled,
                                        batch_size=batch_size, sw=self.tr_weight,
                                        test_size=test_size, optimizer=optimizer, epochs=epochs,
                                        loss=loss, metrics=metrics, random_state=random_state)
        # get best model
        if filepath not in ["", None]:
            self.model.model = tf.keras.models.load_model(filepath)
        # get weights
        new_weights = self.model.model.get_weights()
        self.w = [new_weights[ilayer * 2].T for ilayer in range(self.nlayer + 1)]
        self.b = [new_weights[ilayer * 2 + 1].reshape(-1, 1) for ilayer in range(self.nlayer + 1)]

        return SlamPredictor(self.w, self.b, self.alpha, self.tr_label_min, self.tr_label_max,
                             self.tr_flux_min, self.tr_flux_max, self.wave)


class SlamPredictor:
    def __init__(self, w, b, alpha, xmin, xmax, ymin, ymax, wave=None):

        self.alpha = alpha
        self.w = w
        self.b = b
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xmean = .5*(xmin+xmax)
        self.nlayer = len(w) - 1
        self.wave = wave

    @property
    def get_coef_dict(self):
        return dict(w=self.w, b=self.b, alpha=self.alpha)

    def predict_one_spectrum(self, x):
        """ predict one spectrum """
        # scale label
        xsT = ((np.asarray(x)-self.xmin)/(self.xmax-self.xmin)).reshape(-1, 1) - 0.5
        return (nneval(xsT, self.w, self.b, self.alpha, self.nlayer).reshape(-1) + 0.5) * (self.ymax-self.ymin) + self.ymin

    # def predict_multiple_spectra(self, x):
    #     # scale label
    #     xs = ((np.asarray(x) - self.xmin) / (self.xmax - self.xmin))
    #     # multiple spectra
    #     xs = xs.T
    #     if self.nlayer == 2:
    #         return nneval(xs, self.alpha, self.nlayer).T * (self.ymax-self.ymin) + self.ymin
    #     elif self.nlayer == 3:
    #         return nneval(self.w, self.b, xs, self.alpha).T * (self.ymax-self.ymin) + self.ymin

    def optimize(self, flux_obs, flux_err=None, pw=2, method="Nelder-Mead"):
        return minimize(cost, self.xmean, args=(self, flux_obs, flux_err, pw), method=method)

    def curve_fit(self, flux_obs, flux_err=None, p0=None, method="lm", bounds=(-np.inf, np.inf), **kwargs):
        if p0 is None:
            p0 = self.xmean
        return curve_fit(model_func, self, flux_obs, p0=p0, sigma=flux_err, absolute_sigma=True,
                         method=method, bounds=bounds, **kwargs)

    def curve_fit_multiple(self, flux_obs, flux_err=None, p0=None, method="lm", bounds=(-np.inf, np.inf),
                           n_jobs=2, verbose=10, backend="loky", batch_size=10, **kwargs):
        flux_obs = np.asarray(flux_obs)
        nobs = flux_obs.shape[0]
        if flux_err is None:
            flux_err = [None for i in range(nobs)]
        else:
            flux_err = np.asarray(flux_err)
        if p0 is None:
            p0 = self.xmean
        pool = joblib.Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend, batch_size=batch_size)
        res = pool(joblib.delayed(curve_fit)(
            model_func, self, flux_obs[i], p0=p0, sigma=flux_err[i], absolute_sigma=True,
            method=method, bounds=bounds, **kwargs) for i in range(nobs))
        popt = np.array([res[i][0] for i in range(nobs)])
        pcov = np.array([res[i][1] for i in range(nobs)])
        return popt, pcov


def model_func(sp, *args):
    return sp.predict_one_spectrum(np.array(args))


def nneval(xs, w, b, alpha, nlayer):
    if nlayer == 2:
        w0, w1, w2 = w
        b0, b1, b2 = b
        l0 = leaky_relu(np.matmul(w0, xs) + b0, alpha)
        l1 = leaky_relu(np.matmul(w1, l0) + b1, alpha)
        return np.matmul(w2, l1) + b2
    elif nlayer == 3:
        w0, w1, w2, w3 = w
        b0, b1, b2, b3 = b
        l0 = leaky_relu(np.matmul(w0, xs) + b0, alpha)
        l1 = leaky_relu(np.matmul(w1, l0) + b1, alpha)
        l2 = leaky_relu(np.matmul(w2, l1) + b2, alpha)
        return np.matmul(w3, l2) + b3
    else:
        raise ValueError("Invalid nlayer={}".format(nlayer))


def cost(x, sp, flux_obs, flux_err=None, pw=2):
    flux_mod = sp.predict_one_spectrum(x)
    if flux_err is None:
        return .5 * np.sum(np.abs(flux_mod-flux_obs)**pw)
    else:
        return .5 * np.sum((np.abs(flux_mod-flux_obs)/flux_err)**pw)

            
# deprecated
# def train_one_pixel(x, y, sw, nhidden=(200, 200, 200), activation="leakyrelu", alpha=.01,  # NN parameters
#                     test_size=0.2, random_state=0, epochs=1000, batch_size=256,  # training parameters
#                     optimizer="adam", loss="mae", metrics=['mse', ],
#                     patience_earlystopping=5, patience_reducelronplateau=3, factor_reducelronplateau=0.5,
#                     filepath="",):
#     # initialize NN regressor
#     model = NN(kind="slam", ninput=x.shape[1], nhidden=nhidden, noutput=1, activation=activation, alpha=alpha)
#     # model.summary()
#     # set callbacks
#     model.set_callbacks(monitor_earlystopping="val_loss",
#                         patience_earlystopping=patience_earlystopping,
#                         monitor_modelcheckpoint="val_loss",
#                         filepath=filepath,
#                         monitor_reducelronplateau="val_loss",
#                         patience_reducelronplateau=patience_reducelronplateau,
#                         factor_reducelronplateau=factor_reducelronplateau)
#     # train pixels
#     model.train(x, y, batch_size=batch_size, sw=sw,
#                 test_size=test_size, optimizer=optimizer, epochs=epochs,
#                 loss=loss, metrics=metrics, random_state=random_state)
#     # ypred = model.predict(x).flatten()
#     if filepath not in ["", None]:
#         model = tf.keras.models.load_model(filepath)
#     return model.model.get_weights()

