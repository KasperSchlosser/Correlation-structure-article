import numpy as np
import scipy.integrate as integrate
import properscoring as ps
import keras
import torch
from keras import ops


def variogram_weight(simulations, p=0.5, window=24, offset=24):

    n, m = simulations.shape
    n_windows = (n - offset) // window

    weights = np.zeros((n_windows, window, window))

    for k, start in enumerate(range(offset, n, window)):

        if start + window > n:
            break

        for i in range(0, window - 1):
            for j in range(i + 1, window):

                Ediff = np.abs(simulations[start + i, :] - simulations[start + j, :]) ** p

                weights[k, i, j] = 1 / np.var(Ediff, ddof=1)
    return weights


def variogram_score(simulations, actuals, p=0.5, window=24, offset=0, weight=None):

    # simualtions are n observations times m simulations
    n, m = simulations.shape

    score = 0
    variogram = np.zeros((window, window))
    n_windows = (n - offset) // window

    if weight is None:

        weight = np.ones((window, window))
        # weight = np.arange(window)
        # weight = np.abs(np.subtract.outer(weight, weight))
        # weight = 1/weight
        np.fill_diagonal(weight, 0)
        weight = weight / weight.sum()

    for start in range(offset, n, window):

        # just ignore trailing part
        if start + window > n:
            break

        for lag in range(1, window):

            Ediff = simulations[start : start + window - lag, :] - simulations[start + lag : start + window, :]
            Ediff = np.abs(Ediff) ** p
            Ediff = np.mean(Ediff, axis=1)

            Adiff = actuals[start : start + window - lag] - actuals[start + lag : start + window]
            Adiff = np.abs(Adiff) ** p

            scores = np.abs(Adiff - Ediff) ** 2

            variogram += np.diagflat(scores, lag)
            variogram += np.diagflat(scores, -lag)

    variogram /= n_windows
    score = ((weight * variogram).sum()) ** (1 / (2 * p))

    return (
        score,
        variogram,
    )


def variogram_distribution(simulations, p=0.5, window=24, offset=0, weight=None):

    # simualtions are n observations times m simulations
    n, m = simulations.shape

    e_variogram = np.zeros((window, window))
    std_variogram = np.zeros((window, window))

    for i in range(window - 1):
        for j in range(i + 1, window):

            obsj = simulations[j::window, :]
            obsi = simulations[i::window, :][: obsj.shape[0], :]

            diffs = np.abs(obsi - obsj) ** p

            e_variogram[i, j] = diffs.mean()
            std_variogram[i, j] = diffs.std(ddof=1)

            e_variogram[j, i] = e_variogram[i, j]
            std_variogram[j, i] = std_variogram[i, j]

    return e_variogram, std_variogram


def continous_ranked_probability_score(simulations, actuals):
    # n observation times k simulations
    return np.mean(ps.crps_ensemble(actuals, simulations))


def mean_average_error(actuals, predicted):
    return np.mean(np.abs(actuals - predicted))


def mean_squared_error(actuals, predicted):
    return np.mean((actuals - predicted) ** 2)


def root_mean_squared_error(actuals, predicted):
    return np.sqrt(mean_squared_error(actuals, predicted))


class Quantileloss(keras.losses.Loss):
    def __init__(self, quantiles, **kwargs):
        super().__init__(**kwargs)
        self.quantiles = torch.tensor(quantiles)
        self.neg_quantiles = self.quantiles - 1

    def call(self, y_true, y_pred):
        d = (y_true - y_pred.T).T
        x1 = self.quantiles * d
        x2 = self.neg_quantiles * d

        return ops.sum(ops.maximum(x1, x2), axis=-1)


def calc_scores(actuals, predicted, simulations, VARS_kwargs=None):

    # maybe return dict?

    if VARS_kwargs is None:
        VARS_kwargs = dict()

    MAE = mean_average_error(actuals, predicted)
    RMSE = root_mean_squared_error(actuals, predicted)
    VARS = variogram_score(simulations, actuals, **VARS_kwargs)[0]
    CRPS = continous_ranked_probability_score(simulations, actuals)

    return MAE, RMSE, CRPS, VARS


def continous_wasserstein(F_inv, G_inv, lims=(1e-8, 1 - 1e-8), order=1, **kwargs):
    # quick implementation of the wasserstein metric
    # F_inf, G_inv: quantile (inverse cdf) functions for the distributions F, G
    # lim: limits of integration, change if distribution has infinite support

    def _dist(x):
        return np.abs(F_inv(x) - G_inv(x)) ** order

    res = integrate.quad(_dist, *lims, **kwargs)

    return (res[0] ** (1 / order), *res[1:])


def continous_kl_divergence(f, g, lims=(-np.inf, np.inf), **kwargs):

    def _div(x):
        fx = f(x)
        gx = g(x)
        return fx * (np.log(fx) - np.log(gx))

    return integrate.quad(_div, *lims, **kwargs)
