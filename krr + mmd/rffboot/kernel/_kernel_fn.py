import functools

import numpy as np

from scipy.stats import cauchy, laplace
from scipy.spatial.distance import cdist

from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.metrics.pairwise import laplacian_kernel


def get_rbf(sigma=0.1):
    return functools.partial(rbf, gamma=sigma ** 2 / 2), functools.partial(np.random.normal, 0, sigma)


def _cauchy_kernel(X, Y=None, gamma=1.0):
    Y = X if Y is None else Y

    def _dist(x, y):
        return 1 / np.product(1 + gamma * np.square(x - y))

    return cdist(X, Y, _dist)


def _gen_laplace(sigma, size):
    n, d = size
    rvs = laplace.rvs(loc=0.0, scale=sigma, size=n * d)
    return np.reshape(rvs, (n, d))


def get_cauchy(sigma):
    return functools.partial(_cauchy_kernel, gamma=sigma ** 2), functools.partial(_gen_laplace, sigma=sigma)


def _gen_cauchy(sigma, size):
    n, d = size
    rvs = cauchy.rvs(loc=0.0, scale=sigma, size=n * d)
    return np.reshape(rvs, (n, d))


def get_laplace(sigma=1.0):
    return functools.partial(laplacian_kernel, gamma=sigma), functools.partial(_gen_cauchy, sigma=sigma)
