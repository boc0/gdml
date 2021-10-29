from functools import partial
from time import time
from math import factorial

import pandas as pd
import numpy as onp

from jax import vmap, jacfwd, jacrev, jit
import jax.numpy as np
from jax.numpy import sqrt, exp
from jax.numpy.linalg import norm
from jax.ops import index_update, index
from scipy.stats import loguniform

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn

from utils import KRR, matern, binom, safe_sqrt, fill_diagonal, coulomb, gaussian, get_data


def kernel_big(x, x_, sigma=1.0, n=2, descriptor=coulomb):
    v = n + 1/2
    N, D = x.shape

    Dx, Dx_ = descriptor(x), descriptor(x_)
    diff = x - x_
    d = safe_sqrt(np.sum((Dx - Dx_)**2)) + 1e-1
    d_scaled = np.sqrt(2 * v) * d / sigma
    B = np.exp(- d_scaled)
    Pn = sum([factorial(n + k) / factorial(2*n) * binom(n, k) * (2 * d_scaled)**(n-k) for k in range(n)])

    eye = np.eye(N)
    def kron(i, j):
        return eye[i, j]

    def delta_p(q, i):
        return sum([factorial(n + k) / factorial(2*n) * binom(n, k) * \
                    (n - k) * (x[i, q] - x_[i, q]) / d**2 * (2**(sqrt(2)) * sqrt(v) * d / sigma)**(n-k)
                    for k in range(n)])

    def delta_b(q, i):
        return sqrt(2*v) * (x[i, q] - x_[i, q]) / (sigma * d) * exp(-d_scaled)

    def delta2p(q, i, p, j):
        return sum([factorial(n + k) / factorial(2*n) * binom(n, k) * \
                    (n - k - 2) * (n - k) * diff[i, q] * diff[j, p] / d**4 * (2**(sqrt(2)) * sqrt(v) * d / sigma)**(n-k) \
                    + kron(i, j) * factorial(n + k) / factorial(2*n) * binom(n, k) * (n - k) / d**2 * (2 * d_scaled)**(n-k)
                    for k in range(n)])

    def delta2b(q, i, p, j):
        return sqrt(2*v) * diff[i, q] * diff[j, p] * (sqrt(2*v) * d + sigma) / (sigma**2 * d**3) * exp(-d_scaled) \
               + kron(i, j) * sqrt(2*v) / (sigma*d) * exp(-d_scaled)

    def hess(q, p, i, j):
        return B * delta2p(q, i, p, j) + \
               delta_b(q, i) * delta_p(p, j) + \
               delta_b(p, j) * delta_p(q, i) + \
               Pn * delta2b(q, i, p, j)

    rangeD = np.arange(D)
    rangeN = np.arange(N)

    def kernel_pq(q, p):
        _hess = partial(hess, q, p)
        sums = vmap(vmap(_hess, (0, None)), (None, 0))(rangeN, rangeN)
        return sums

    K = vmap(vmap(kernel_pq, (0, None)), (None, 0))(rangeD, rangeD)
    K = (K + K.T) / 2
    K = K.reshape(3 * N, 3 * N)
    return K


class ForceKRR(VectorValuedKRR):
    kernel = kernel_big

    def fit(self, X, y):
        self.X = X
        self.y = y
        samples = X.shape[0]

        K = kernel_matrix(X, sigma=self.sigma, kernel=self.__class__.kernel)
        K = fill_diagonal(K, K.diagonal() + self.lamb)
        y = (y - self.means) / self.stdevs
        y = y.reshape(samples * 3 * self.n_atoms)
        alphas = np.linalg.solve(K, y)
        self.alphas = alphas.reshape(samples, 3 * self.n_atoms)

    def predict(self, x):
        def contribution(i, x):
            return self.__class__.kernel(x, self.X[i], sigma=self.sigma) @ self.alphas[i]
        @vmap
        def predict(x):
            indices = np.arange(self.samples)
            _contribution = vmap(partial(contribution, x=x))
            contributions = _contribution(indices)
            mu = np.sum(contributions, axis=0)
            return mu
        results = predict(x)
        res = np.array(results) * self.stdevs + self.means # "de-normalize"
        res = res.reshape(self.n_samples, self.n_atoms)
        return res

    def score(self, x, y, angle=False):
        yhat = self.predict(x)
        error = mean_squared_error(yhat, self.y.reshape(), squared=True)
        if not angle:
            return -onp.mean(error)
        angle = [np.degrees(np.arccos(np.clip(unit(yhat[i]) @ unit(y[i]), -1.0, 1.0)))
                 for i in range(y.shape[0])]
        angle = np.array(angle)
        return np.mean(error), np.mean(angle)
