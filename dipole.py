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

OFFSET = 1e-6

def kernel_matern(x, x_, sigma=1.0, n=2, descriptor=coulomb, offset=OFFSET):
    v = n + 1/2
    N, D = x.shape

    Dx, Dx_ = descriptor(x), descriptor(x_)
    diff = x - x_
    d = safe_sqrt(np.sum((Dx - Dx_)**2)) + offset # we add this to avoid a math error during scoring
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
        return np.sum(sums)

    K = vmap(vmap(kernel_pq, (0, None)), (None, 0))(rangeD, rangeD)
    # K = (K + K.T) / 2
    return K


def kernel_gauss(x, x_, sigma=1.0, descriptor=coulomb):
    N, D = x.shape

    Dx, Dx_ = descriptor(x), descriptor(x_)
    D_difference = Dx - Dx_
    k_x = gaussian(x, x_)

    eye = np.eye(N)
    def kronecker(i, j):
        return eye[i, j]

    def kronecker_sign_factor(k):
        delta = np.zeros((N, N))
        delta = index_update(delta, index[k, :], 1)
        delta = index_update(delta, index[:, k], 1)
        delta = index_update(delta, index[k, k], 0)
        return delta.flatten()

    def difference_at(q):
        return (x[:, None] - x[None, :])[:, :, q].flatten()

    def gamma(q, k):
        g = -Dx**3 * difference_at(q) * kronecker_sign_factor(k)
        return g.flatten()

    def delta_gamma(q, k, p, l):
        dg = (-1 + 2 * kronecker(k, l)) * Dx**3 * (kronecker(p, q) - 3 * difference_at(q) * difference_at(p) * Dx**2)
        return dg.flatten()

    def phi(q, k):
        return sigma**(-2) * ( (D_difference) @ gamma(q, k) )

    def delta_phi(q, k, p, l):
        return sigma**(-2) * ( D_difference @ delta_gamma(q, k, p, l) + gamma(q, k) @ gamma(p, l) )

    def hess(q, p, k, l):
        return -k_x * ( phi(q, k) * phi(p, l) - delta_phi(q, k, p, l) )

    rangeD = np.arange(D)
    rangeN = np.arange(N)

    def kernel_pq(q, p):
        _hess = lambda k, l: -k_x * ( phi(q, k) * phi(p, l) - delta_phi(q, k, p, l) )
        _hess = partial(hess, q, p)
        sums = vmap(vmap(_hess, (0, None)), (None, 0))(rangeN, rangeN)
        return np.sum(sums)

    K = vmap(vmap(kernel_pq, (0, None)), (None, 0))(rangeD, rangeD)
    K = (K + K.T) / 2
    return K


def hessian(f):
    return jacfwd(jacrev(f))


def kernel(x, x_, sigma=1.0, similarity=matern, descriptor=coulomb):
    if 'sigma' in similarity.__code__.co_varnames:
        similarity = partial(similarity, sigma=sigma)
    similarity = partial(similarity, x_, descriptor=descriptor)
    hess = hessian(similarity)
    H = hess(x)
    K = np.sum(H, axis=(0, 2))
    K = (K + K.T) / 2
    return K




def kernel_matrix(data, sigma=1.0, kernel=kernel_matern, descriptor=coulomb):
    if descriptor is None:
        _kernel = partial(kernel, sigma=sigma)
    else:
        _kernel = partial(kernel, sigma=sigma, descriptor=descriptor)   # _kernel: kernel ((x, x') -> K^(3x3)) parametrized with sigma
    @vmap
    def _kernels(x):
        vec_kernel = vmap(partial(_kernel, x_=x))          # vec_kernel: (x_1, x_2, ..., x_M) -> (k(x, x_1), k(x, x_2), ..., k(x, x_M))
        return vec_kernel(data)
    K = _kernels(data)                                     # K: list of lists of kernels for each 2 data points
    blocks = [list(x) for x in K]
    return np.block(blocks)                             # from list of lists of K(3x3)-matrices to block matrix of K(3x3) matrices


def unit(vector):
    return vector / np.linalg.norm(vector)


class VectorValuedKRR(KRR):
    kernel = kernel_matern
    descriptor = coulomb

    @property
    def stdevs(self):
        return np.std(self.y)

    @property
    def means(self):
        return np.mean(self.y)

    @property
    def n_samples(self):
        return self.X.shape[0]

    @property
    def n_atoms(self):
        return self.X.shape[1]

    def fit(self, X, y):
        self.X = X
        self.y = y
        samples = X.shape[0]

        K = kernel_matrix(X, sigma=self.sigma, kernel=self.__class__.kernel)
        K = fill_diagonal(K, K.diagonal() + self.lamb)
        y = (y - self.means) / self.stdevs
        y = y.reshape(samples * 3)
        alphas = np.linalg.solve(K, y)
        self.alphas = alphas.reshape(samples, 3)

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
        return np.array(results) * self.stdevs + self.means # "de-normalize"

    def score(self, x, y, angle=False):
        yhat = self.predict(x)
        # error = np.linalg.norm(y - yhat, axis=1)**2
        error = mean_squared_error(yhat, y, squared=True)
        if not angle:
            return -onp.mean(error)
        angle = [np.degrees(np.arccos(np.clip(unit(yhat[i]) @ unit(y[i]), -1.0, 1.0)))
                 for i in range(y.shape[0])]
        angle = np.array(angle)
        return np.mean(error), np.mean(angle)

    def errors(self, X, y):
        yhat = self.predict(X)
        angles = np.array([np.degrees(np.arccos(np.clip(unit(yhat[i]) @ unit(y[i]), -1.0, 1.0)))
                           for i in range(y.shape[0])])
        magnitudes = norm(yhat, axis=1) - norm(y, axis=1)
        return angles, magnitudes


class KRRGauss(VectorValuedKRR):
    kernel = kernel_gauss
