from functools import partial
from time import time
from math import factorial

import pandas as pd
import numpy as onp

from jax import vmap, jacfwd, jacrev
import jax.numpy as np
from jax.numpy import sqrt, exp

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
# from sklearn.utils.fixes import loguniform

import mlflow

from utils import KRR, matern, binom, safe_sqrt, fill_diagonal, coulomb


def hessian(f):
    return jacfwd(jacrev(f))


def symmetric(matrix):
    return np.allclose(matrix, matrix.T)


def kernel_matern_explicit(x, x_, sigma=1.0, n=2):
    v = n + 1/2
    N, D = x.shape

    Dx, Dx_ = coulomb(x), coulomb(x_)
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

    '''
    K = np.zeros((D, D))
    for p in range(D):
        for q in range(D):
            for i in range(N):
                for j in range(N):
                    K = index_update(K, index[p, q], K[p, q] + hess(q, p, i, j))
                    # K[p, q] += hess(p, q, i, j)
                    # K[i, j] += hess(q, p, i, j)
    # K = np.array(K)
    # K = (K + K.T) / 2
    return K
    '''

    rangeD = np.arange(D)
    rangeN = np.arange(N)

    def kernel_pq(q, p):
        _hess = partial(hess, q, p)
        sums = vmap(vmap(_hess, (0, None)), (None, 0))(rangeN, rangeN)
        return np.sum(sums)

    K = vmap(vmap(kernel_pq, (0, None)), (None, 0))(rangeD, rangeD)
    # K = (K + K.T) / 2
    return K



def kernel(x, x_, sigma=1.0, similarity=matern):
    _kernel = partial(similarity, x, sigma=sigma)
    hess = hessian(_kernel)
    H = hess(x_)
    K = np.sum(H, axis=(0, 2))
    K = (K + K.T) / 2
    return K



def kernel_matrix(X, sigma=1.0, kernel=kernel):
    '''
    if similarity is None:
        _kernel = partial(kernel, sigma=sigma)
    else:
        _kernel = partial(kernel, sigma=sigma, similarity=similarity)   # _kernel: kernel ((x, x') -> K^(3x3)) parametrized with sigma
    '''
    _kernel = partial(kernel, sigma=sigma)
    @vmap
    def _kernels(x):
        vec_kernel = vmap(partial(_kernel, x))          # vec_kernel: (x_1, x_2, ..., x_M) -> (k(x, x_1), k(x, x_2), ..., k(x, x_M))
        return vec_kernel(X)
    K = _kernels(X)                                     # K: list of lists of kernels for each 2 data points
    blocks = [list(x) for x in K]
    return np.block(blocks)                             # from list of lists of K(3x3)-matrices to block matrix of K(3x3) matrices



def unit(vector):
    return vector / np.linalg.norm(vector)


class VectorValuedKRR(KRR):
    def __init__(self, lamb=1e-5, sigma=1.0):
        super().__init__(lamb=lamb, sigma=sigma)
        # self.similarity = similarity
        # self.kernel = jit(partial(kernel, similarity=similarity))
        self.kernel = kernel_matern_explicit

    def fit(self, X, y):
        self.X = X
        samples = X.shape[0]

        K = kernel_matrix(X, sigma=self.sigma, kernel=self.kernel)
        K = fill_diagonal(K, K.diagonal() + self.lamb)
        self.means = np.mean(y)
        self.stdevs = np.std(y)
        y = (y - self.means) / self.stdevs
        y = y.reshape(samples * 3)
        alphas = np.linalg.solve(K, y)
        self.alphas = alphas.reshape(samples, 3)

    def predict(self, x):
        def contribution(i, x):
            return self.kernel(x, self.X[i], sigma=self.sigma) @ self.alphas[i]
        @vmap
        def predict(x):
            indices = np.arange(self.samples)
            _contribution = vmap(partial(contribution, x=x))
            contributions = _contribution(indices)
            # contributions = [contribution(i, x) for i in indices]
            mu = np.sum(contributions, axis=0)
            return mu
        results = predict(x)
        return np.array(results) * self.stdevs + self.means # "de-normalize"

    def score(self, x, y, angle=False):
        yhat = self.predict(x)
        # error = np.linalg.norm(y - yhat, axis=1)**2
        error = mean_squared_error(yhat, y, squared=False)
        if not angle:
            return -np.mean(error)
        angle = []
        for i in range(y.shape[0]):
            angle.append( np.degrees(np.arccos(np.clip(unit(yhat[i]) @ unit(y[i]), -1.0, 1.0))) )
        angle = np.array(angle)
        return np.mean(error), np.mean(angle)


# PARAMETERS = {'sigma': loguniform(10**1, 10**4), 'lamb': loguniform(10**-2, 10**3)}
sigmas = list(np.logspace(1, 4, 19))
lambdas = list(np.logspace(-2, 3, 21))
PARAMETERS = {'sigma': sigmas, 'lamb': lambdas}

def train(Xtrain, ytrain, Xtest, ytest,
          parameters=PARAMETERS,
          cv=GridSearchCV(VectorValuedKRR(), PARAMETERS),
          return_results=False):
          # cv=RandomizedSearchCV(VectorValuedKRR(), PARAMETERS, n_iter=50)):

    start = time()
    size = Xtrain.shape[0]
    with mlflow.start_run(nested=True) as run:
        mlflow.log_param('n_samples', size)
        cv.fit(Xtrain, ytrain)
        results = cv.cv_results_
        best = np.argmin(results['rank_test_score'])
        best_params = results['params'][best]
        print(f'best params: {best_params}')
        best_model = VectorValuedKRR(**best_params)
        best_model.fit(Xtrain, ytrain)
        best_test_error, angle = (result.item() for result in best_model.score(Xtest, ytest, angle=True))
        print(f'test error: {best_test_error}')
        print(f'mean angle: {angle}')
        mlflow.log_metric('test error', best_test_error)
        mlflow.log_metric('test angle', angle)
    print(f'time taken: {time() - start}')
    if return_results:
        return best_test_error, angle, results
    return best_test_error, angle


if __name__ == '__main__':
    mlflow.sklearn.autolog()

    data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
    X = np.array(data['R'])
    y = np.array(data['D'])
    M = X.shape[0]

    # sigmas = list(np.logspace(1, 4, 19))
    # lambdas = list(np.logspace(-2, 3, 16))
    # parameters = {'sigma': sigmas, 'lamb': lambdas}
    # parameters = {'sigma': loguniform(10**1, 10**4), 'lamb': loguniform(10**-2, 10**3)}
    data_subset_sizes = np.linspace(10, 100, 10, dtype=int)
    test_indices = onp.random.choice(M, size=100, replace=False)
    Xtest, ytest = X[test_indices], y[test_indices]
    mask = onp.ones(M, dtype=bool)
    mask[test_indices] = False
    X, y = X[mask], y[mask]

    train_indices = onp.random.choice(M-100, size=data_subset_sizes[-1], replace=False)
    X, y = X[train_indices], y[train_indices]

    errors, angles = [], []

    with mlflow.start_run():
        for size in data_subset_sizes:
            Xtrain, ytrain = X[:size], y[:size]
            error, angle = train(Xtrain, ytrain, Xtest, ytest)
            errors.append(error)
            angles.append(angle)

        data = pd.DataFrame({'samples trained on': data_subset_sizes, 'mean squared error norm': errors, 'mean angle': angles})
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        sns.pointplot(x='samples trained on', y='mean squared error norm', data=data, s=100, ax=ax, color='royalblue')
        sns.pointplot(x='samples trained on', y='mean angle', data=data, s=100, ax=ax2, color='coral')
        plt.savefig(f'learning_curve.png')
        mlflow.log_figure(fig, 'learning_curve.png')


# old_results = results.copy()
