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

from utils import KRR, matern, binom, safe_sqrt, fill_diagonal, coulomb, gaussian


def kernel_matern(x, x_, sigma=1.0, n=2, descriptor=coulomb):
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


def kernel(x, x_, sigma=1.0, similarity=matern):
    _kernel = partial(similarity, x_, sigma=sigma)
    hess = hessian(_kernel)
    H = hess(x_)
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



PARAMETERS = {'sigma': loguniform(10**1, 10**4), 'lamb': loguniform(10**-2, 10**3)}
# sigmas = list(np.logspace(1, 4, 19))
# lambdas = list(np.logspace(-2, 3, 21))
# PARAMETERS = {'sigma': sigmas, 'lamb': lambdas}


def train(Xtrain, ytrain, Xdev, ydev, Xtest, ytest,
          cv=RandomizedSearchCV(VectorValuedKRR(), PARAMETERS, n_iter=3),
          return_results=False,
          n_best=10):

    start = time()
    size = Xtrain.shape[0]
    with mlflow.start_run(nested=True) as run:
        mlflow.log_param('n_samples', size)
        print(f'\nsize: {size}')
        cv.fit(Xtrain, ytrain)
        results = cv.cv_results_
        indices = onp.argpartition(results['rank_test_score'], n_best)[:n_best]
        # print('errors:')

        def test(params):
            model = VectorValuedKRR(**params)
            model.fit(Xtrain, ytrain)
            error, angle = (result.item() for result in model.score(Xdev, ydev, angle=True))
            # print(f'{str(params).ljust(60)} {error:.4f} {angle:.2f}')
            return model, error, angle

        models, errors, angles = zip(*map(test, onp.array(results['params'])[indices]))
        _best = np.argmin(np.array(errors))
        best = indices[_best]
        best_params = results['params'][best]
        print(f'best params: {best_params}')
        mlflow.log_params(best_params)
        error, angle = errors[_best], angles[_best]
        print(f'dev error: {error}')
        print(f'mean angle: {angle}')
        mlflow.log_metric('dev error', error)
        mlflow.log_metric('dev angle', angle)
        best_model = VectorValuedKRR(**best_params)
        best_model.fit(Xtrain, ytrain)
        error, angle = (result.item() for result in best_model.score(Xtest, ytest, angle=True))
        print(f'test error: {error}')
        print(f'mean angle: {angle}')
        mlflow.log_metric('test error', error)
        mlflow.log_metric('test angle', angle)
        # mlflow.sklearn.save_model(models[_best], f'mlruns/0/{run.info.run_id}/best_model')
        mlflow.log_metric('time', time() - start)
    print(f'time taken: {time() - start}')
    if return_results:
        return error, angle, results
    return error, angle



if __name__ == '__main__':
    mlflow.sklearn.autolog()

    data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
    X = np.array(data['R'])
    y = np.array(data['D'])
    M = X.shape[0]

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
