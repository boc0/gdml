from functools import partial

import pandas as pd
import numpy as onp

from jax import jit, vmap, jacfwd, jacrev
import jax.numpy as np
from jax.numpy import sqrt, exp
from jax.ops import index, index_update
import jax

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from utils import KRR, gaussian, matern, binom, safe_sqrt, fill_diagonal, coulomb


def hessian(f):
    return jacfwd(jacrev(f))

def hess_at(H, i, j):
    return H[i, :, j, :]

def symmetric(H):
    sym = np.allclose(H, H.T)
    return sym

from math import factorial


def kernel_matern_explicit(x, x_, sigma=1.0, n=2):
    v = n + 1/2
    N, D = x.shape

    Dx, Dx_ = coulomb(x), coulomb(x_)
    diff = x - x_
    d = safe_sqrt(np.sum((Dx - Dx_)**2))
    d_scaled = np.sqrt(2 * v) * d / sigma
    B = np.exp(- d_scaled)
    Pn = sum([factorial(n + k) / factorial(2*n) * binom(n, k) * (2 * d_scaled)**(n-k) for k in range(n)])

    eye = np.eye(N)
    def kron(i, j):
        return eye[i, j]

    def delta_p(q, i):
        return sum([factorial(n + k) / factorial(2*n) * binom(n, k) * \
                    (n - k) * (x[q, i] - x_[q, i]) / d**2 * (2**(sqrt(2)) * sqrt(v) * d / sigma)**(n-k)
                    for k in range(n)])

    def delta_b(q, i):
        return sqrt(2*v) * (x[q, i] - x_[q, i]) / (sigma * d) * exp(-d_scaled)

    def delta2p(q, i, p, j):
        return sum([factorial(n + k) / factorial(2*n) * binom(n, k) * \
                    (n - k - 2) * (n - k) * diff[q, i] * diff[p, j] / d**4 * (2**(sqrt(2)) * sqrt(v) * d / sigma)**(n-k) \
                    + kron(i, j) * factorial(n + k) / factorial(2*n) * binom(n, k) * (n - k) / d**2 * (2 * d_scaled)**(n-k)
                    for k in range(n)])

    def delta2b(q, i, p, j):
        return sqrt(2*v) * diff[q, i] * diff[p, j] * (sqrt(2*v) * d + sigma) / (sigma**2 * d**3) * exp(-d_scaled) \
               + kron(i, j) * sqrt(2*v) / (sigma*d) * exp(-d_scaled)

    def hess(q, p, i, j):
        return B * delta2p(q, i, p, j) + \
               delta_b(q, i) * delta_p(p, j) + \
               delta_b(p, j) * delta_p(q, i) + \
               Pn * delta2b(q, i, p, j)


    K = np.zeros((D, D))
    for p in range(D):
        for q in range(D):
            for i in range(N):
                for j in range(N):
                    K = index_update(K, index[p, q], K[p, q] + hess(q, p, i, j))
                    # K[i, j] += hess(q, p, i, j)
    # K = np.array(K)
    K = (K + K.T) / 2
    return K

    rangeD = np.arange(D)
    rangeN = np.arange(N)

    def kernel_pq(q, p):
        _hess = partial(hess, q, p)
        sums = vmap(vmap(_hess, (0, None)), (None, 0))(rangeN, rangeN)
        return np.sum(sums)

    K = vmap(vmap(kernel_pq, (0, None)), (None, 0))(rangeD, rangeD)
    K = (K + K.T) / 2
    return K

@jit
def kernel_gauss_explicit(x, x_, sigma=1.0):
    N, D = x.shape

    Dx, Dx_ = coulomb(x), coulomb(x_)
    D_difference = Dx - Dx_
    k_x = gaussian(x, x_)

    eye = np.eye(N)
    def kronecker(i, j):
        return eye[i, j]

    def kronecker_sign_factor(k):
        delta = np.zeros((N, N))
        delta = jax.ops.index_update(delta, jax.ops.index[k, :], 1)
        delta = jax.ops.index_update(delta, jax.ops.index[:, k], 1)
        delta = jax.ops.index_update(delta, jax.ops.index[k, k], 0)
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


def kernel(x, x_, sigma=1.0, similarity=matern):
    _kernel = partial(similarity, x, sigma=sigma)
    hess = hessian(_kernel)
    H = hess(x_)
    K = np.zeros((3,3))
    for i in range(4):
        for j in range(4):
            K += H[i, :, j, :]
    K = (K + K.T) / 2
    return K


def kernel_matrix(X, sigma=1.0, similarity=matern):
    _kernel = partial(kernel, sigma=sigma, similarity=similarity)   # _kernel: kernel ((x, x') -> K^(3x3)) parametrized with sigma
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
    def __init__(self, similarity=matern, lamb=1e-5, sigma=1.0):
        super().__init__(lamb=lamb, sigma=sigma)
        self.similarity = similarity
        self.kernel = jit(partial(kernel, similarity=similarity))

    def fit(self, X, y):
        self.X = X
        samples = X.shape[0]

        K = kernel_matrix(X, sigma=self.sigma, similarity=self.similarity)
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
            mu = np.sum(contributions, axis=0)
            return mu
        results = predict(x)
        return np.array(results) * self.stdevs + self.means # "de-normalize"

    def score(self, x, y, angle=False):
        yhat = self.predict(x)
        error = np.linalg.norm(y - yhat, axis=1)**2
        if not angle:
            return -np.mean(error)
        angle = []
        for i in range(y.shape[0]):
            angle.append( np.degrees(np.arccos(np.clip(unit(yhat[i]) @ unit(y[i]), -1.0, 1.0))) )
        angle = np.array(angle)
        return np.mean(error), np.mean(angle)


if __name__ == '__main__':
    data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
    X = np.array(data['R'])
    y = np.array(data['D'])

    sigma_choices = [1000.0, 5000.0]# list(np.linspace(1000, 6000, 6)) + [10000.0]
    lambda_choices = [3e-5, 1e-5]#, 3e-4, 1e-4, 3e-3, 1e-3]
    parameters = {'sigma': sigma_choices, 'lamb': lambda_choices}
    data_subset_sizes = np.linspace(5, 10, 2, dtype=int)
    test = slice(20000, 20100)
    Xtest, ytest = X[test], y[test]
    errors, angles = [], []
    M = X.shape[0]

    all_indices = onp.random.choice(M, size=data_subset_sizes[-1], replace=False)
    X, y = X[all_indices], y[all_indices]

    from time import time

    for size in data_subset_sizes:
        start = time()
        print(f'size: {size}')

        # indices = onp.random.choice(all_indices, size=size, replace=False)
        Xtrain, ytrain = X[:size], y[:size]

        cross_validation = GridSearchCV(VectorValuedKRR(), parameters)
        cross_validation.fit(Xtrain, ytrain)
        results = cross_validation.cv_results_
        best = np.argmin(results['rank_test_score'])
        best_params = results['params'][best]
        print(f'best params: {best_params}')
        best_model = VectorValuedKRR(**best_params)
        best_model.fit(Xtrain, ytrain)
        best_test_error, angle = (result.item() for result in best_model.score(Xtest, ytest, angle=True))
        errors.append(best_test_error)
        angles.append(angle)
        print(f'test error: {best_test_error}')
        print(f'mean angle: {angle}')

        best_model.save()

        taken = time() - start
        print(f'time taken: {taken}', end='\n\n')

    data = pd.DataFrame({'samples trained on': data_subset_sizes, 'mean squared error norm': errors, 'mean angle': angles})
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    sns.pointplot(x='samples trained on', y='mean squared error norm', data=data, s=100, ax=ax, color='royalblue')
    sns.pointplot(x='samples trained on', y='mean angle', data=data, s=100, ax=ax2, color='coral')
    plt.savefig(f'learning_curve.png')
