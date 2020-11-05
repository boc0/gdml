from functools import partial

import pandas as pd

import jax.numpy as np
import jax.ops
from jax import grad, jit, vmap, jacfwd, jacrev

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

from sklearn.model_selection import GridSearchCV

from utils import KRR

data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
X = np.array(data['R'])
y = np.array(data['D'])

def fill_diagonal(a, value):
    return jax.ops.index_update(a, np.diag_indices(a.shape[0]), value)

@jit
def descriptor(x):
    distances = np.sum((x[:, None] - x[None, :])**2, axis=-1)
    distances = fill_diagonal(distances, 1) # because sqrt fails to compute gradient if called on 0s
    distances = np.sqrt(distances)
    D = 1 / distances
    D = np.tril(D)
    D = fill_diagonal(D, 0)
    return D.flatten()

@jit
def gaussian(x, x_, sigma=1):
    d, d_ = descriptor(x), descriptor(x_)
    sq_distance = np.sum((d - d_)**2)
    return np.exp(-sq_distance / sigma)

@jit
def hess_ij(H):
    return np.sum(H, axis=(0, 2))

def hessian(f):
    return jacfwd(jacrev(f))

@jit
def kernel(x, x_, sigma=1):
    _gaussian = partial(gaussian, x, sigma=sigma)
    hess = hessian(_gaussian)
    return hess_ij(hess(x_))

@jit
def kernel_matrix(X, sigma=1):
    def kernel(x, x_):
        _gaussian = partial(gaussian, x, sigma=sigma)
        hess = hessian(_gaussian)
        return hess_ij(hess(x_))

    @vmap
    def _kernels(x, sigma=sigma):
        _kernel = vmap(partial(kernel, x))
        return _kernel(X)

    K = _kernels(X)
    blocks = [list(x) for x in K]
    return np.block(blocks)


class VectorValuedKRR(KRR):

    def fit(self, X, y):
        self.X = X
        samples = X.shape[0]
        K = kernel_matrix(X, sigma=self.sigma)
        y = y.reshape(samples * 3)
        K = fill_diagonal(K, K.diagonal() + self.lamb)
        alphas = np.linalg.solve(K, y)
        self.alphas = alphas.reshape(samples, 3)

    def predict(self, x):
        @jit
        @vmap
        def predict(x):
            mu = np.zeros(3)
            for i in range(samples):
                mu += kernel(x, X[i], sigma=sigma) @ alphas[i]
            return mu
        results = predict(x)
        return np.array(results)

    def score(self, x, y):
        yhat = self.predict(x)
        return -np.mean(np.sum(np.abs(y - yhat), axis=1))


sigma_choices = list(np.linspace(0.25, 3, 12))
parameters = {'sigma': sigma_choices}
data_subset_sizes = np.linspace(10, 100, 10, dtype=int)
test = slice(20000, 20100)
errors = []

for size in data_subset_sizes:
    print(f'{size=}')

    cross_validation = GridSearchCV(VectorValuedKRR(), parameters)
    cross_validation.fit(X[:size], y[:size])
    results = cross_validation.cv_results_
    best = np.argmin(results['rank_test_score'])
    best_sigma = results['param_sigma'][best]
    print(f'{best_sigma=}')
    best_model = VectorValuedKRR(sigma=best_sigma)
    best_model.fit(X[:size], y[:size])
    best_test_error = -best_model.score(X[test], y[test]).item()
    best_model.save()
    print(f'{best_test_error=}')
    errors.append(best_test_error)


data = pd.DataFrame({'samples trained on': data_subset_sizes, 'mean absolute error': errors})
sns.pointplot(x='samples trained on', y='mean absolute error', data=data, s=100)
plt.savefig('learning_curve.png')
