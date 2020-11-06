from sklearn.base import BaseEstimator
import numpy as np
import jax.ops


class KRR(BaseEstimator):
    def __init__(self, sigma=1):
        self.lamb = 1e-15
        self.sigma = sigma

    @property
    def samples(self):
        return self.X.shape[0]

    def save(self):
        np.savez_compressed(f'models/{self.samples}', X=self.X, alphas=self.alphas)

    def load(self, name):
        data = np.load(f'models/{name}.npz')
        self.X = data['X']
        self.alphas = data['alphas']


def fill_diagonal(a, value):
    return jax.ops.index_update(a, np.diag_indices(a.shape[0]), value)


def descriptor(x):
    distances = np.sum((x[:, None] - x[None, :])**2, axis=-1)
    distances = fill_diagonal(distances, 1) # because sqrt fails to compute gradient if called on 0s
    distances = np.sqrt(distances)
    D = 1 / distances
    D = np.tril(D)
    D = fill_diagonal(D, 0)
    return D.flatten()


def gaussian(x, x_, sigma=1):
    d, d_ = descriptor(x), descriptor(x_)
    sq_distance = np.sum((d - d_)**2)
    return np.exp(-sq_distance / sigma)
