from sklearn.base import BaseEstimator
import jax.ops
import jax.numpy as np
from jax import jit

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation


def heatmap_animation(generator, name='what', **heatmap_kwargs):
    kwargs = {'xticklabels': False, 'yticklabels': False, **heatmap_kwargs}

    def animate(i):
        plt.clf()
        data = generator(i)
        sns.heatmap(data, **kwargs)

    anim = animation.FuncAnimation(plt.figure(), animate, frames=20, repeat=False)
    anim.save(f'images/{name}.gif')


class KRR(BaseEstimator):
    def __init__(self, sigma=1, lamb=1e-5):
        self.lamb = lamb
        self.sigma = sigma

    @property
    def samples(self):
        return self.X.shape[0]

    def save(self):
        np.savez(f'models/{self.samples}', X=self.X, alphas=self.alphas)

    def load(self, name):
        data = np.load(f'models/{name}.npz')
        self.X = data['X']
        self.alphas = data['alphas']


def fill_diagonal(a, value):
    return jax.ops.index_update(a, np.diag_indices(a.shape[0]), value)

@jit
def descriptor(x):
    distances = np.sum((x[:, None, :] - x[None, :, :])**2, axis=-1)
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
    return 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-sq_distance / sigma**2)

def matrix_heatmap(matrix, **kwargs):
    ax = sns.heatmap(matrix, xticklabels=False, yticklabels=False, **kwargs)
    plt.show()
