from sklearn.base import BaseEstimator
import numpy as np

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
