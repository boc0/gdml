import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model._ridge import _solve_cholesky_kernel

def descriptor(x):
    atoms, dimension = x.shape
    D = np.zeros((atoms, atoms))
    for i in range(atoms):
        for j in range(atoms):
            if i > j:
                D[i, j] = 1 / np.linalg.norm(x[i] - x[j])
            else:
                D[i, j] = 0
    return D.flatten()


def gaussian(data, sigma=1):
    X = [descriptor(x) for x in data]
    dists = squareform(pdist(X, 'euclidean'))
    return np.exp(-dists ** 2 / sigma)

def kstar(input, data, sigma=1):
    x = np.array([descriptor(x_) for x_ in input])
    X = np.array([descriptor(x_) for x_ in data])
    dists = cdist(x, X)
    kstar = np.exp(-dists ** 2 / sigma)
    return kstar


class GaussianKRR(KernelRidge):
    def __init__(self, sigma=1):
        lamb=1e-15
        self.alpha = lamb
        self.sigma = sigma
        self.kernel = None
    def _validate_data(self, X, y, **_):
        return X, y
    def _get_kernel(self, X, Y=None):
        if Y is not None:
            return kstar(X, Y, sigma=self.sigma)
        else:
            return gaussian(X, sigma=self.sigma)

    def fit(self, X, y):
        K = self._get_kernel(X)
        alpha = np.atleast_1d(self.alpha)

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        n_samples = K.shape[0]
        K.flat[::n_samples + 1] += alpha[0]

        self.dual_coef_ = np.linalg.solve(K, y)
        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        self.X_fit_ = X

        return self


if __name__ == '__main__':
    data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
    X = data['R'][:1000]
    y = data['E'][:1000]
    K = gaussian(X)

    model = GaussianKRR(sigma=1)
    model.fit(X, y)
    yreal = y[1]
    yhat = model.predict(X[0:1])
    yhat, yreal
