from functools import partial
import numpy as np
from numdifftools.core import Hessian
from sklearn.base import BaseEstimator

from main import descriptor


def similarity(x, x_, sigma=1):
    x, x_ = descriptor(x.reshape(3,4)), descriptor(x_.reshape(3,4))
    dist = np.linalg.norm(x - x_)
    return -dist**2 / sigma



def kernel_matrix(X, sigma=1):
    samples, dimension = X.shape
    kernel_shape = (dimension, dimension)
    K = np.zeros((samples, samples) + kernel_shape) # M x M x 3N x 3N
    for i in range(samples):
        kernel = partial(similarity, X[i], sigma=sigma)
        hess = Hessian(kernel)
        for j in range(samples):
            K[i, j] = hess(X[j])
    K = [list(k) for k in K]
    K = np.block(K)
    return K

def kstar(x, X, sigma=1):
    samples, dimension = x.shape
    training_samples, dimension = X.shape
    kernel_shape = (dimension, dimension)
    K = np.zeros((samples, training_samples) + kernel_shape)
    for i in range(samples):
        kernel = partial(similarity, x[i], sigma=sigma)
        hess = Hessian(kernel)
        for j in range(training_samples):
            K[i, j] = hess(X[j])
    K = [list(k) for k in K]
    K = np.block(K)
    return K


class VectorValuedKRR(sklearn.base.BaseEstimator):
    def __init__(self, sigma=1):
        self.sigma = sigma
        self.lamb = 1e-15

    def fit(self, X, y):
        self.X = X
        K = kernel_matrix(X, sigma=self.sigma)
        samples, features = y.shape
        y = y.reshape(samples * features)
        K.flat[::10 + 1] += self.lamb
        self.alphas = np.linalg.solve(K, y)

    def predict(self, x):
        samples = x.shape[0]
        k = kstar(x, self.X)
        yhat = k @ self.alphas
        features = yhat.shape[0] // samples
        return yhat.reshape(samples, features)

    def score(self, X, y):
        yhat = self.predict(X)
        return -np.mean(np.abs(y - yhat))

data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
X = data['R'].reshape(50000, 12)
y = data['F'].reshape(50000, 12)

model = VectorValuedKRR(sigma=1)
model.fit(X, y)
model.score(X[:2], y[:2])

from sklearn.model_selection import GridSearchCV

sigma_choices = np.linspace(0.25, 1, 4)
parameters = {'sigma': sigma_choices}
cross_validation = GridSearchCV(VectorValuedKRR(), parameters)
cross_validation.fit(X, y)
results = cross_validation.cv_results_
results


data_subset_sizes = [2, 3, 5, 7, 10]
errors = []
for size in data_subset_sizes:
    print(f'{size=}')
    cross_validation = GridSearchCV(VectorValuedKRR(), parameters, cv=2)
    cross_validation.fit(X[:size], y[:size])
    results = cross_validation.cv_results_
    best = np.argmin(results['rank_test_score'])
    best_sigma = results['param_sigma'][best]
    print(f'{best_sigma=}')
    best_test_error = -results['mean_test_score'][best]
    print(f'{best_test_error=}')
    errors.append(best_test_error)

'''
y = y.reshape(120)
K.flat[::10 + 1] += 1e-15
alphas = np.linalg.solve(K, y)

x = X[:2].reshape(2, 12)
k = kstar(x, X.reshape(10, 12))
yhat = k @ alphas
yhat = yhat.reshape(2, 12)
np.mean(np.abs(yhat - y.reshape(10, 12)[:2]))


yhat
'''
