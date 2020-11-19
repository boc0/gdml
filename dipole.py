from functools import partial

import pandas as pd

import jax.numpy as np
from jax import jit, vmap, jacfwd, jacrev
import jax

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from utils import KRR, gaussian, fill_diagonal, descriptor


data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
X = np.array(data['R'])
y = np.array(data['D'])


def hessian(f):
    return jacfwd(jacrev(f))



def hess_at(H, i, j):
    return H[i, :, j, :]

def symmetric(H):
    sym = np.allclose(H, H.T)
    return sym


def kernel(x, x_, sigma=1):
    N, D = x.shape

    Dx, Dx_ = descriptor(x), descriptor(x_)
    D_difference = Dx - Dx_
    k_x = gaussian(x, x_)

    def kronecker(i, j):
        return 1 if i == j else 0
    
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

    K = np.zeros((D, D))
    K = []
    for q in range(D):
        K.append([])
        for p in range(D):
            s = 0
            for k in range(N):
                for l in range(N):
                    s += -k_x * ( phi(q, k) * phi(p, l) - delta_phi(q, k, p, l) )
            K[q].append(s)
    K = np.array(K)
    return K

    _gaussian = partial(gaussian, x_, sigma=sigma)
    hess = hessian(_gaussian)
    H = hess(x)
    K = np.zeros((3,3))
    for i in range(4):
        for j in range(4):
            new = hess_at(H, i, j)
            K += new
            print(symmetric(new), end=' ')
    print(symmetric(K))
    return K

k01 = kernel(X[0], X[1])
k10 = kernel(X[1], X[0])

print(k01)
print(k10)



@jit
def kernel_matrix(X, sigma=1):
    _kernel = partial(kernel, sigma=sigma)
    @vmap
    def _kernels(x):
        vec_kernel = vmap(partial(_kernel, x))
        return vec_kernel(X)
    K = _kernels(X)
    blocks = [list(x) for x in K]
    return np.block(blocks)

def unit(vector):
    return vector / np.linalg.norm(vector)

class VectorValuedKRR(KRR):

    def fit(self, X, y):
        self.X = X
        samples = X.shape[0]
        K = kernel_matrix(X, sigma=self.sigma)
        self.means = np.mean(y, axis=0)
        self.stdevs = np.std(y, axis=0)
        y = (y - self.means) / self.stdevs
        y = y.reshape(samples * 3)
        K = fill_diagonal(K, K.diagonal() + self.lamb)
        alphas = np.linalg.solve(K, y)
        self.alphas = alphas.reshape(samples, 3)

    def predict(self, x):
        def contribution(i, x):
            return kernel(x, X[i], sigma=self.sigma) @ self.alphas[i]
        @vmap
        def predict(x):
            indices = np.arange(self.samples)
            _contribution = vmap(partial(contribution, x=x))
            contributions = _contribution(indices)
            mu = np.sum(contributions, axis=0)
            return mu
        results = predict(x)
        return np.array(results) * self.stdevs + self.means

    def score(self, x, y, angle=False):
        yhat = self.predict(x)
        error = np.linalg.norm(y - yhat, axis=1)**2
        if not angle:
            return np.mean(error)
        angle = []
        for i in range(y.shape[0]):
            angle.append( np.arccos(np.clip(unit(yhat[i]) @ unit(y[i]), -1.0, 1.0)) )
        angle = np.array(angle)
        return np.mean(error), np.mean(angle)






'''

sigma_choices = list(np.linspace(0.25, 2, 8))
lambda_choices = [1e-5]#, 1e-4, 1e-3, 1e-2, 1]
parameters = {'sigma': sigma_choices}#, 'lamb': lambda_choices}
data_subset_sizes = np.linspace(10, 100, 10, dtype=int)
test = slice(20000, 20100)
errors, angles = [], []

from time import time

for size in data_subset_sizes:
    start = time()
    print(f'size: {size}')

    cross_validation = GridSearchCV(VectorValuedKRR(), parameters)
    cross_validation.fit(X[:size], y[:size])
    results = cross_validation.cv_results_
    best = np.argmin(results['rank_test_score'])
    best_params = results['params'][best]
    print(f'best params: {best_params}')
    best_model = VectorValuedKRR(**best_params)
    best_model.fit(X[:size], y[:size])
    best_test_error, angle = (result.item() for result in best_model.score(X[test], y[test], angle=True))
    best_model.save()
    print(f'best test error: {best_test_error}')
    print(f'best mean angle: {angle}')

    errors.append(best_test_error)
    angles.append(angle)
    taken = time() - start
    print(f'time taken: {taken}', end='\n\n')

data = pd.DataFrame({'samples trained on': data_subset_sizes, 'mean squared error norm': errors, 'mean angle': angles})
fig, ax = plt.subplots()
ax2 = ax.twinx()
sns.pointplot(x='samples trained on', y='mean squared error norm', data=data, s=100, ax=ax, color='royalblue')
sns.pointplot(x='samples trained on', y='mean angle', data=data, s=100, ax=ax2, color='coral')



plt.savefig('learning_curve.png')

'''