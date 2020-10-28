import numpy as np
from sklearn.kernel_ridge import KernelRidge
from scipy.spatial.distance import pdist, cdist, squareform

def descriptor(x: np.ndarray, Z=[8,8,1,1]) -> np.ndarray:
    atoms, dimension = x.shape
    C = np.zeros((atoms, atoms))
    for i in range(atoms):
        for j in range(atoms):
            if i == j:
                C[i, j] = 0.5 * Z[i] ** 2.4
            else:
                C[i, j] = Z[i] * Z[j] / np.linalg.norm(x[i] - x[j])
    return C

def kernel(x, x_, sigma=1) -> float:
    # x, x_ = descriptor(x).flatten(), descriptor(x_).flatten()
    return np.exp(-np.linalg.norm(x - x_)**2 / sigma)


def gaussian(X,sigma=1):
    X = [descriptor(x).flatten() for x in X]
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    return np.exp(-pairwise_dists ** 2 / sigma ** 2)

def kstar(x, X, sigma=1):
    x = [descriptor(x_).flatten() for x_ in x]
    X = [descriptor(x_).flatten() for x_ in X]
    dists = cdist(x, X)
    kstar = np.exp(-dists ** 2 / sigma ** 2)
    return kstar

if __name__ == '__main__':
    data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
    X = data['R']
    y = data['E']
    K = gaussian(X)

    model = KernelRidge(kernel='precomputed')
    model.fit(K, y)

    k = kstar([X[10002]], X[:10000])
    k.shape
    yhat = model.predict(k)
    yreal = y[10002]
    yhat, yreal
    
