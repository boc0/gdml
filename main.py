import numpy as np
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


def gaussian(X,sigma=1):
    X = [descriptor(x).flatten() for x in X]
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    return np.exp(-pairwise_dists ** 2 / sigma ** 2)

def kstar(x, X, sigma=1):
    x = descriptor(x).flatten()
    X = [descriptor(x_).flatten() for x_ in X]
    dists = cdist([x], X)
    kstar = np.exp(-dists ** 2 / sigma ** 2)
    return kstar[0, :]

if __name__ == '__main__':
    data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
    X = data['R']
    y = data['E']
    K = gaussian(X)

    lamb = 1
    alphas = np.linalg.solve(K + lamb * np.eye(50000), y)
    alphas = alphas[:, 0]

    k = kstar(X[1], X)
    yhat = np.sum(alphas * k)
    yreal = y[1]
    yhat, yreal
