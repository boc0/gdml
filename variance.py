import argparse

import jax.numpy as np
import numpy as onp

import mlflow
from tqdm import tqdm

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

onp.random.seed(0)

mlflow.set_experiment('variance')


def variance(size):
    data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
    X = np.array(data['R'])
    y = np.array(data['D'])
    M = X.shape[0]

    test_indices = onp.random.choice(M, size=500, replace=False)
    Xtest, ytest = X[test_indices], y[test_indices]

    # remove test samples from X, y
    mask = onp.ones(M, dtype=bool)
    mask[test_indices] = False
    X, y = X[mask], y[mask]

    repetitions = 10
    errors, angles, results = [], [], []

    results.append([])
    with mlflow.start_run():
        mlflow.log_param('n_samples', size)
        for j in tqdm(range(repetitions)):
            train_indices = onp.random.choice(M-500, size=size, replace=False)
            Xtrain, ytrain = X[train_indices], y[train_indices]
            error, angle, result = train(Xtrain, ytrain, Xtest, ytest, cv=cv_instance('random'), return_results=True)
            results.append(result)
            errors.append(error)
            angles.append(angle)

        path = f'{size}.npz'
        np.savez(path, errors=errors, angles=angles, results=results)
        mlflow.log_artifact(path)

    return errors, angles, results


res = onp.load('results/variance/100.npz', allow_pickle=True)
list(res.keys())

results = res['results'][1]
results.keys()
lambs, sigmas, scores = (onp.array(results[key]).astype(onp.float32) for key in ['param_lamb', 'param_sigma', 'mean_test_score'])
lambs.shape

lambs, sigmas = onp.meshgrid(lambs, sigmas)
lambs.shape == sigmas.shape
scores.shape


lambs.shape

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
Z


fig = plt.figure()
ax = plt.axes(projection='3d')
ax = fig.add_subplot(111, projection='3d')
%matplotlib notebook
ax.plot_trisurf(lambs, sigmas, scores)
fig.savefig('rez.png')
ax.savefig('rez.png')
plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('size', type=int)

    size = parser.parse_args().size
    errors, angles, results = variance(size)
