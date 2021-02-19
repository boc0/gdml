import argparse

import jax.numpy as np
import numpy as onp

import mlflow
from tqdm import tqdm

from dipole import train
from learning_curve import cv_instance

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

    np.savez(f'variance/{size}.npz', errors=errors, angles=angles, results=results)
    return errors, angles, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('size', type=int)

    size = parser.parse_args().size
    errors, angles, results = variance(size)
