from time import time

import mlflow
import pandas as pd
import numpy as onp
import jax.numpy as np
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt

from dipole import VectorValuedKRR, train

from sklearn.utils.fixes import loguniform

parameters_random = {'sigma': loguniform(10**1, 10**5), 'lamb': loguniform(10**-2, 10**3)}
parameters_grid = {'sigma': list(np.logspace(1, 4, 19)), 'lamb': list(np.logspace(-2, 3, 21))}


def cv_instance(kind='grid'):
    if kind == 'grid':
        return GridSearchCV(VectorValuedKRR(), parameters_grid, verbose=True)
    elif kind == 'random':
        return RandomizedSearchCV(VectorValuedKRR(), parameters_random, n_iter=200, verbose=True, random_state=0)
    else:
        raise ValueError(f'Unrecognized kind of Cross-Validation: {kind}')


def learning_curve(cv='grid', randomize=False):
    mlflow.sklearn.autolog()

    data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
    X = np.array(data['R'])
    y = np.array(data['D'])
    M = X.shape[0]

    data_subset_sizes = np.linspace(10, 100, 10, dtype=int)

    test_indices = onp.random.choice(M, size=500, replace=False)
    Xtest, ytest = X[test_indices], y[test_indices]

    # remove test samples from X, y
    mask = onp.ones(M, dtype=bool)
    mask[test_indices] = False
    X, y = X[mask], y[mask]

    if randomize:
        train_indices = onp.random.choice(M-100, size=data_subset_sizes[-1], replace=False)
        X, y = X[train_indices], y[train_indices]

    errors, angles = [], []

    with mlflow.start_run() as run:
        for size in data_subset_sizes:
            Xtrain, ytrain = X[:size], y[:size]
            error, angle = train(Xtrain, ytrain, Xtest, ytest, cv=cv_instance(kind=cv))
            errors.append(error)
            angles.append(angle)

        data = pd.DataFrame({'samples trained on': data_subset_sizes, 'mean squared error norm (blue)': errors, 'mean angle (orange)': angles})
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        sns.pointplot(x='samples trained on', y='mean squared error norm (blue)', data=data, s=100, ax=ax, color='royalblue')
        sns.pointplot(x='samples trained on', y='mean angle (orange)', data=data, s=100, ax=ax2, color='coral')
        plt.savefig(f'learning_curve.png')
        mlflow.log_figure(fig, 'learning_curve.png')

        run_id = run.info.run_id
    return run_id
