from time import time

import numpy as onp
import jax.numpy as np

from dipole import VectorValuedKRR, GridSearchCV


SIZE = 50

data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')

X = np.array(data['R'])
y = np.array(data['D'])

M = X.shape[0]
for _ in range(3):
    start = time()
    indices = onp.random.choice(M, size=SIZE, replace=False)

    sigma_choices = [1000.0]
    lambda_choices = [1e-10, 1e-5, 1e-2, 1, 10, 100]
    parameters = {'sigma': sigma_choices, 'lamb': lambda_choices}

    Xtrain, ytrain = X[indices], y[indices]

    cross_validation = GridSearchCV(VectorValuedKRR(), parameters)
    cross_validation.fit(Xtrain, ytrain)
    results = cross_validation.cv_results_
    print(results)
    print('took ', time() - start)

