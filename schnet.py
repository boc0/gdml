import os

import numpy as onp
import jax.numpy as np
import pandas as pd
import torch
import schnetpack as spk
from sklearn.base import BaseEstimator
from schnetpack.data import AtomsData

import matplotlib.pyplot as plt
import seaborn as sns

from dipole import VectorValuedKRR, train
from utils import matern, coulomb, get_data


# loss function
def squared_error(batch, result):
    diff = batch['dipole'].squeeze() - result['dipole'].squeeze()
    err_sq = diff**2
    return err_sq.mean().item()


def train_schnet(
        train, val, size=50, n_epochs=50,
        # hyperparameters
        n_atom_basis=128, n_filters=128, n_gaussians=25,
        n_interactions=6, cutoff=5., cutoff_network=spk.nn.cutoff.CosineCutoff):

    train_loader = spk.AtomsLoader(train, batch_size=2048, shuffle=True)
    val_loader = spk.AtomsLoader(val, batch_size=2048)

    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_filters=n_filters, n_gaussians=n_gaussians, n_interactions=n_interactions,
        cutoff=cutoff, cutoff_network=cutoff_network
    )
    output_alpha = spk.atomistic.DipoleMoment(n_in=n_atom_basis, property='dipole')
    model = spk.AtomisticModel(representation=schnet, output_modules=output_alpha)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = spk.train.build_mse_loss(['dipole'])

    metrics = [spk.metrics.MeanAbsoluteError('dipole')]
    hooks = [
        spk.train.ReduceLROnPlateauHook(
            optimizer,
            patience=5, factor=0.8, min_lr=1e-6,
            stop_after_min=True
        ),
        spk.train.EarlyStoppingHook(8),
        spk.train.CSVHook(f'logs/{size}', metrics=metrics)
    ]

    model_dir = f'models/model {size} {n_atom_basis} {n_interactions}'
    trainer = spk.train.Trainer(
        model_path=model_dir,
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
    )

    trainer.train(device='cpu', n_epochs=n_epochs)

    return torch.load(os.path.join(model_dir, 'best_model'))



def to_spk_dataset(Xcut, ycut):
    _, y = get_data()
    atomsdb = AtomsData('data/data.db')
    n_samples = Xcut.shape[0]
    res = np.array([np.argwhere((y == ycut[i]))[0][0].item() for i in range(n_samples)])
    return atomsdb.create_subset(res)


def to_batch(dataset):
    return next(iter(spk.AtomsLoader(dataset, batch_size=2048)))



class SchNet(BaseEstimator):
    def __init__(self, n_atom_basis=128, n_interactions=6):
        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.model = None

    def fit(self, Xtrain, ytrain):
        M = Xtrain.shape[0]
        dev_size = max(M // 10, 1)
        Xtrain, Xdev = np.split(Xtrain, [dev_size])
        ytrain, ydev = np.split(ytrain, [dev_size])
        train = to_spk_dataset(Xtrain, ytrain)
        dev = to_spk_dataset(Xdev, ydev)
        self.model = train_schnet(train, dev, size=M,
                                  n_atom_basis=self.n_atom_basis,
                                  n_interactions=self.n_interactions)

    def predict(self, inputs):
        return self.model(inputs)

    def score(self, inputs, targets):
        test_batch = to_batch(to_spk_dataset(inputs, targets))
        pred = self.predict(test_batch)
        return -squared_error(pred, test_batch)




if __name__ == '__main__':
    data_subset_sizes = list(np.linspace(10, 100, 10, dtype=int))

    X, y = get_data()

    atomsdb = AtomsData('data/data.db')
    M = len(atomsdb)
    M

    test_indices = onp.random.choice(M, size=500, replace=False)
    test_indices, dev_indices = np.split(test_indices, 2)
    Xtest, ytest = X[test_indices], y[test_indices]
    Xdev, ydev = X[dev_indices], y[dev_indices]

    test_data = atomsdb.create_subset(list(test_indices))
    val = atomsdb.create_subset(list(dev_indices))
    loader = spk.AtomsLoader(test_data, batch_size=512)

    test_batch = next(iter(spk.AtomsLoader(test_data, batch_size=2048)))

    mask = onp.ones(M, dtype=bool)
    mask[test_indices] = False
    X, y = X[mask], y[mask]

    train_indices = onp.random.choice(M-100, size=data_subset_sizes[-1], replace=False)
    X, y = X[train_indices], y[train_indices]

    errors_gdml, errors_schnet = [], []

    for size in data_subset_sizes:
        print(size)
        Xtrain, ytrain = X[:size], y[:size]
        # train(Xtrain, ytrain, Xdev, ydev, Xtest, ytest, n_best=2)

        train = atomsdb.create_subset(train_indices[:size])
        best_model = train_schnet(train, val, size=size)
        prediction = best_model(test_batch)
        # torch.mean((prediction['dipole'] - test_batch['dipole'])**2)
        # diff = prediction['dipole'].squeeze() - test_batch['dipole'].squeeze()
        # torch.mean(diff**2)
        # torch.abs(diff)
        test_error_schnet = squared_error(prediction, test_batch)
        print(f'schnet error: {test_error_schnet:.6f}')
        errors_schnet.append(test_error_schnet)
