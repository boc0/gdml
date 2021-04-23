import os

import numpy as onp
import jax.numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from dipole import VectorValuedKRR, train
from utils import matern, coulomb

import schnetpack as spk
from ase.db.jsondb import JSONDatabase
from utils import AtomsDataFix as AtomsData

# loss function
def squared_error(batch, result):
    diff = batch['dipole'].squeeze() - result['dipole'].squeeze()
    # err_sq = torch.linalg.norm(diff, axis=1)**2
    err_sq = diff**2
    return err_sq.mean().item()


def train_schnet(train, val, size=50, n_epochs=50):

    train_loader = spk.AtomsLoader(train, batch_size=2048, shuffle=True)
    val_loader = spk.AtomsLoader(val, batch_size=2048)

    schnet = spk.representation.SchNet(
        n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
        cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff
    )

    output_alpha = spk.atomistic.DipoleMoment(n_in=30, property='dipole')
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

    model_dir = f'model-{size}'
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


if __name__ == '__main__':
    data_subset_sizes = list(np.linspace(5, 10, 2, dtype=int))

    data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
    X = np.array(data['R'])
    y = np.array(data['D'])

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
    batch = next(iter(loader))

    test_batch = next(iter(spk.AtomsLoader(test_data, batch_size=2048)))

    mask = onp.ones(M, dtype=bool)
    mask[test_indices] = False
    X, y = X[mask], y[mask]

    train_indices = onp.random.choice(M-100, size=data_subset_sizes[-1], replace=False)
    X, y = X[train_indices], y[train_indices]

    errors_gdml, errors_schnet = [], []

    for size in data_subset_sizes:
        size = 100
        print(f'size: {size}')

        Xtrain, ytrain = X[:size], y[:size]
        train(Xtrain, ytrain, Xdev, ydev, Xtest, ytest)

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
