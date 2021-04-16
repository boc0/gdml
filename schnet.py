import jax.numpy as np
import schnetpack as spk
from ase import Atoms
import torch
from torch.optim import Adam

from ase.db.jsondb import JSONDatabase
from ase.db.mysql import MySQLDatabase
from schnetpack.data import AtomsData
from ase.db import connect
from ase.calculators.interface import Calculator
from tqdm import tqdm


data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
X = np.array(data['R'])
y = np.array(data['D'])
z = data['z']


class FakeCalculator(Calculator):
    def __init__(self, mu):
        self.mu = mu

    def get_dipole_moment(self, atoms):
        return self.mu


db = JSONDatabase('data/data.json')

for i in tqdm(range(2000)):

    pos = X[i]
    mu = y[i]
    mu = [float(x) for x in mu]

    atoms = Atoms('OOHH', positions=pos)
    atoms.set_atomic_numbers(z)

    db.write(atoms, data={'dipole': mu})


atomsdb = AtomsData('data/data.json')

train, val, test = spk.train_test_split(
        data=atomsdb,
        num_train=10,
        num_val=5
    )

train_loader = spk.AtomsLoader(train, batch_size=2, shuffle=True)
val_loader = spk.AtomsLoader(val, batch_size=2)



schnet = spk.representation.SchNet(
    n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
    cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff
)

output_alpha = spk.atomistic.DipoleMoment(n_in=30, property='dipole')
model = spk.AtomisticModel(representation=schnet, output_modules=output_alpha)


# loss function
def mse_loss(batch, result):
    diff = batch['dipole']-result['dipole']
    err_sq = torch.mean(diff ** 2)
    return err_sq

# build optimizer
optimizer = Adam(model.parameters(), lr=1e-2)

import schnetpack.train as trn

loss = trn.build_mse_loss(['dipole'])

metrics = [spk.metrics.MeanAbsoluteError('dipole')]
hooks = [
    trn.CSVHook(log_path='logs', metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

trainer = trn.Trainer(
    model_path='model',
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)


device = "cpu" # change to 'cpu' if gpu is not available
n_epochs = 2 # takes about 10 min on a notebook GPU. reduces for playing around
trainer.train(device=device, n_epochs=n_epochs)
best_model = torch.load('model/best_model')

converter = spk.data.AtomsConverter(device=device)
at, props = test.get_properties(idx=4)

inputs = converter(at)
best_model(inputs)

print(len(train_loader))
for batch in train_loader:
    print(batch)
