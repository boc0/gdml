import os
from typing import List, Tuple

import jax.numpy as np

from ase import Atoms
from ase.db.sqlite import SQLite3Database

from tqdm import tqdm

data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
X = np.array(data['R'])
y = np.array(data['D'])
z = data['z']


db = SQLite3Database('data/data.db')
mols = []

for i in tqdm(range(50000)):

    pos = X[i]
    mu = y[i]
    mu = [float(x) for x in mu]

    atoms = Atoms('OOHH', positions=pos)
    atoms.set_atomic_numbers(z)

    db.write(atoms, data={'dipole': mu})
