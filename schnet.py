import os
from typing import List, Tuple

import jax.numpy as np
import torch
from torch.optim import Adam

import schnetpack as spk
from ase import Atoms
from ase.db.core import now
from ase.db.jsondb import JSONDatabase
from ase.db.row import AtomsRow
from schnetpack.data import AtomsData

from tqdm import tqdm


class BatchJSONDatabase(JSONDatabase):
    def write_all(self, mols: List[Tuple[Atoms, float]], id=None):
        bigdct = {}
        ids = []
        nextid = 1

        if isinstance(self.filename, str) \
           and os.path.isfile(self.filename):
            try:
                bigdct, ids, nextid = self._read_json()
            except (SyntaxError, ValueError):
                pass

        for atoms, mu in mols:
            mtime = now()

            if isinstance(atoms, AtomsRow):
                row = atoms
            else:
                row = AtomsRow(atoms)
                row.ctime = mtime
                row.user = os.getenv('USER')

            dct = {}
            for key in row.__dict__:
                if key[0] == '_' or key in row._keys or key == 'id':
                    continue
                dct[key] = row[key]

            dct['mtime'] = mtime

            dct['data'] = {'dipole': mu}

            constraints = row.get('constraints')
            if constraints:
                dct['constraints'] = constraints


            id = nextid
            ids.append(id)
            nextid += 1

            bigdct[id] = dct

        self._write_json(bigdct, ids, nextid)


data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
X = np.array(data['R'])
y = np.array(data['D'])
z = data['z']


db = BatchJSONDatabase('data/data.json')
mols = []

for i in tqdm(range(50000)):

    pos = X[i]
    mu = y[i]
    mu = [float(x) for x in mu]

    atoms = Atoms('OOHH', positions=pos)
    atoms.set_atomic_numbers(z)

    mols.append((atoms, mu))


db.write_all(mols)
