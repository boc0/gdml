import warnings
from dipole import VectorValuedKRR
from experiment import Atoms
warnings.filterwarnings('ignore')


class Molecules(Atoms):
    class HOOH(VectorValuedKRR):
        ...

    class Cubane(VectorValuedKRR):
        data = 'cubane.DFT.NVT.PBE-TS.l1t.MD.300K.R_E_F_D'

    class Glycine(VectorValuedKRR):
        data = 'glycine.DFT.PBE-TS.l1t.MD.500K.R_E_F_D'

    class Alanine(VectorValuedKRR):
        data = 'alanine.DFT.PBE-TS.l1t.MD.500K.R_E_F_D'


if __name__ == '__main__':
    ex = Molecules()
    ex.run()
