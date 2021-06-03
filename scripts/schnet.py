from sklearn.model_selection import GridSearchCV
from dipole import VectorValuedKRR
from schnet import SchNet as SchNetModel
from experiment import Experiment


class SchNet(Experiment):
    class SchNet(SchNetModel):
        cv = GridSearchCV
        parameters = dict(
            n_atom_basis=[32, 64, 128],
            n_interactions=[3, 6])

    class KRR(VectorValuedKRR):
        ...


if __name__ == '__main__':
    ex = SchNet()
    ex.run()
