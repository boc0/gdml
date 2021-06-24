from dipole import VectorValuedKRR, kernel_gauss, kernel_matern
from experiment import Experiment


class Similarity(Experiment):
    class Matern(VectorValuedKRR):
        kernel = kernel_matern

    class Gauss(VectorValuedKRR):
        kernel = kernel_gauss


if __name__ == '__main__':
    ex = Similarity()
    ex.run()
