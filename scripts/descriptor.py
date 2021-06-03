from functools import partial
from dipole import VectorValuedKRR, kernel as kernel_jax
from utils import coulomb
from experiment import Experiment


nop = lambda x: x


class Descriptor(Experiment):
    class Coulomb(VectorValuedKRR):
        kernel = partial(kernel_jax, descriptor=coulomb)

    class Identity(VectorValuedKRR):
        kernel = partial(kernel_jax, descriptor=nop)


if __name__ == '__main__':
    ex = Descriptor()
    ex.run()
