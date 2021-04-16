import jax.numpy as np
import numpy as onp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dipole import VectorValuedKRR

model = VectorValuedKRR(lamb=1.313, sigma=3833.3)
try:
    model.load(100)
except Exception:
    data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
    X = np.array(data['R'])
    y = np.array(data['D'])
    M = X.shape[0]
    model.fit(X[:100], y[:100])

test_indices = onp.random.choice(M, size=500, replace=False)
Xtest, ytest = X[test_indices], y[test_indices]

model.score(Xtest, ytest)

yhat = model.predict(Xtest)

d = yhat - ytest

x, y, z = np.hsplit(d, 3)
x, y, z = (np.squeeze(each) for each in [x, y, z])
x.shape


fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
plot = ax.plot(x, y, z)
plt.show()
plt.savefig("plot.png")
model.save()
