import numpy as onp
import jax.numpy as np
import pandas as pd
import mlflow
from matplotlib import pyplot as plt
import seaborn as sns


class Experiment:
    shuffle = True
    max_size = 10

    @property
    def classes(self) -> list:
        return [
            x
            for key in dir(self)
            if (x := getattr(self, key))
            and not key.startswith('__')
            and not key == "classes"
            and isinstance(x, type)
            ]

    @property
    def multiple(self) -> bool:
        return len(self.classes) > 1

    def __init__(self):
        pass

    def run(self):
        # prepare data
        data = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
        X = np.array(data['R'])
        y = np.array(data['D'])
        M = X.shape[0]

        test_indices = onp.random.choice(M, size=500, replace=False)
        Xtest, ytest = X[test_indices], y[test_indices]
        Xdev, Xtest = np.split(X[test_indices], 2)
        ydev, ytest = np.split(y[test_indices], 2)

        # remove test samples from X, y
        mask = onp.ones(M, dtype=bool)
        mask[test_indices] = False
        X, y = X[mask], y[mask]

        if self.shuffle:
            train_indices = onp.random.choice(M-self.max_size, size=self.max_size, replace=False)
            X, y = X[train_indices], y[train_indices]

        data_subset_sizes = np.linspace(5, self.max_size, 2, dtype=int)
        errors = {cls: [] for cls in self.classes}

        with mlflow.start_run():
            for size in data_subset_sizes:
                Xtrain, ytrain = X[:size], y[:size]
                with mlflow.start_run():
                    mlflow.log_param('size', size)
                    for cls in self.classes:
                        with mlflow.start_run():
                            error = cls.train(Xtrain, ytrain, Xdev, ydev, Xtest, ytest)
                            errors[cls].append(error)

            if not self.multiple:
                return

            cls1, cls2 = self.classes
            data = pd.DataFrame({'samples trained on': data_subset_sizes,
                                 f'error {cls1.description} (blue)': errors[cls1],
                                 f'error {cls1.description} (orange)': errors[cls2]
                                 })
            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            sns.pointplot(x='samples trained on', y=f'error {cls1.description} (blue)', data=data, s=100, ax=ax, color='royalblue')
            sns.pointplot(x='samples trained on', y=f'error {cls2.description} (orange)', data=data, s=100, ax=ax2, color='orange')
            plt.savefig('learning_curve.png')
            mlflow.log_figure(fig, 'learning_curve.png')
            plt.show()






class Descriptor(Experiment):
    pass




Descriptor()
