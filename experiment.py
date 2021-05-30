from time import time
from copy import deepcopy
import numpy as onp
import jax.numpy as np
import pandas as pd
import mlflow
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import loguniform
from dipole import VectorValuedKRR, kernel_gauss, kernel_matern
from utils import to_snake_case, classproperty
# from schnet import SchNet as SchNetModel
from schnet import to_spk_dataset, to_batch, train_schnet, squared_error
from sklearn.base import BaseEstimator



class Experiment:
    shuffle = True
    max_size = 100

    @property
    def classes(self) -> list:
        res = []
        for key in dir(self):
            if not key.startswith('__') and key not in ['classes', 'multiple']:
                val = getattr(self, key)
                if isinstance(val, type):
                    res.append(val)
        return res

    @property
    def multiple(self) -> bool:
        return len(self.classes) > 1

    def __init__(self):
        mlflow.set_experiment(to_snake_case(self.__class__.__name__))

    def plot(self, data):
        cls1, cls2 = ex.classes
        fig, ax = plt.subplots()
        desc1, desc2 = (cls.description for cls in (cls1, cls2))
        sns.pointplot(x='samples trained on', y=f'error {desc1}', data=data, color='royalblue', label=desc1)
        sns.pointplot(x='samples trained on', y=f'error {desc2}', data=data, color='orange', label=desc2)
        ax.legend(handles=ax.lines[::len(data)+1], labels=[cls1.__name__, cls2.__name__])
        ax.set_ylabel('error')
        plt.savefig('learning_curve.png')
        mlflow.log_figure(fig, 'learning_curve.png')
        # plt.show()

    def run(self):
        # prepare data
        dataset = np.load('data/HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q.npz')
        X = np.array(dataset['R'])
        y = np.array(dataset['D'])
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

        data_subset_sizes = np.linspace(10, self.max_size, 10, dtype=np.int32)
        errors = {cls: [] for cls in self.classes}

        mlflow.start_run()
        for size in data_subset_sizes:
            Xtrain, ytrain = X[:size], y[:size]
            with mlflow.start_run(nested=True):
                mlflow.log_param('n_samples', size)
                for cls in self.classes:
                    error = cls.train(Xtrain, ytrain, Xdev, ydev, Xtest, ytest)
                    errors[cls].append(error)

        if not self.multiple:
            return

        cls1, cls2 = self.classes
        data = pd.DataFrame({'samples trained on': data_subset_sizes,
                             f'error {cls1.description}': errors[cls1],
                             f'error {cls2.description}': errors[cls2]
                             })
        self.plot(data)
        mlflow.end_run()
        return data




class Model:
    """
    Second superclass for model classes which are a part of an experiment.

    Must provide the parameters argument as a dict which maps argument names to
    a list or distribution of parameter values to search.
    """
    cv = RandomizedSearchCV
    parameters = {'sigma': loguniform(10**1, 10**4), 'lamb': loguniform(10**-2, 10**3)}
    n_iter = 20
    n_best = 5

    @classproperty
    def description(self):
        return to_snake_case(self.__name__)

    @classmethod
    def train(cls, Xtrain, ytrain, Xdev, ydev, Xtest, ytest):
        kwargs = {'n_iter': cls.n_iter} if cls.cv is RandomizedSearchCV else {}
        cv = cls.cv(cls(), cls.parameters, **kwargs)

        start = time()
        size = Xtrain.shape[0]
        with mlflow.start_run(nested=True):
            print(f'\nsize: {size}')
            cv.fit(Xtrain, ytrain)
            results = cv.cv_results_
            indices = onp.argpartition(results['rank_test_score'], cls.n_best)[:cls.n_best]
            # print('errors:')

            def test(params):
                model = cls(**params)
                model.fit(Xtrain, ytrain)
                error = -model.score(Xdev, ydev)
                # print(f'{str(params).ljust(60)} {error:.4f} {angle:.2f}')
                return error

            errors = [test(results['params'][idx]) for idx in indices]
            errors = onp.array(errors)
            _best = onp.argmin(errors)
            best = indices[_best]
            best_params = results['params'][best]
            print(f'best params: {best_params}')
            mlflow.log_params(best_params)
            error = errors[_best]
            print(f'dev error: {error}')
            mlflow.log_metric('dev error', error)
            best_model = cls(**best_params)
            best_model.fit(Xtrain, ytrain)
            error = -best_model.score(Xtest, ytest)
            print(f'test error: {error}')
            mlflow.log_metric('test error', error)
            # mlflow.sklearn.save_model(models[_best], f'mlruns/0/{run.info.run_id}/best_model')
            mlflow.log_metric('time', time() - start)
        print(f'time taken: {time() - start}')
        return error


class Similarity(Experiment):
    class Matern(VectorValuedKRR, Model):
        similarity = kernel_matern

    class Gauss(VectorValuedKRR, Model):
        similarity = kernel_gauss


class SchNetModel(BaseEstimator):
    def __init__(self, n_atom_basis=128, n_interactions=6):
        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.model = None

    def fit(self, Xtrain, ytrain):
        M = Xtrain.shape[0]
        dev_size = max(M // 10, 1)
        Xtrain, Xdev = np.split(Xtrain, [dev_size])
        ytrain, ydev = np.split(ytrain, [dev_size])
        train = to_spk_dataset(Xtrain, ytrain)
        dev = to_spk_dataset(Xdev, ydev)
        self.model = train_schnet(train, dev, size=M,
                                  n_atom_basis=self.n_atom_basis,
                                  n_interactions=self.n_interactions)
        # print('fit completed: ', end='')
        # print(self.score(Xtrain, ytrain))

    def predict(self, inputs):
        return self.model(inputs)

    def score(self, inputs, targets):
        test_batch = to_batch(to_spk_dataset(inputs, targets))
        pred = self.predict(test_batch)
        return -squared_error(pred, test_batch)


class SchNet(Experiment):
    class SchNet(SchNetModel, Model):
        cv = GridSearchCV
        parameters = dict(
            n_atom_basis=[32, 64, 128],
            n_interactions=[3, 6])

    class KRR(VectorValuedKRR, Model):
        ...


if __name__ == '__main__':

    ex = SchNet()
    data = ex.run()
