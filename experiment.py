from dipole import VectorValuedKRR, kernel_gauss, kernel_matern
import os, psutil
from time import time
import numpy as onp
import jax.numpy as np
import pandas as pd
import mlflow
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform
from utils import to_snake_case, classproperty, get_data
from jax.interpreters import xla
import pandas


PARAM_GRID_RANDOM = {'sigma': loguniform(10**1, 10**4), 'lamb': loguniform(10**-2, 10**3)}
sigmas = list(np.logspace(1, 4, 38))
lambdas = list(np.logspace(-2, 3, 42))
PARAMETERS = {'sigma': sigmas, 'lamb': lambdas}


class Model:
    """
    Second superclass for model classes which are a part of an experiment.

    Must provide the parameters argument as a dict which maps argument names to
    a list or distribution of parameter values to search.
    """
    cv = RandomizedSearchCV
    n_iter = 20
    parameters = PARAM_GRID_RANDOM

    @classproperty
    def description(self):
        return to_snake_case(self.__name__)

    @classmethod
    def train(cls, Xtrain, ytrain, Xtest, ytest):
        kwargs = {
            'n_iter': cls.n_iter,
            'random_state': 1
            } if cls.cv is RandomizedSearchCV else {}
        cv = cls.cv(cls(), cls.parameters, **kwargs)

        start = time()
        with mlflow.start_run(nested=True):
            cv.fit(Xtrain, ytrain)
            results = cv.cv_results_
            best = onp.argmin(results['mean_test_score'])
            best_params = results['params'][best]
            results = pandas.DataFrame(cv.cv_results_)
            filename = 'results.xlsx'
            results.to_excel(filename)
            mlflow.log_artifact(filename, '')
            # print(f'{cls.__name__} best params: {best_params}')
            mlflow.log_params(best_params)
            error = -cv.score(Xtest, ytest)
            # print(f'error: {error}')
            mlflow.log_metric('error', error)
            mlflow.log_metric('time', time() - start)
        # print(f'time taken: {time() - start}')
        return error


DEVSIZE = 0.1
EXPERIMENTS_FOLDER = 'experiments/results'
THRESHOLD = 1
COLORS = ['royalblue', 'orange', 'darkmagenta', 'darkgreen']

class Experiment:
    shuffle = True
    min_size = 10
    max_size = 100
    test_size = 500
    n_subsets = 10
    n_runs = 6

    @property
    def classes(self) -> list:
        res = []
        for key in dir(self):
            if not key.startswith('__') and key not in ['classes', 'multiple']:
                val = getattr(self, key)
                if isinstance(val, type):
                    if Model not in val.__mro__:
                        val = type(val.__name__, (Model,) + val.__mro__, dict(val.__dict__))
                    res.append(val)
        return res

    @property
    def multiple(self) -> bool:
        return len(self.classes) > 1

    @property
    def name(self) -> str:
        return to_snake_case(self.__class__.__name__)

    @property
    def folder(self) -> str:
        return os.path.join(EXPERIMENTS_FOLDER, self.name)

    def __init__(self):
        # if not self.multiple:
        #     raise ValueError('Experiment should include multiple models!')
        mlflow.set_experiment(self.name)
        os.makedirs(self.folder, exist_ok=True)

    def reduce(self):
        try:
            errs = self.errors
        except AttributeError:
            raise ValueError('Trying to reduce while errors not yet computed')

        descriptions = [cls.description for cls in self.classes]
        # for cls in descriptions:
        #     while np.max(errs[cls]) >= THRESHOLD:  # delete rows with very high error
        #         row, column = np.unravel_index(np.argmax(errs[cls]), errs[cls].shape)
        #         for _cls in descriptions:
        #             errs[_cls] = np.delete(errs[_cls], row, 0)

        res = {f'error {cls.description}': onp.mean(errs[cls.description], axis=0) for cls in self.classes}
        print(errs)
        print(res)
        return res

    def plot(self):
        errors = self.reduce()
        errors |= {'samples trained on': self.sizes}

        errors = pd.DataFrame(errors)
        cls1, cls2 = self.classes
        data = errors
        fig, ax = plt.subplots()
        desc1, desc2 = (cls.description for cls in (cls1, cls2))
        sns.pointplot(x='samples trained on',
                      y=f'error {desc1}', data=data, color='royalblue', label=desc1)
        sns.pointplot(x='samples trained on',
                      y=f'error {desc2}', data=data, color='orange', label=desc2)
        ax.legend(handles=ax.lines[::len(data)+1], labels=[cls1.__name__, cls2.__name__])
        ax.set_ylabel('error')
        fig_path = os.path.join(self.folder, 'learning_curve.png')
        plt.savefig(fig_path)
        mlflow.log_figure(fig, 'learning_curve.png')
        mlflow.log_artifact(fig_path, '')
        # plt.show()

    @property
    def sizes(self):
        return np.linspace(self.min_size, self.max_size, self.n_subsets, dtype=np.int32)

    def run(self):
        # prepare data
        X, y = get_data()
        M = X.shape[0]

        test_indices = onp.random.choice(M, size=self.test_size, replace=False)
        Xtest, ytest = X[test_indices], y[test_indices]

        # remove test samples from X, y
        mask = onp.ones(M, dtype=bool)
        mask[test_indices] = False
        X, y = X[mask], y[mask]

        sizes = self.sizes
        n_runs = self.n_runs
        self.errors = {cls.description: onp.zeros((n_runs, len(sizes))) for cls in self.classes}

        with mlflow.start_run():
            for i in tqdm(range(n_runs)):
                if self.shuffle:
                    train_indices = onp.random.choice(M-self.test_size, size=self.max_size, replace=False)
                    Xtrain, ytrain = X[train_indices], y[train_indices]
                else:
                    train_start = onp.random.choice(M-self.test_size, size=self.max_size)[0]
                    train_indices = slice(train_start, train_start+self.max_size)
                    Xtrain, ytrain = X[train_indices], y[train_indices]
                with mlflow.start_run(nested=True):
                    mlflow.log_param('run', i)
                    for j, size in tqdm(list(enumerate(sizes))):
                        Xcut, ycut = Xtrain[:size], ytrain[:size]
                        with mlflow.start_run(nested=True):
                            # print(f'\nn_samples: {size}')
                            mlflow.log_param('n_samples', size)
                            for cls in self.classes:
                                error = cls.train(Xcut, ycut, Xtest, ytest)
                                self.errors[cls.description][i, j] = error


                xla._xla_callable.cache_clear()

        self.plot()
        return self.errors



class Atoms(Experiment):
    min_size = 5
    max_size = 10
    test_size = 1
    n_subsets = 2
    n_runs = 2
    shuffle = False
    data = 'HOOH.DFT.PBE-TS.light.MD.500K.50k.R_E_F_D_Q'

    def run(self):
        # prepare data
        X, y = get_data(self.data)
        M = X.shape[0]

        test_indices = onp.random.choice(M, size=self.test_size, replace=False)
        Xtest, ytest = X[test_indices], y[test_indices]

        # remove test samples from X, y
        mask = onp.ones(M, dtype=bool)
        mask[test_indices] = False
        X, y = X[mask], y[mask]

        sizes = self.sizes
        n_runs = self.n_runs
        self.errors = {cls.description: onp.zeros((n_runs, len(sizes))) for cls in self.classes}

        with mlflow.start_run():
            for i in tqdm(range(n_runs)):
                if self.shuffle:
                    train_indices = onp.random.choice(M-self.test_size, size=self.max_size, replace=False)
                    Xtrain, ytrain = X[train_indices], y[train_indices]
                else:
                    train_start = onp.random.choice(M-self.test_size, size=self.max_size)[0]
                    train_indices = slice(train_start, train_start+self.max_size)
                    Xtrain, ytrain = X[train_indices], y[train_indices]
                with mlflow.start_run(nested=True):
                    mlflow.log_param('run', i)
                    for j, size in tqdm(list(enumerate(sizes))):
                        Xcut, ycut = Xtrain[:size], ytrain[:size]
                        with mlflow.start_run(nested=True):
                            # print(f'\nn_samples: {size}')
                            mlflow.log_param('n_samples', size)
                            for cls in self.classes:
                                error = cls.train(Xcut, ycut, Xtest, ytest)
                                self.errors[cls.description][i, j] = error


                xla._xla_callable.cache_clear()

        print(self.errors)
        return self.errors



class HOOHExp(Atoms):
    class HOOH(VectorValuedKRR):
        n_iter = 2


ex = HOOHExp()
ex.run()


class Cubane(Atoms):
    data = 'cubane.DFT.NVT.PBE-TS.l1t.MD.300K.R_E_F_D'
    class Cubane(VectorValuedKRR):
        n_iter = 2

ex = Cubane()
ex.run()

class Glycine(Atoms):
    data = 'glycine.DFT.PBE-TS.l1t.MD.500K.R_E_F_D'
    class Glycine(VectorValuedKRR):
        n_iter = 2

X, y = get_data('cubane.DFT.NVT.PBE-TS.l1t.MD.300K.R_E_F_D')
X.shape

ex = Glycine()
ex.run()

class Alanine(Atoms):
    data = 'alanine.DFT.PBE-TS.l1t.MD.500K.R_E_F_D'
    class Alanine(VectorValuedKRR):
        n_iter = 2

ex = Alanine()
ex.run()
'''

class Similarity(Experiment):
    min_size = 5
    max_size = 10
    test_size = 1
    n_subsets = 2
    n_runs = 2
    shuffle = False

    class Matern(VectorValuedKRR):
        n_iter = 2
        kernel = kernel_matern

    class Gauss(VectorValuedKRR):
        n_iter = 2
        kernel = kernel_gauss


if __name__ == '__main__':
    ex = Similarity()
    ex.run()
'''
