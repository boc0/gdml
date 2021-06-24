import os
from time import time
import numpy as onp
import jax.numpy as np
import pandas as pd
import mlflow
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform
from utils import to_snake_case, classproperty


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
    n_iter = 60
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
            print(f'{cls.__name__} best params: {best_params}')
            mlflow.log_params(best_params)
            error = -cv.score(Xtest, ytest)
            print(f'error: {error}')
            mlflow.log_metric('error', error)
            mlflow.log_metric('time', time() - start)
        print(f'time taken: {time() - start}')
        return error


DEVSIZE = 0.1
EXPERIMENTS_FOLDER = 'experiments'


class Experiment:
    shuffle = True
    max_size = 100
    n_subsets = 10

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
        mlflow.set_experiment(self.name)
        os.makedirs(self.folder, exist_ok=True)

    def plot(self):
        cls1, cls2 = self.classes
        data = self.errors
        fig, ax = plt.subplots()
        desc1, desc2 = (cls.description for cls in (cls1, cls2))
        sns.pointplot(x='samples trained on', y=f'error {desc1}', data=data, color='royalblue', label=desc1)
        sns.pointplot(x='samples trained on', y=f'error {desc2}', data=data, color='orange', label=desc2)
        ax.legend(handles=ax.lines[::len(data)+1], labels=[cls1.__name__, cls2.__name__])
        ax.set_ylabel('error')
        plt.savefig(os.path.join(self.folder, 'learning_curve.png'))
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

        # remove test samples from X, y
        mask = onp.ones(M, dtype=bool)
        mask[test_indices] = False
        X, y = X[mask], y[mask]

        if self.shuffle:
            train_indices = onp.random.choice(M-self.max_size, size=self.max_size, replace=False)
            X, y = X[train_indices], y[train_indices]

        data_subset_sizes = np.linspace(10, self.max_size, self.n_subsets, dtype=np.int32)
        # errors = {cls.description: [] for cls in self.classes}
        self.errors = {f'error {cls.description}': [] for cls in self.classes}

        mlflow.start_run()
        for size in data_subset_sizes:
            Xtrain, ytrain = X[:size], y[:size]

            with mlflow.start_run(nested=True):
                print(f'\nn_samples: {size}')
                mlflow.log_param('n_samples', size)
                for cls in self.classes:
                    error = cls.train(Xtrain, ytrain, Xtest, ytest)
                    self.errors[f'error {cls.description}'].append(error)

        if not self.multiple:
            return

        self.errors |= {'samples trained on': data_subset_sizes}
        self.errors = pd.DataFrame(self.errors)
        self.plot()
        mlflow.end_run()
        return self.errors
