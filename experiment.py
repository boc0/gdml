from time import time
from functools import partial
import numpy as onp
import jax.numpy as np
import pandas as pd
import mlflow
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from utils import to_snake_case, classproperty


class Model:
    """
    Second superclass for model classes which are a part of an experiment.

    Must provide the parameters argument as a dict which maps argument names to
    a list or distribution of parameter values to search.
    """
    cv = RandomizedSearchCV
    parameters = {'sigma': loguniform(10**1, 10**4), 'lamb': loguniform(10**-2, 10**3)}
    n_iter = 3
    n_best = 2

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

    def __init__(self):
        mlflow.set_experiment(to_snake_case(self.__class__.__name__))

    def plot(self):
        cls1, cls2 = self.classes
        data = self.errors
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
                    error = cls.train(Xtrain, ytrain, Xdev, ydev, Xtest, ytest)
                    self.errors[f'error {cls.description}'].append(error)

        if not self.multiple:
            return

        self.errors |= {'samples trained on': data_subset_sizes}
        self.errors = pd.DataFrame(self.errors)
        self.plot()
        mlflow.end_run()
        return self.errors





if __name__ == '__main__':
    data = experiment.run()
