from sklearn.base import BaseEstimator

class KRR(BaseEstimator):
    def __init__(self, sigma=1):
        self.lamb = 1e-15
        self.sigma = sigma

    @property
    def samples(self):
        return self.X.shape[0]
