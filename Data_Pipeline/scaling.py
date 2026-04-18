import numpy as np
from Data_Pipeline.base import BaseTransformer


class StandardScaler(BaseTransformer):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)