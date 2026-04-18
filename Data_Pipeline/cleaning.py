import numpy as np
from Data_Pipeline.base import BaseTransformer


class SimpleImputer(BaseTransformer):
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.fill_values = None

    def fit(self, X, y=None):
        if self.strategy == "mean":
            self.fill_values = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.fill_values = np.nanmedian(X, axis=0)
        elif self.strategy == "most_frequent":
            self.fill_values = np.array([
                np.bincount(col[~np.isnan(col)].astype(int)).argmax()
                for col in X.T
            ])
        else:
            raise ValueError("Invalid strategy")
        return self

    def transform(self, X):
        X_copy = X.copy()
        inds = np.where(np.isnan(X_copy))
        X_copy[inds] = np.take(self.fill_values, inds[1])
        return X_copy