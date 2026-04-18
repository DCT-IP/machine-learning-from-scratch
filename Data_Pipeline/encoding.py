import numpy as np
from Data_Pipeline.base import BaseTransformer


class OneHotEncoder(BaseTransformer):
    def __init__(self):
        self.categories = None

    def fit(self, X, y=None):
        self.categories = [np.unique(col) for col in X.T]
        return self

    def transform(self, X):
        encoded_cols = []

        for i, col in enumerate(X.T):
            cats = self.categories[i]
            one_hot = np.zeros((len(col), len(cats)))

            for j, val in enumerate(col):
                idx = np.where(cats == val)[0][0]
                one_hot[j, idx] = 1

            encoded_cols.append(one_hot)

        return np.hstack(encoded_cols)