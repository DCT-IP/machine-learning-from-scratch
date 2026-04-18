class BaseTransformer:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        raise NotImplementedError("Transform method not immplemented")
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)