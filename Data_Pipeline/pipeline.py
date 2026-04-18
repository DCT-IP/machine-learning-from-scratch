class Pipeline:
    def __init__(self, steps):
        self.steps = steps  
    
    def fit(self, X, y=None):
        for name, step in self.steps[:-1]:
            X = step.fit_transform(X, y)
        self.steps[-1][1].fit(X, y)
        return self

    def transform(self, X):
        for name, step in self.steps:
            X = step.transform(X)
        return X

    def fit_transform(self, X, y=None):
        for name, step in self.steps:
            X = step.fit_transform(X, y)
        return X

