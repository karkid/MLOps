import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        X = np.array(X)
        self.mean = np.mean(X, axis=0) # axis = 0 feature(cloumn)
        self.std = np.std(X, axis=0) # axis = 1 datum(rows)
        return self

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet.")
        X = np.array(X)
        return (X - self.mean) / self.std

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
