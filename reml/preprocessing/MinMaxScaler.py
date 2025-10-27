import numpy as np

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.X_min = None
        self.X_max = None
        self.feature_min, self.feature_max = feature_range

    def fit(self, X, y=None):
        X = np.array(X)
        self.X_min = np.min(X, axis=0)
        self.X_max = np.max(X, axis=0)
        return self

    def transform(self, X):
        if self.X_min is None or self.X_max is None:
            raise ValueError("The scaler has not been fitted yet.")
        X = np.array(X)
        range_ = self.X_max - self.X_min
        range_[range_ == 0] = 1  # avoid division by zero
        X_std = (X - self.X_min) / range_
        return X_std * (self.feature_max - self.feature_min) + self.feature_min

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def __repr__(self):
        return f"MinMaxScaler(feature_range=({self.feature_min, self.feature_max}))"

