import numpy as np

from reml.utils.decorators import auto_repr, check_fitter


@auto_repr
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.feature_min, self.feature_max = feature_range
        self.is_fitted = False

    def fit(self, X, y=None):
        X = np.array(X)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.scale_ = (self.feature_max - self.feature_min) / (self.max_ - self.min_)
        self.is_fitted = True
        return self

    @check_fitter
    def transform(self, X):
        X = np.array(X)
        X_std = (X - self.min_) * self.scale_
        return X_std + self.feature_min

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
