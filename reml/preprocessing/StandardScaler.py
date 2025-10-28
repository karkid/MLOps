import numpy as np

from reml.utils.decorators import check_fitter


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.scale_ = None
        self.is_fitted = False

    def fit(self, X, y=None):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=1)  # Using N-1 denominator
        # Handle zeros in scale
        self.scale_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.is_fitted = True
        return self

    @check_fitter
    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def __repr__(self):
        return "StandardScaler()"
