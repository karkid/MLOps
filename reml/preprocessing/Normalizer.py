import numpy as np

from reml.utils.decorators import check_fitter


class Normalizer:
    def __init__(self, norm="l2"):
        if norm not in ["l1", "l2"]:
            raise ValueError("norm must be 'l1' or 'l2'")
        self.norm = norm

    def fit(self, X, y=None):
        # Nothing to learn — Normalizer just scales each row
        self.is_fitted = True
        return self

    @check_fitter
    def transform(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.norm == "l2":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
        else:  # l1
            norms = np.sum(np.abs(X), axis=1, keepdims=True)

        # Check for zero vectors
        if np.any(norms == 0):
            raise ValueError("Found zero-length vector, cannot normalize")

        return X / norms

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def __repr__(self):
        return f"Normalizer(norm={self.norm})"
