import numpy as np

class Normalizer:
    def __init__(self, norm='l2'):
        self.norm = norm  # allow 'l1' or 'l2'

    def fit(self, X, y=None):
        # Nothing to learn — Normalizer just scales each row
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        
        if self.norm == 'l2':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
        elif self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
        else:
            raise ValueError("norm must be 'l1' or 'l2'")
        
        # Avoid division by zero (if a row is all zeros)
        norms[norms == 0] = 1
        
        return X / norms

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
