import numpy as np


def bootstrap_sampling(X, y, random_state=42):
    n_samples = X.shape[0]
    rng = np.random.default_rng(random_state)
    idxs = rng.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]
