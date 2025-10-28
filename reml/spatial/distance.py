import numpy as np


def euclidean(X1, X2):
    return np.sqrt(np.sum((X1 - X2) ** 2))
