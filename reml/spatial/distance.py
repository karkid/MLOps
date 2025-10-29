import numpy as np


def euclidean(X1, X2):
    return np.sqrt(np.sum((X1 - X2) ** 2))


def squared_euclidean(X1, X2):
    return np.sum((X1 - X2) ** 2)


def manhattan(X1, X2):
    return np.sum(np.abs(X1 - X2))


def canberra(X1, X2):
    numinator = np.abs(X1 - X2)
    denominator = np.abs(X1) + np.abs(X2)
    fraction = np.zeros_like(denominator, dtype=float)
    mask = denominator != 0
    fraction[mask] = numinator[mask] / denominator[mask]
    return np.sum(fraction)


def chebyshev(X1, X2):
    return np.max(np.abs(X1 - X2))


def minkowski(X1, X2, p):
    if p <= 0:
        raise ValueError("p must be greater than 0 for Minkowski distance")
    return np.sum(np.abs(X1 - X2) ** p) ** (1 / p)


def cosine_distance(X1, X2):
    dot_product = np.dot(X1, X2)
    norm_X1 = np.linalg.norm(X1)
    norm_X2 = np.linalg.norm(X2)
    if norm_X1 == 0 or norm_X2 == 0:
        return 1.0
    cosine_similarity = dot_product / (norm_X1 * norm_X2)
    return 1 - cosine_similarity
