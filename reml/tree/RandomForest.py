from collections import Counter

import numpy as np

from reml.utils.decorators import auto_repr, check_fitter
from reml.utils.resample import bootstrap_sampling

from .DecisionTree import DecisionTree


@auto_repr
class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=20, n_features=None):
        self.n_trees = n_trees
        self.trees = []
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features,
            )
            X_sample, y_sample = bootstrap_sampling(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    @check_fitter
    def predict(self, X):
        predictions = np.array([tree.prediction(X) for tree in self.trees])
        trees_predictions = np.swapaxes(predictions, 0, 1)
        return np.array([Counter(pred).most_common(1)[0] for pred in trees_predictions])
