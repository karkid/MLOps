from collections import Counter

import numpy as np

from reml.utils.decorators import auto_repr, check_fitter


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.left = left
        self.right = right
        self.threshold = threshold
        self.value = value

    def is_leaf(self):
        return self.value is not None


@auto_repr
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=20, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.n_features = (
            X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        )
        self.root = self._grow_tree(X, y)

        self.is_fitted = True
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # cutoff condition
        if (
            depth > self.max_depth
            or n_samples < self.min_samples_split
            or n_labels == 1
        ):
            most_common_label = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_label)

        # find the best split
        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feature_idx, best_threshold = self._best_split(X, y, feature_idxs)

        # If no good split found, create leaf node
        if best_feature_idx is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_label)

        # Create left and right child
        left_idxs, right_idxs = self._split(X[:, best_feature_idx], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature_idx, best_threshold, left, right)

    def _best_split(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # weighted child entropy
        n = len(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        # If split results in empty partitions, return 0 gain
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n_left, n_right = len(left_idxs), len(right_idxs)
        left_e, right_e = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        children_entropy = (n_left / n) * left_e + (n_right / n) * right_e

        # information gain
        return parent_entropy - children_entropy

    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """Calculate the entropy of a set of labels."""
        if len(y) == 0:
            return 0
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum(p * np.log2(p) for p in ps if p > 0)

    @check_fitter
    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
