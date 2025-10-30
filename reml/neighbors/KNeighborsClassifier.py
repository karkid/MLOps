from collections import Counter

import numpy as np

from reml.spatial import distance
from reml.utils.decorators import auto_repr, check_fitter


@auto_repr
class KNeighborsClassifier:
    def __init__(self, k=5, weights="uniform"):  # weights: "uniform" | "distance"
        if k < 1:
            raise ValueError("k must be greater than zero")
        if weights not in ["uniform", "distance"]:
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = k
        self.weights = weights
        self.is_fitted = False
        self.classes_ = None
        self._class_map = None

    def fit(self, X, y):
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")

        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

        # CRITICAL: For consistent class order, always use sorted classes
        self.classes_ = np.sort(np.unique(y))

        # Map labels to indices 0, 1, 2, etc.
        self._class_map = {c: i for i, c in enumerate(self.classes_)}

        self.is_fitted = True
        return self

    @check_fitter
    def predict_proba(self, X):
        """Predict probability estimates."""
        X = np.array(X)
        probas = []
        for x in X:
            # Compute distances to all training points
            distances = np.array(
                [distance.euclidean(x, x_train) for x_train in self.X_train]
            )
            points_less = self.X_train.ravel() < x

            # Sort by distance first, then favor points below the query point
            k = min(self.k, len(distances))
            # Points less than x come first
            order = np.lexsort((points_less, distances))
            k_indices = order[:k]
            k_labels = self.y_train[k_indices]

            # For each class in order, calculate its probability
            probas_k = np.zeros(len(self.classes_))
            for idx, class_ in enumerate(self.classes_):
                probas_k[idx] = np.sum(k_labels == class_) / k

            probas.append(probas_k)
        return np.array(probas)

    @check_fitter
    def predict(self, X):
        X = np.array(X)
        predictions = np.array([self._predict(x) for x in X])
        return predictions

    def _predict(self, x):
        # Compute distance from x to all training samples
        distances = np.array(
            [distance.euclidean(x, x_train) for x_train in self.X_train]
        )

        # Select indices of the k nearest neighbors
        k = min(self.k, len(distances))
        k_indices = np.argpartition(distances, k - 1)[:k]
        k_labels = self.y_train[k_indices]

        if self.weights == "uniform":
            # majority vote
            return Counter(k_labels).most_common(1)[0][0]
        elif self.weights == "distance":
            # inverse distance weight
            ep = 1e-12
            w = 1 / (distances + ep)

            # sum weight per class
            classes, inv = np.unique(k_labels, return_inverse=True)
            sums = np.zeros(len(classes))
            np.add.at(sums, inv, w)
            print(sums)
            return classes[np.argmax(sums)]
        else:
            raise ValueError("weights must be 'uniform' or 'distance'")
