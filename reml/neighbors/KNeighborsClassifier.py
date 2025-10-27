import numpy as np
from collections import Counter 
from reml.spatial import distance 

class KNeighborsClassifier:

    def __init__(self, k=5, weights = "uniform"): # weights: "uniform" | "distance"
        self.k = k
        self.weights = weights

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions =  np.array([self._predict(x) for x in X])
        return predictions

    def _predict(self, x):
        # Compute distance from x to all training samples
        distances = np.array([distance.euclidean(x, x_train) for x_train in self.X_train])

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
            w = 1/(distances + ep)

            # sum weight per class
            classes, inv = np.unique(k_labels, return_inverse=True)
            sums = np.zeros(len(classes))
            np.add.at(sums, inv, w)
            print(sums)
            return classes[np.argmax(sums)]
        else:
            raise ValueError("weights must be 'uniform' or 'distance'")

        
    def __repr__(self):
        return f"KNeighborsClassifier(k={self.k})"