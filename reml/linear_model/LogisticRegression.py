import numpy as np

from reml.utils.decorators import check_fitter


class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iteration=1000):
        self.learning_rate = float(learning_rate)
        self.n_iteration = int(n_iteration)
        self.weights = None
        self.bias = None
        self.is_fitted = False
        self.losses = []

    def _sigmoid(self, z):
        # piecewise to avoid overflow
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    def _as2d(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _as1d(self, y):
        y = np.asarray(y, dtype=float).ravel()
        return y

    def fit(self, X, y):
        X, y = self._as2d(X), self._as1d(y)
        n_samples, n_features = X.shape

        # init
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.losses = []

        for _ in range(self.n_iteration):
            # forward
            logits = X @ self.weights + self.bias
            probs = self._sigmoid(logits)

            # gradients for logistic loss (cross-entropy)
            error = probs - y
            dw = (X.T @ error) / n_samples
            db = np.sum(error) / n_samples

            # update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # track log-loss (with clipping for numerical safety)
            p = np.clip(probs, 1e-12, 1 - 1e-12)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            self.losses.append(loss)

        self.is_fitted = True
        return self

    @check_fitter
    def predict_proba(self, X):
        X = self._as2d(X)
        logits = X @ self.weights + self.bias
        proba_class_1 = self._sigmoid(logits)  # P(y=1)
        # Stack P(y=0) and P(y=1) horizontally
        return np.column_stack([1 - proba_class_1, proba_class_1])

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        # Use probability of class 1 for thresholding
        return (proba[:, 1] >= threshold).astype(int)

    def __repr__(self):
        return (
            f"LogisticRegression(learning_rate={self.learning_rate}, "
            f"n_iteration={self.n_iteration})"
        )
