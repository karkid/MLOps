import numpy as np

from reml.utils.decorators import auto_repr, check_fitter


@auto_repr
class GaussianNB:
    def __init__(self):
        self._classes = None
        self._mean = None
        self._var = None
        self._prior = None
        self.is_fitted = False

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # calculate mean variance and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float32)
        self._var = np.zeros((n_classes, n_features), dtype=np.float32)
        self._prior = np.zeros((n_classes,), dtype=np.float32)
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = np.mean(X_c, axis=0)
            self._var[idx, :] = np.var(X_c, axis=0)
            self._prior[idx] = X_c.shape[0] / float(n_samples)

        self.is_fitted = True
        return self

    @check_fitter
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    @check_fitter
    def _predict(self, x):
        posteriors = []

        # calculate the posterior probability for each class
        for idx, _ in enumerate(self._classes):
            prior = np.log(self._prior[idx])
            posterior = np.log(self._pdf(idx, x)).sum()
            posterior = posterior + prior
            posteriors.append(posterior)

        # class with highest posterior
        return self._classes[np.argmax(posteriors)]

    @check_fitter
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        # Add a small value to variance to prevent division by zero or log(0)
        # , especially when features have zero variance.
        # The value 1e-10 is chosen as a trade-off: small enough
        # not to affect results, but large enough to ensure stability.
        epsilon = 1e-10
        var += epsilon
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator
