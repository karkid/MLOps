import numpy as np


def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(y_pred == y_true) / len(y_true)


def mean_square_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    square_error = (y_true - y_pred) ** 2
    return np.mean(square_error)


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    absolute_error = np.abs(y_true - y_pred)
    return np.mean(absolute_error)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.

    Returns
    -------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates the
        number of samples with true label being i-th class and predicted
        label being j-th class.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Arrays cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("Arrays must have equal length")

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Use ordered set to maintain order of first appearance
    labels = []
    for label in np.concatenate([y_true, y_pred]):
        if label not in labels:
            labels.append(label)
    n_labels = len(labels)

    label_to_index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for t, p in zip(y_true, y_pred, strict=True):
        i, j = label_to_index[t], label_to_index[p]
        cm[i, j] += 1

    return cm
