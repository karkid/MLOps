import numpy as np

def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(y_pred == y_true)/len(y_true)

def mean_square_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    square_error = (y_true - y_pred) **2
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
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    label_to_index = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_index[t], label_to_index[p]] += 1
    return cm
