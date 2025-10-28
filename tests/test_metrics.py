import numpy as np
import pytest

from reml.metrics import confusion_matrix


def test_confusion_matrix_binary():
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]
    cm = confusion_matrix(y_true, y_pred)
    expected = np.array([[3, 0], [1, 2]])
    np.testing.assert_array_equal(cm, expected)


def test_confusion_matrix_multiclass():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 1, 2]
    cm = confusion_matrix(y_true, y_pred)
    expected = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
    np.testing.assert_array_equal(cm, expected)


def test_confusion_matrix_string_labels():
    y_true = ["cat", "dog", "bird", "cat", "dog", "bird"]
    y_pred = ["cat", "dog", "dog", "cat", "dog", "bird"]
    cm = confusion_matrix(y_true, y_pred)
    expected = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
    np.testing.assert_array_equal(cm, expected)


def test_confusion_matrix_empty():
    with pytest.raises(ValueError):
        confusion_matrix([], [])