import numpy as np
import pytest

from reml.preprocessing import Normalizer
from tests.utils import check_repr


def test_normalizer_l2():
    normalizer = Normalizer(norm="l2")
    X = np.array([[3, 4], [6, 8]])
    X_normalized = normalizer.fit_transform(X)
    
    expected = np.array([[0.6, 0.8], [0.6, 0.8]])
    np.testing.assert_array_almost_equal(X_normalized, expected)


def test_normalizer_l1():
    normalizer = Normalizer(norm="l1")
    X = np.array([[3, 4], [6, 8]])
    X_normalized = normalizer.fit_transform(X)
    
    expected = np.array([[3/7, 4/7], [6/14, 8/14]])
    np.testing.assert_array_almost_equal(X_normalized, expected)


def test_normalizer_invalid_norm():
    with pytest.raises(ValueError):
        Normalizer(norm="invalid")


def test_normalizer_zero_vector():
    normalizer = Normalizer()
    X = np.array([[0, 0]])
    
    with pytest.raises(ValueError):
        normalizer.fit_transform(X)

def test_repr():
    check_repr(Normalizer)