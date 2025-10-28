import numpy as np
import pytest

from reml.preprocessing import StandardScaler


def test_standard_scaler_fit():
    scaler = StandardScaler()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler.fit(X)
    
    np.testing.assert_array_almost_equal(scaler.mean_, np.array([3, 4]))
    np.testing.assert_array_almost_equal(scaler.std_, np.array([2, 2]))


def test_standard_scaler_transform():
    scaler = StandardScaler()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    expected = np.array([[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_array_almost_equal(X_scaled, expected)


def test_standard_scaler_fit_transform():
    scaler = StandardScaler()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_scaled = scaler.fit_transform(X)
    
    expected = np.array([[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_array_almost_equal(X_scaled, expected)


def test_standard_scaler_transform_before_fit():
    scaler = StandardScaler()
    X = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError):
        scaler.transform(X)