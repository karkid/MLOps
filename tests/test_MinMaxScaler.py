import numpy as np
import pytest

from reml.preprocessing import MinMaxScaler


def test_minmax_scaler_fit():
    scaler = MinMaxScaler()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler.fit(X)
    
    np.testing.assert_array_almost_equal(scaler.min_, np.array([1, 2]))
    np.testing.assert_array_almost_equal(scaler.max_, np.array([5, 6]))


def test_minmax_scaler_transform():
    scaler = MinMaxScaler()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    
    expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    np.testing.assert_array_almost_equal(X_scaled, expected)


def test_minmax_scaler_fit_transform():
    scaler = MinMaxScaler()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_scaled = scaler.fit_transform(X)
    
    expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    np.testing.assert_array_almost_equal(X_scaled, expected)


def test_minmax_scaler_transform_before_fit():
    scaler = MinMaxScaler()
    X = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError):
        scaler.transform(X)