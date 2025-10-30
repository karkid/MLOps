import numpy as np
import pytest

from reml.preprocessing import MinMaxScaler
from tests.utils import check_repr


class TestMinMaxScalerInitialization:
    """Test MinMaxScaler initialization and parameter setting."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        scaler = MinMaxScaler()
        assert scaler.feature_range == (0, 1)
        assert scaler.feature_min == 0
        assert scaler.feature_max == 1
        assert not scaler.is_fitted
        assert scaler.min_ is None
        assert scaler.max_ is None
        assert scaler.scale_ is None
    
    def test_custom_feature_range(self):
        """Test initialization with custom feature range."""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        assert scaler.feature_range == (-1, 1)
        assert scaler.feature_min == -1
        assert scaler.feature_max == 1
    
    def test_repr(self):
        """Test repr functionality."""
        check_repr(MinMaxScaler)


class TestMinMaxScalerFitting:
    """Test MinMaxScaler fitting functionality."""
    
    def test_fit_basic(self):
        """Test basic fitting functionality."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X)
        
        np.testing.assert_array_almost_equal(scaler.min_, np.array([1, 2]))
        np.testing.assert_array_almost_equal(scaler.max_, np.array([5, 6]))
        assert scaler.is_fitted
    
    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]])
        result = scaler.fit(X)
        assert result is scaler
    
    def test_fit_single_feature(self):
        """Test fitting with single feature."""
        scaler = MinMaxScaler()
        X = np.array([[1], [3], [5]])
        scaler.fit(X)
        
        np.testing.assert_almost_equal(scaler.min_, 1.0)
        np.testing.assert_almost_equal(scaler.max_, 5.0)
    
    def test_fit_constant_feature(self):
        """Test fitting with constant feature values."""
        scaler = MinMaxScaler()
        X = np.array([[1, 5], [1, 5], [1, 5]])
        scaler.fit(X)
        
        np.testing.assert_array_almost_equal(scaler.min_, np.array([1, 5]))
        np.testing.assert_array_almost_equal(scaler.max_, np.array([1, 5]))


class TestMinMaxScalerTransformation:
    """Test MinMaxScaler transformation functionality."""
    
    def test_transform_basic(self):
        """Test basic transformation functionality."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        np.testing.assert_array_almost_equal(X_scaled, expected)
    
    def test_transform_before_fit(self):
        """Test that transform before fit raises error."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError):
            scaler.transform(X)
    
    def test_fit_transform(self):
        """Test combined fit and transform operation."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_scaled = scaler.fit_transform(X)
        
        expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        np.testing.assert_array_almost_equal(X_scaled, expected)
    
    def test_transform_new_data(self):
        """Test transformation of new data with fitted scaler."""
        scaler = MinMaxScaler()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X_train)
        
        X_new = np.array([[7, 8]])
        X_scaled = scaler.transform(X_new)
        
        expected = np.array([[1.5, 1.5]])  # Values beyond [0,1] range
        np.testing.assert_array_almost_equal(X_scaled, expected)
    
    def test_custom_feature_range_transform(self):
        """Test transformation with custom feature range."""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_scaled = scaler.fit_transform(X)
        
        expected = np.array([[-1, -1], [0, 0], [1, 1]])
        np.testing.assert_array_almost_equal(X_scaled, expected)


class TestMinMaxScalerEdgeCases:
    """Test MinMaxScaler edge cases and robustness."""
    
    def test_zero_range_handling(self):
        """Test handling of features with zero range (constant values)."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [1, 4], [1, 6]])  # First feature has zero range
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        # First column should have all zeros or same values due to division by zero handling
        # Second column should be scaled normally
        expected_col2 = np.array([0, 0.5, 1])
        np.testing.assert_array_almost_equal(X_scaled[:, 1], expected_col2)
    
    def test_single_sample_fitting(self):
        """Test fitting with single sample."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2]])
        scaler.fit(X)
        
        # With single sample, min and max should be the same
        np.testing.assert_array_almost_equal(scaler.min_, np.array([1, 2]))
        np.testing.assert_array_almost_equal(scaler.max_, np.array([1, 2]))
    
    def test_negative_values(self):
        """Test scaling with negative values."""
        scaler = MinMaxScaler()
        X = np.array([[-2, -1], [0, 0], [2, 1]])
        X_scaled = scaler.fit_transform(X)
        
        expected = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        np.testing.assert_array_almost_equal(X_scaled, expected)
    
    def test_empty_data_handling(self):
        """Test behavior with empty data."""
        scaler = MinMaxScaler()
        X = np.array([]).reshape(0, 2)
        
        # Empty array should raise an error during fitting
        with pytest.raises(ValueError):
            scaler.fit(X)