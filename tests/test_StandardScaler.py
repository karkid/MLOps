import numpy as np
import pytest

from reml.preprocessing import StandardScaler
from tests.utils import check_repr


class TestStandardScalerInitialization:
    """Test StandardScaler initialization and parameter setting."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        scaler = StandardScaler()
        assert not scaler.is_fitted
        assert scaler.mean_ is None
        assert scaler.std_ is None
    
    def test_repr(self):
        """Test repr functionality."""
        check_repr(StandardScaler)


class TestStandardScalerFitting:
    """Test StandardScaler fitting functionality."""
    
    def test_fit_basic(self):
        """Test basic fitting functionality."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X)
        
        np.testing.assert_array_almost_equal(scaler.mean_, np.array([3, 4]))
        np.testing.assert_array_almost_equal(scaler.std_, np.array([2, 2]))
        assert scaler.is_fitted
    
    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]])
        result = scaler.fit(X)
        assert result is scaler
    
    def test_fit_single_feature(self):
        """Test fitting with single feature."""
        scaler = StandardScaler()
        X = np.array([[1], [3], [5]])
        scaler.fit(X)
        
        np.testing.assert_almost_equal(scaler.mean_, 3.0)
        np.testing.assert_almost_equal(scaler.std_, 2.0)
    
    def test_fit_constant_feature(self):
        """Test fitting with constant feature values."""
        scaler = StandardScaler()
        X = np.array([[1, 5], [1, 5], [1, 5]])
        scaler.fit(X)
        
        np.testing.assert_array_almost_equal(scaler.mean_, np.array([1, 5]))
        np.testing.assert_array_almost_equal(scaler.std_, np.array([0, 0]))


class TestStandardScalerTransformation:
    """Test StandardScaler transformation functionality."""
    
    def test_transform_basic(self):
        """Test basic transformation functionality."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        expected = np.array([[-1, -1], [0, 0], [1, 1]])
        np.testing.assert_array_almost_equal(X_scaled, expected)
    
    def test_transform_before_fit(self):
        """Test that transform before fit raises error."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError):
            scaler.transform(X)
    
    def test_fit_transform(self):
        """Test combined fit and transform operation."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_scaled = scaler.fit_transform(X)
        
        expected = np.array([[-1, -1], [0, 0], [1, 1]])
        np.testing.assert_array_almost_equal(X_scaled, expected)
    
    def test_transform_new_data(self):
        """Test transformation of new data with fitted scaler."""
        scaler = StandardScaler()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X_train)
        
        X_new = np.array([[7, 8]])
        X_scaled = scaler.transform(X_new)
        
        expected = np.array([[2, 2]])  # (7-3)/2, (8-4)/2
        np.testing.assert_array_almost_equal(X_scaled, expected)


class TestStandardScalerEdgeCases:
    """Test StandardScaler edge cases and robustness."""
    
    def test_zero_variance_handling(self):
        """Test handling of features with zero variance."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [1, 4], [1, 6]])  # First feature has zero variance
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        # First column should remain unchanged due to zero std (handled by scale_)
        # Second column should be standardized normally
        expected_col2 = np.array([-1, 0, 1])
        np.testing.assert_array_almost_equal(X_scaled[:, 1], expected_col2)
    
    def test_single_sample_fitting(self):
        """Test fitting with single sample produces NaN for std."""
        scaler = StandardScaler()
        X = np.array([[1, 2]])
        scaler.fit(X)
        
        # With single sample and ddof=1, std should be NaN
        np.testing.assert_array_almost_equal(scaler.mean_, np.array([1, 2]))
        assert np.isnan(scaler.std_).all()  # std with ddof=1 on single sample gives NaN
    
    def test_two_sample_fitting(self):
        """Test fitting with two samples works correctly."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]])
        scaler.fit(X)
        
        np.testing.assert_array_almost_equal(scaler.mean_, np.array([2, 3]))
        np.testing.assert_array_almost_equal(scaler.std_, np.array([np.sqrt(2), np.sqrt(2)]))
    
    def test_empty_data_handling(self):
        """Test behavior with empty data."""
        scaler = StandardScaler()
        X = np.array([]).reshape(0, 2)
        scaler.fit(X)
        
        # Empty array should result in NaN values
        assert np.isnan(scaler.mean_).all()
        assert np.isnan(scaler.std_).all()