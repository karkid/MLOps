import numpy as np
import pytest

from reml.preprocessing import Normalizer
from tests.utils import check_repr


class TestNormalizerInitialization:
    """Test Normalizer initialization and parameter validation."""
    
    def test_default_initialization(self):
        """Test default initialization with L2 norm."""
        normalizer = Normalizer()
        assert normalizer.norm == "l2"
    
    def test_custom_norm_initialization(self):
        """Test initialization with custom norm."""
        normalizer = Normalizer(norm="l1")
        assert normalizer.norm == "l1"
    
    def test_invalid_norm(self):
        """Test validation of norm parameter."""
        with pytest.raises(ValueError):
            Normalizer(norm="invalid")
    
    def test_repr(self):
        """Test repr functionality."""
        check_repr(Normalizer)


class TestNormalizerL2Normalization:
    """Test Normalizer L2 (Euclidean) normalization."""
    
    def test_l2_basic(self):
        """Test basic L2 normalization."""
        normalizer = Normalizer(norm="l2")
        X = np.array([[3, 4], [6, 8]])
        X_normalized = normalizer.fit_transform(X)
        
        expected = np.array([[0.6, 0.8], [0.6, 0.8]])
        np.testing.assert_array_almost_equal(X_normalized, expected)
    
    def test_l2_unit_vectors(self):
        """Test that L2 normalized vectors have unit length."""
        normalizer = Normalizer(norm="l2")
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_normalized = normalizer.fit_transform(X)
        
        # Check that each row has L2 norm of 1
        norms = np.linalg.norm(X_normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1, 1])
    
    def test_l2_single_sample(self):
        """Test L2 normalization with single sample."""
        normalizer = Normalizer(norm="l2")
        X = np.array([[3, 4]])
        X_normalized = normalizer.fit_transform(X)
        
        expected = np.array([[0.6, 0.8]])
        np.testing.assert_array_almost_equal(X_normalized, expected)


class TestNormalizerL1Normalization:
    """Test Normalizer L1 (Manhattan) normalization."""
    
    def test_l1_basic(self):
        """Test basic L1 normalization."""
        normalizer = Normalizer(norm="l1")
        X = np.array([[3, 4], [6, 8]])
        X_normalized = normalizer.fit_transform(X)
        
        expected = np.array([[3/7, 4/7], [6/14, 8/14]])
        np.testing.assert_array_almost_equal(X_normalized, expected)
    
    def test_l1_sum_to_one(self):
        """Test that L1 normalized vectors sum to 1."""
        normalizer = Normalizer(norm="l1")
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_normalized = normalizer.fit_transform(X)
        
        # Check that each row sums to 1
        sums = np.sum(np.abs(X_normalized), axis=1)
        np.testing.assert_array_almost_equal(sums, [1, 1])


class TestNormalizerEdgeCases:
    """Test Normalizer edge cases and robustness."""
    
    def test_zero_vector_l2(self):
        """Test L2 normalization with zero vector."""
        normalizer = Normalizer(norm="l2")
        X = np.array([[0, 0]])
        
        with pytest.raises(ValueError):
            normalizer.fit_transform(X)
    
    def test_zero_vector_l1(self):
        """Test L1 normalization with zero vector."""
        normalizer = Normalizer(norm="l1")
        X = np.array([[0, 0]])
        
        with pytest.raises(ValueError):
            normalizer.fit_transform(X)
    
    def test_negative_values(self):
        """Test normalization with negative values."""
        normalizer = Normalizer(norm="l2")
        X = np.array([[-3, 4], [6, -8]])
        X_normalized = normalizer.fit_transform(X)
        
        # Check that normalized vectors have unit L2 norm
        norms = np.linalg.norm(X_normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1, 1])
    
    def test_mixed_zero_nonzero(self):
        """Test normalization with mix of zero and non-zero rows."""
        normalizer = Normalizer(norm="l2")
        X = np.array([[3, 4], [0, 0], [6, 8]])
        
        # Should raise error due to zero vector in the middle
        with pytest.raises(ValueError):
            normalizer.fit_transform(X)