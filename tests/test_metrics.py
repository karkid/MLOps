import numpy as np
import pytest

from reml.metrics import confusion_matrix


class TestConfusionMatrix:
    """Test confusion matrix functionality."""
    
    def test_binary_classification(self):
        """Test confusion matrix for binary classification."""
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 0, 1]
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[3, 0], [1, 2]])
        np.testing.assert_array_equal(cm, expected)
    
    def test_multiclass_classification(self):
        """Test confusion matrix for multiclass classification."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 1, 2]
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
        np.testing.assert_array_equal(cm, expected)
    
    def test_string_labels(self):
        """Test confusion matrix with string labels."""
        y_true = ["cat", "dog", "bird", "cat", "dog", "bird"]
        y_pred = ["cat", "dog", "dog", "cat", "dog", "bird"]
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[2, 0, 0], [0, 2, 0], [0, 1, 1]])
        np.testing.assert_array_equal(cm, expected)
    
    def test_perfect_classification(self):
        """Test confusion matrix with perfect predictions."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        np.testing.assert_array_equal(cm, expected)
    
    def test_single_class(self):
        """Test confusion matrix with single class."""
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[4]])
        np.testing.assert_array_equal(cm, expected)


class TestConfusionMatrixEdgeCases:
    """Test confusion matrix edge cases and error handling."""
    
    def test_empty_arrays(self):
        """Test confusion matrix with empty arrays."""
        with pytest.raises(ValueError):
            confusion_matrix([], [])
    
    def test_mismatched_lengths(self):
        """Test confusion matrix with mismatched array lengths."""
        with pytest.raises(ValueError):
            confusion_matrix([0, 1], [0, 1, 2])
    
    def test_single_prediction(self):
        """Test confusion matrix with single prediction."""
        y_true = [1]
        y_pred = [0]
        cm = confusion_matrix(y_true, y_pred)
        expected = np.array([[0, 1], [0, 0]])
        np.testing.assert_array_equal(cm, expected)
    
    def test_unseen_classes_in_predictions(self):
        """Test confusion matrix when predictions contain unseen classes."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 2, 1]  # Class 2 not in y_true
        cm = confusion_matrix(y_true, y_pred)
        
        # Should handle the extra class in predictions
        assert cm.shape[0] >= 2  # At least 2 classes from y_true
        assert cm.shape[1] >= 3  # At least 3 classes from y_pred