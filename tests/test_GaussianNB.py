import numpy as np
import pytest
from sklearn.datasets import make_classification, load_iris
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.model_selection import train_test_split

from reml.naive_bayes import GaussianNB
from tests.utils import check_repr


@pytest.fixture
def simple_binary_data():
    """Simple 2D binary classification dataset."""
    # Two well-separated clusters
    X = np.array([
        [1.0, 1.0],
        [1.5, 1.2],
        [1.2, 1.5],
        [1.1, 0.9],  # Class 0
        [4.0, 4.0],
        [4.2, 4.1],
        [3.8, 4.2],
        [4.1, 3.9],  # Class 1
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


@pytest.fixture
def multiclass_data():
    """Three-class classification dataset."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=150,
        n_features=4,
        n_classes=3,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y


@pytest.fixture
def iris_data():
    """Real iris dataset."""
    iris = load_iris()
    return iris.data, iris.target


class TestGaussianNBInitialization:
    """Test GaussianNB initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        nb = GaussianNB()
        assert nb._classes is None
        assert nb._mean is None
        assert nb._var is None
        assert nb._prior is None
        assert nb.is_fitted is False
    
    def test_repr(self):
        """Test string representation."""
        check_repr(GaussianNB)


class TestGaussianNBFitting:
    """Test GaussianNB fitting process."""
    
    def test_fit_returns_self(self, simple_binary_data):
        """Test that fit returns self."""
        X, y = simple_binary_data
        nb = GaussianNB()
        result = nb.fit(X, y)
        assert result is nb
    
    def test_fit_binary_data(self, simple_binary_data):
        """Test fitting on binary data."""
        X, y = simple_binary_data
        nb = GaussianNB()
        nb.fit(X, y)
        
        assert nb.is_fitted is True
        assert len(nb._classes) == 2
        assert np.array_equal(nb._classes, [0, 1])
        assert nb._mean.shape == (2, 2)  # 2 classes, 2 features
        assert nb._var.shape == (2, 2)
        assert nb._prior.shape == (2,)
        
        # Check priors sum to 1
        assert np.isclose(np.sum(nb._prior), 1.0)
    
    def test_fit_multiclass_data(self, multiclass_data):
        """Test fitting on multiclass data."""
        X, y = multiclass_data
        nb = GaussianNB()
        nb.fit(X, y)
        
        assert nb.is_fitted is True
        assert len(nb._classes) == 3
        assert nb._mean.shape == (3, X.shape[1])
        assert nb._var.shape == (3, X.shape[1])
        assert nb._prior.shape == (3,)
        assert np.isclose(np.sum(nb._prior), 1.0)
    
    def test_fit_single_feature(self):
        """Test fitting with single feature."""
        X = np.array([[1], [1], [2], [2], [3], [3]])
        y = np.array([0, 0, 1, 1, 2, 2])
        
        nb = GaussianNB()
        nb.fit(X, y)
        
        assert nb.is_fitted is True
        assert len(nb._classes) == 3
        assert nb._mean.shape == (3, 1)
        assert nb._var.shape == (3, 1)
    
    def test_fit_single_sample_per_class(self):
        """Test fitting with single sample per class."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 2])
        
        nb = GaussianNB()
        nb.fit(X, y)
        
        assert nb.is_fitted is True
        assert len(nb._classes) == 3
        # Variance should be zero (handled by epsilon in _pdf)
        assert np.all(nb._var == 0)


class TestGaussianNBPrediction:
    """Test GaussianNB prediction functionality."""
    
    def test_predict_binary_data(self, simple_binary_data):
        """Test prediction on binary data."""
        X, y = simple_binary_data
        nb = GaussianNB()
        nb.fit(X, y)
        
        # Test predictions
        predictions = nb.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should predict reasonably well on training data
        accuracy = np.mean(predictions == y)
        assert accuracy >= 0.7  # Allow some flexibility
    
    def test_predict_multiclass_data(self, multiclass_data):
        """Test prediction on multiclass data."""
        X, y = multiclass_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        
        predictions = nb.predict(X_test)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y_test)
        assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_predict_single_sample(self, simple_binary_data):
        """Test prediction on single sample."""
        X, y = simple_binary_data
        nb = GaussianNB()
        nb.fit(X, y)
        
        # Predict single sample
        single_prediction = nb.predict(X[[0]])
        assert isinstance(single_prediction, np.ndarray)
        assert len(single_prediction) == 1
        assert single_prediction[0] in [0, 1]
    
    def test_predict_unfitted_raises_error(self):
        """Test that predicting on unfitted model raises error."""
        nb = GaussianNB()
        X = np.array([[1, 2]])
        
        with pytest.raises(ValueError, match="must be fitted"):
            nb.predict(X)
    
    def test_predict_new_data_shape(self, simple_binary_data):
        """Test prediction on data with different number of samples."""
        X, y = simple_binary_data
        nb = GaussianNB()
        nb.fit(X, y)
        
        # Test on different sized arrays
        for n_samples in [1, 5, 10]:
            X_new = np.random.randn(n_samples, 2)
            predictions = nb.predict(X_new)
            assert len(predictions) == n_samples


class TestGaussianNBAccuracy:
    """Test GaussianNB accuracy against known results."""
    
    def test_iris_dataset_accuracy(self, iris_data):
        """Test accuracy on iris dataset."""
        X, y = iris_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Test our implementation
        nb_reml = GaussianNB()
        nb_reml.fit(X_train, y_train)
        predictions_reml = nb_reml.predict(X_test)
        accuracy_reml = np.mean(predictions_reml == y_test)
        
        # Compare with sklearn
        nb_sklearn = SklearnGaussianNB()
        nb_sklearn.fit(X_train, y_train)
        predictions_sklearn = nb_sklearn.predict(X_test)
        accuracy_sklearn = np.mean(predictions_sklearn == y_test)
        
        # Our implementation should be reasonably close to sklearn
        assert accuracy_reml >= 0.8  # Minimum acceptable accuracy
        assert abs(accuracy_reml - accuracy_sklearn) <= 0.1  # Close to sklearn
    
    def test_perfect_separation(self):
        """Test on perfectly separable data."""
        # Create perfectly separable data
        X = np.array([
            [0, 0], [0.1, 0.1], [0.2, 0.0],  # Class 0
            [10, 10], [10.1, 10.1], [10.2, 10.0]  # Class 1
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        nb = GaussianNB()
        nb.fit(X, y)
        
        # Should predict training data perfectly
        predictions = nb.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy == 1.0
    
    def test_comparison_with_sklearn(self, multiclass_data):
        """Compare predictions with sklearn on same data."""
        X, y = multiclass_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Fit both models
        nb_reml = GaussianNB()
        nb_reml.fit(X_train, y_train)
        
        nb_sklearn = SklearnGaussianNB()
        nb_sklearn.fit(X_train, y_train)
        
        # Get predictions
        pred_reml = nb_reml.predict(X_test)
        pred_sklearn = nb_sklearn.predict(X_test)
        
        # Calculate accuracies
        acc_reml = np.mean(pred_reml == y_test)
        acc_sklearn = np.mean(pred_sklearn == y_test)
        
        # Should be close to sklearn accuracy
        assert abs(acc_reml - acc_sklearn) <= 0.15


class TestGaussianNBEdgeCases:
    """Test GaussianNB edge cases and robustness."""
    
    def test_zero_variance_handling(self):
        """Test handling of zero variance features."""
        # Create data where one feature has zero variance
        X = np.array([
            [1, 5], [1, 5], [1, 5],  # Class 0, feature 0 has zero variance
            [2, 10], [2, 10], [2, 10]  # Class 1, feature 0 has zero variance
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        nb = GaussianNB()
        nb.fit(X, y)
        
        # Should still be able to predict
        predictions = nb.predict(X)
        assert len(predictions) == len(y)
        
        # Test prediction on new data
        X_new = np.array([[1, 5], [2, 10]])
        predictions_new = nb.predict(X_new)
        assert len(predictions_new) == 2
    
    def test_large_numbers(self):
        """Test with large numbers."""
        X = np.array([
            [1e6, 2e6], [1.1e6, 2.1e6],  # Class 0
            [5e6, 6e6], [5.1e6, 6.1e6]   # Class 1
        ])
        y = np.array([0, 0, 1, 1])
        
        nb = GaussianNB()
        nb.fit(X, y)
        
        predictions = nb.predict(X)
        assert len(predictions) == len(y)
    
    def test_negative_numbers(self):
        """Test with negative numbers."""
        X = np.array([
            [-1, -2], [-1.1, -2.1],  # Class 0
            [-5, -6], [-5.1, -6.1]   # Class 1
        ])
        y = np.array([0, 0, 1, 1])
        
        nb = GaussianNB()
        nb.fit(X, y)
        
        predictions = nb.predict(X)
        assert len(predictions) == len(y)
    
    def test_single_class(self):
        """Test with single class data."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 0])  # All same class
        
        nb = GaussianNB()
        nb.fit(X, y)
        
        # Should predict the only available class
        predictions = nb.predict(X)
        assert all(pred == 0 for pred in predictions)
        
        # Test on new data
        X_new = np.array([[10, 20]])
        pred_new = nb.predict(X_new)
        assert pred_new[0] == 0


class TestGaussianNBStatistics:
    """Test statistical properties of GaussianNB."""
    
    def test_computed_statistics(self, simple_binary_data):
        """Test that computed statistics are reasonable."""
        X, y = simple_binary_data
        nb = GaussianNB()
        nb.fit(X, y)
        
        # Check that means are computed correctly
        for class_idx, class_label in enumerate(nb._classes):
            class_mask = y == class_label
            X_class = X[class_mask]
            
            # Check mean calculation
            expected_mean = np.mean(X_class, axis=0)
            computed_mean = nb._mean[class_idx]
            np.testing.assert_array_almost_equal(computed_mean, expected_mean)
            
            # Check variance calculation
            expected_var = np.var(X_class, axis=0)
            computed_var = nb._var[class_idx]
            np.testing.assert_array_almost_equal(computed_var, expected_var)
            
            # Check prior calculation
            expected_prior = len(X_class) / len(X)
            computed_prior = nb._prior[class_idx]
            assert abs(computed_prior - expected_prior) < 1e-10
    
    def test_pdf_function(self, simple_binary_data):
        """Test the PDF function."""
        X, y = simple_binary_data
        nb = GaussianNB()
        nb.fit(X, y)
        
        # Test PDF for a sample point
        x = X[0]
        pdf_values = nb._pdf(0, x)
        
        # PDF values should be positive
        assert np.all(pdf_values > 0)
        
        # PDF should be an array with length equal to number of features
        assert len(pdf_values) == X.shape[1]


if __name__ == "__main__":
    pytest.main([__file__])