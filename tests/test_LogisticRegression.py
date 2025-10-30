import numpy as np
import pytest

from reml.linear_model import LogisticRegression
from tests.utils import check_repr


class TestLogisticRegressionInitialization:
    """Test LogisticRegression initialization and parameter setting."""
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        model = LogisticRegression()
        assert model.learning_rate == 0.001
        assert model.n_iteration == 1000
        assert not model.is_fitted
        assert model.weights is None
        assert model.bias is None
        assert model.losses == []
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        model = LogisticRegression(learning_rate=0.05, n_iteration=500)
        assert model.learning_rate == 0.05
        assert model.n_iteration == 500
    
    def test_repr(self):
        """Test repr functionality."""
        check_repr(LogisticRegression)


class TestLogisticRegressionFitting:
    """Test LogisticRegression fitting functionality."""
    
    def test_fit_basic(self):
        """Test basic fitting with binary classification data."""
        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([0, 0, 0, 1, 1])  # binary labels

        model = LogisticRegression(learning_rate=0.1, n_iteration=1000)
        model.fit(X, y)

        # Model should be fitted
        assert model.is_fitted
        assert model.weights is not None
        assert model.bias is not None
        assert len(model.losses) == 1000
    
    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        X = np.array([[0], [1], [2]])
        y = np.array([0, 0, 1])
        model = LogisticRegression()
        result = model.fit(X, y)
        assert result is model
    
    def test_convergence(self):
        """Test that loss decreases during training."""
        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([0, 0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iteration=100)
        model.fit(X, y)
        
        # Loss should generally decrease (may have some fluctuation)
        assert model.losses[-10:][0] > model.losses[-1]  # Compare last 10 vs last


class TestLogisticRegressionPrediction:
    """Test LogisticRegression prediction functionality."""
    
    def test_predict_basic(self):
        """Test basic prediction functionality."""
        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([0, 0, 0, 1, 1])  # binary labels

        model = LogisticRegression(learning_rate=0.1, n_iteration=1000)
        model.fit(X, y)

        preds = model.predict([[1.5], [3.5]])
        assert preds.shape == (2,)
        assert set(preds).issubset({0, 1})
    
    def test_predict_before_fit(self):
        """Test that predict fails before fitting."""
        model = LogisticRegression()
        with pytest.raises(ValueError):
            model.predict([1, 2, 3])
    
    def test_input_shape_handling(self):
        """Test handling of different input shapes."""
        X_1d = np.array([1, 2, 3])
        X_2d = X_1d.reshape(-1, 1)
        y = np.array([0, 0, 1])

        model = LogisticRegression(learning_rate=0.1, n_iteration=500).fit(X_1d, y)
        preds_1d = model.predict(X_1d)
        preds_2d = model.predict(X_2d)

        np.testing.assert_array_almost_equal(preds_1d, preds_2d)


class TestLogisticRegressionProbabilities:
    """Test LogisticRegression probability prediction functionality."""
    
    def test_predict_proba(self):
        """Test probability predictions."""
        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([0, 0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iteration=1000)
        model.fit(X, y)
        
        proba = model.predict_proba([[2.0]])
        assert proba.shape == (1, 2)
        assert np.all((proba >= 0) & (proba <= 1))
        np.testing.assert_almost_equal(np.sum(proba), 1.0)
    
    def test_decision_boundary(self):
        """Test classification around decision boundary."""
        X = np.array([[0], [1], [2], [3], [4]])
        y = np.array([0, 0, 0, 1, 1])
        
        model = LogisticRegression(learning_rate=0.1, n_iteration=1000)
        model.fit(X, y)
        
        # Points far from boundary should have high confidence
        proba_low = model.predict_proba([[0.0]])[0]
        proba_high = model.predict_proba([[4.0]])[0]
        
        assert proba_low[0] > 0.7  # High confidence for class 0 (relaxed threshold)
        assert proba_high[1] > 0.7  # High confidence for class 1 (relaxed threshold)
    
    def test_proba_before_fit(self):
        """Test that predict_proba fails before fitting."""
        model = LogisticRegression()
        with pytest.raises(ValueError):
            model.predict_proba([[1, 2, 3]])
