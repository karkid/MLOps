import numpy as np
import pytest

from reml.linear_model import LinearRegression
from tests.utils import check_repr


class TestLinearRegressionInitialization:
    """Test LinearRegression initialization and parameter setting."""
    
    def test_default_initialization(self):
        """Test default parameter initialization."""
        model = LinearRegression()
        assert model.learning_rate == 0.001
        assert model.n_iteration == 1000
        assert not model.is_fitted
        assert model.weights is None
        assert model.bias is None
        assert model.losses == []
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        model = LinearRegression(learning_rate=0.05, n_iteration=500)
        assert model.learning_rate == 0.05
        assert model.n_iteration == 500
    
    def test_repr_contains_params(self):
        """Check repr displays learning_rate and n_iteration."""
        model = LinearRegression(learning_rate=0.05, n_iteration=500)
        text = repr(model)
        assert "learning_rate=0.05" in text
        assert "n_iteration=500" in text
    
    def test_repr(self):
        """Test repr functionality."""
        check_repr(LinearRegression)


class TestLinearRegressionFitting:
    """Test LinearRegression fitting functionality."""
    
    def test_fit_basic(self):
        """Test basic fitting with simple linear data."""
        # Simple linear data: y = 2x + 1
        X = np.array([1, 2, 3, 4, 5])
        y = 2 * X + 1

        model = LinearRegression(learning_rate=0.01, n_iteration=1000)
        model.fit(X, y)

        # Model should be fitted
        assert model.is_fitted
        assert model.weights is not None
        assert model.bias is not None
        assert len(model.losses) == 1000
    
    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        X = np.array([1, 2, 3])
        y = np.array([2, 4, 6])
        model = LinearRegression()
        result = model.fit(X, y)
        assert result is model
    
    def test_convergence(self):
        """Test that loss decreases during training."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        
        model = LinearRegression(learning_rate=0.01, n_iteration=100)
        model.fit(X, y)
        
        # Loss should decrease
        assert model.losses[-1] < model.losses[0]
    
    def test_learning_rate_impact(self):
        """Test impact of different learning rates."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        
        model_fast = LinearRegression(learning_rate=0.1, n_iteration=100)
        model_slow = LinearRegression(learning_rate=0.01, n_iteration=100)
        
        model_fast.fit(X, y)
        model_slow.fit(X, y)
        
        # Fast learning should converge more in fewer iterations
        assert model_fast.losses[10] < model_slow.losses[10]


class TestLinearRegressionPrediction:
    """Test LinearRegression prediction functionality."""
    
    def test_predict_basic(self):
        """Test basic prediction functionality."""
        X = np.array([1, 2, 3])
        y = 2 * X + 1

        model = LinearRegression(learning_rate=0.01, n_iteration=1000)
        model.fit(X, y)

        preds = model.predict([6, 7])
        # Expect approximately 2*x + 1
        np.testing.assert_allclose(preds, [13, 15], atol=0.5)
    
    def test_predict_before_fit(self):
        """Test that predict fails before fitting."""
        model = LinearRegression()
        with pytest.raises(ValueError):
            model.predict([1, 2, 3])
    
    def test_input_shape_handling(self):
        """Test handling of different input shapes."""
        X_1d = np.array([1, 2, 3])
        X_2d = X_1d.reshape(-1, 1)
        y = np.array([2, 4, 6])

        model = LinearRegression().fit(X_1d, y)
        preds_1d = model.predict(X_1d)
        preds_2d = model.predict(X_2d)

        np.testing.assert_array_almost_equal(preds_1d, preds_2d)


class TestLinearRegressionFeatures:
    """Test LinearRegression with multiple features."""
class TestLinearRegressionFeatures:
    """Test LinearRegression with multiple features."""
    
    def test_multiple_features(self):
        """Test with multiple input features."""
        X = np.array([[1, 1], [2, 2], [3, 3]])
        y = np.array([2, 4, 6])  # y = x1 + x2
        
        model = LinearRegression(learning_rate=0.01, n_iteration=1000)
        model.fit(X, y)
        
        pred = model.predict([[4, 4]])
        np.testing.assert_allclose(pred, [8], atol=0.5)
    
    def test_feature_scaling_robustness(self):
        """Test robustness with different feature scales."""
        X = np.array([[0.1, 100], [0.2, 200], [0.3, 300]])
        y = np.array([1, 2, 3])
        
        model = LinearRegression(learning_rate=0.001, n_iteration=2000)
        model.fit(X, y)
        
        # Should be able to fit and predict
        assert model.is_fitted
        pred = model.predict([[0.4, 400]])
        assert len(pred) == 1
