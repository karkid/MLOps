import numpy as np
import pytest

from reml.linear_model import LinearRegression

# --- Basic Fit Test ---
def test_fit_basic():
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

# --- Prediction Consistency ---
def test_predict():
    X = np.array([1, 2, 3])
    y = 2 * X + 1

    model = LinearRegression(learning_rate=0.01, n_iteration=1000)
    model.fit(X, y)

    preds = model.predict([6, 7])
    # Expect approximately 2*x + 1
    np.testing.assert_allclose(preds, [13, 15], atol=0.5)

# --- Shape Handling ---
def test_input_shape_handling():
    X_1d = np.array([1, 2, 3])
    X_2d = X_1d.reshape(-1, 1)
    y = np.array([2, 4, 6])

    model = LinearRegression().fit(X_1d, y)
    preds_1d = model.predict(X_1d)
    preds_2d = model.predict(X_2d)

    np.testing.assert_array_almost_equal(preds_1d, preds_2d)

# --- Check that predict fails before fitting ---
def test_predict_before_fit():
    model = LinearRegression()
    with pytest.raises(ValueError):
        model.predict([1, 2, 3])

# --- Check that predict fails before fitting ---
def test_repr_contains_params():
    """Check repr displays learning_rate and n_iteration."""
    model = LinearRegression(learning_rate=0.05, n_iteration=500)
    text = repr(model)
    assert "learning_rate=0.05" in text
    assert "n_iteration=500" in text


def test_multiple_features():
    """Test with multiple input features."""
    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([2, 4, 6])  # y = x1 + x2
    
    model = LinearRegression(learning_rate=0.01, n_iteration=1000)
    model.fit(X, y)
    
    pred = model.predict([[4, 4]])
    np.testing.assert_allclose(pred, [8], atol=0.5)


def test_learning_rate_impact():
    """Test impact of different learning rates."""
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    model_fast = LinearRegression(learning_rate=0.1, n_iteration=100)
    model_slow = LinearRegression(learning_rate=0.01, n_iteration=100)
    
    model_fast.fit(X, y)
    model_slow.fit(X, y)
    
    # Fast learning should converge more in fewer iterations
    assert model_fast.losses[10] < model_slow.losses[10]


def test_convergence():
    """Test that loss decreases during training."""
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    model = LinearRegression(learning_rate=0.01, n_iteration=100)
    model.fit(X, y)
    
    # Loss should decrease
    assert model.losses[-1] < model.losses[0]
