import numpy as np
import pytest

from reml.linear_model import LogisticRegression
from tests.utils import check_repr

# --- Basic Fit Test ---
def test_fit_basic():
    # Simple linear data: y = 2x + 1
    X = np.array([1, 2, 3, 4, 5])
    y = 2 * X + 1

    model = LogisticRegression(learning_rate=0.01, n_iteration=1000)
    model.fit(X, y)

    # Model should be fitted
    assert model.is_fitted
    assert model.weights is not None
    assert model.bias is not None
    assert len(model.losses) == 1000

# --- Prediction Consistency ---
def test_predict():
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 0, 1, 1])  # binary labels

    model = LogisticRegression(learning_rate=0.1, n_iteration=1000)
    model.fit(X, y)

    preds = model.predict([[1.5], [3.5]])
    assert preds.shape == (2,)
    assert set(preds).issubset({0, 1})


# --- Shape Handling ---
def test_input_shape_handling():
    X_1d = np.array([1, 2, 3])
    X_2d = X_1d.reshape(-1, 1)
    y = np.array([2, 4, 6])

    model = LogisticRegression().fit(X_1d, y)
    preds_1d = model.predict(X_1d)
    preds_2d = model.predict(X_2d)

    np.testing.assert_array_almost_equal(preds_1d, preds_2d)

# --- Check that predict fails before fitting ---
def test_predict_before_fit():
    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.predict([1, 2, 3])


def test_predict_proba():
    """Test probability predictions."""
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 0, 1, 1])
    
    model = LogisticRegression(learning_rate=0.1, n_iteration=1000)
    model.fit(X, y)
    
    proba = model.predict_proba([[2.0]])
    assert proba.shape == (1, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_almost_equal(np.sum(proba), 1.0)


def test_decision_boundary():
    """Test classification around decision boundary."""
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([0, 0, 0, 1, 1])
    
    model = LogisticRegression(learning_rate=0.1, n_iteration=1000)
    model.fit(X, y)
    
    # Points far from boundary should have high confidence
    proba_low = model.predict_proba([[0.0]])[0]
    proba_high = model.predict_proba([[4.0]])[0]
    
    assert proba_low[0] > 0.9  # High confidence for class 0
    assert proba_high[1] > 0.9  # High confidence for class 1


def test_repr():
    check_repr(LogisticRegression)
