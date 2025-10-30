import numpy as np
import pytest

from reml.tree.RandomForest import RandomForest
from tests.utils import check_repr


def test_initialization():
    """Test DecisionTree initialization with default parameters."""
    rf = RandomForest()
    assert rf.min_samples_split == 2
    assert rf.max_depth == 20
    assert rf.n_features is None
    assert rf.n_trees == 10
    assert len(rf.trees) == 0

def test_initialization_with_params():
    """Test DecisionTree initialization with custom parameters."""
    rf = RandomForest(n_trees= 3, min_samples_split=5, max_depth=10, n_features=3)
    assert rf.min_samples_split == 5
    assert rf.max_depth == 10
    assert rf.n_features == 3
    assert rf.n_trees == 3
    assert len(rf.trees) == 0

def test_fit_simple_dataset():
    """Test fitting on a simple dataset."""
    X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y = np.array([0, 1, 1, 0])
    
    rf = RandomForest(max_depth=3)
    rf.fit(X, y)
    
    assert rf.is_fitted is True
    assert len(rf.trees) == 10

def test_predict_unfitted_raises_error():
    """Test that predicting on unfitted model raises an error."""
    dt = RandomForest()
    X = np.array([[1, 2]])
    
    with pytest.raises(ValueError, match="must be fitted"):
        dt.predict(X)

def test_repr():
    check_repr(RandomForest)