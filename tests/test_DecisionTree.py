import numpy as np
import pytest

from reml.tree.DecisionTree import DecisionTree


class TestDecisionTree:
    def test_initialization(self):
        """Test DecisionTree initialization with default parameters."""
        dt = DecisionTree()
        assert dt.min_sample_split == 2
        assert dt.max_depth == 20
        assert dt.n_features is None
        assert dt.root is None

    def test_initialization_with_params(self):
        """Test DecisionTree initialization with custom parameters."""
        dt = DecisionTree(min_sample_split=5, max_depth=10, n_features=3)
        assert dt.min_sample_split == 5
        assert dt.max_depth == 10
        assert dt.n_features == 3

    def test_fit_simple_dataset(self):
        """Test fitting on a simple dataset."""
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        y = np.array([0, 1, 1, 0])
        
        dt = DecisionTree(max_depth=3)
        dt.fit(X, y)
        
        assert dt.is_fitted is True
        assert dt.root is not None

    def test_predict_simple_dataset(self):
        """Test prediction on a simple dataset."""
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        y = np.array([0, 1, 1, 0])
        
        dt = DecisionTree(max_depth=3)
        dt.fit(X, y)
        
        predictions = dt.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)

    def test_iris_like_dataset(self):
        """Test on a more complex dataset similar to iris."""
        np.random.seed(42)
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        
        dt = DecisionTree(max_depth=5)
        dt.fit(X, y)
        predictions = dt.predict(X)
        
        assert len(predictions) == len(y)
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_single_class_dataset(self):
        """Test on a dataset with only one class."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])
        
        dt = DecisionTree()
        dt.fit(X, y)
        predictions = dt.predict(X)
        
        assert all(pred == 1 for pred in predictions)

    def test_min_sample_split(self):
        """Test that min_sample_split is respected."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        
        dt = DecisionTree(min_sample_split=3)  # More than available samples
        dt.fit(X, y)
        
        # Should create a leaf node with most common class
        predictions = dt.predict(X)
        assert len(predictions) == 2

    def test_predict_unfitted_raises_error(self):
        """Test that predicting on unfitted model raises an error."""
        dt = DecisionTree()
        X = np.array([[1, 2]])
        
        with pytest.raises(ValueError, match="must be fitted"):
            dt.predict(X)

    def test_node_is_leaf(self):
        """Test Node.is_leaf() method."""
        from reml.tree.DecisionTree import Node
        
        leaf_node = Node(value=1)
        assert leaf_node.is_leaf()
        
        internal_node = Node(feature=0, threshold=0.5)
        assert not internal_node.is_leaf()