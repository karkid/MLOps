import numpy as np
import pytest

from reml.tree.DecisionTree import DecisionTree
from tests.utils import check_repr


class TestDecisionTreeInitialization:
    """Test DecisionTree initialization and parameter setting."""
    
    def test_default_initialization(self):
        """Test DecisionTree initialization with default parameters."""
        dt = DecisionTree()
        assert dt.min_samples_split == 2
        assert dt.max_depth == 20
        assert dt.n_features is None
        assert dt.root is None
    
    def test_custom_initialization(self):
        """Test DecisionTree initialization with custom parameters."""
        dt = DecisionTree(min_samples_split=5, max_depth=10, n_features=3)
        assert dt.min_samples_split == 5
        assert dt.max_depth == 10
        assert dt.n_features == 3
    
    def test_repr(self):
        """Test repr functionality."""
        check_repr(DecisionTree)


class TestDecisionTreeFitting:
    """Test DecisionTree fitting functionality."""
    
    def test_fit_simple_dataset(self):
        """Test fitting on a simple dataset."""
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        y = np.array([0, 1, 1, 0])
        
        dt = DecisionTree(max_depth=3)
        dt.fit(X, y)
        
        assert dt.is_fitted is True
        assert dt.root is not None
    
    def test_fit_returns_self(self):
        """Test that fit returns self for method chaining."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        dt = DecisionTree()
        result = dt.fit(X, y)
        assert result is dt
    
    def test_fit_single_class_dataset(self):
        """Test fitting on a dataset with only one class."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])
        
        dt = DecisionTree()
        dt.fit(X, y)
        assert dt.is_fitted
        assert dt.root is not None
    
    def test_min_samples_split_constraint(self):
        """Test that min_samples_split is respected during fitting."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        
        dt = DecisionTree(min_samples_split=3)  # More than available samples
        dt.fit(X, y)
        
        assert dt.is_fitted
        assert dt.root is not None


class TestDecisionTreePrediction:
    """Test DecisionTree prediction functionality."""
    
    def test_predict_simple_dataset(self):
        """Test prediction on a simple dataset."""
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        y = np.array([0, 1, 1, 0])
        
        dt = DecisionTree(max_depth=3)
        dt.fit(X, y)
        
        predictions = dt.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_before_fit(self):
        """Test that predicting on unfitted model raises an error."""
        dt = DecisionTree()
        X = np.array([[1, 2]])
        
        with pytest.raises(ValueError, match="must be fitted"):
            dt.predict(X)
    
    def test_predict_single_class(self):
        """Test prediction when all training samples have same class."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])
        
        dt = DecisionTree()
        dt.fit(X, y)
        predictions = dt.predict(X)
        
        assert all(pred == 1 for pred in predictions)
    
    def test_predict_with_min_samples_split(self):
        """Test prediction respects min_samples_split constraint."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])
        
        dt = DecisionTree(min_samples_split=3)  # More than available samples
        dt.fit(X, y)
        
        # Should create a leaf node with most common class
        predictions = dt.predict(X)
        assert len(predictions) == 2


class TestDecisionTreeComplexData:
    """Test DecisionTree with complex datasets."""
    
    def test_multiclass_dataset(self):
        """Test on a multiclass dataset similar to iris."""
        np.random.seed(42)
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100)
        
        dt = DecisionTree(max_depth=5)
        dt.fit(X, y)
        predictions = dt.predict(X)
        
        assert len(predictions) == len(y)
        assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_high_dimensional_data(self):
        """Test with higher dimensional data."""
        np.random.seed(42)
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)
        
        dt = DecisionTree(max_depth=8)
        dt.fit(X, y)
        predictions = dt.predict(X)
        
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)


class TestDecisionTreeInternalStructure:
    """Test DecisionTree internal structure and components."""
    
    def test_node_is_leaf(self):
        """Test Node.is_leaf() method."""
        from reml.tree.DecisionTree import Node
        
        leaf_node = Node(value=1)
        assert leaf_node.is_leaf()
        
        internal_node = Node(feature=0, threshold=0.5)
        assert not internal_node.is_leaf()
    
    def test_tree_structure_after_fit(self):
        """Test that tree structure is properly created after fitting."""
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        y = np.array([0, 1, 1, 0])
        
        dt = DecisionTree(max_depth=2)
        dt.fit(X, y)
        
        assert dt.root is not None
        # Root should not be a leaf for this dataset
        assert not dt.root.is_leaf()
    
    def test_depth_constraint(self):
        """Test that max_depth constraint is respected."""
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        
        dt = DecisionTree(max_depth=1)
        dt.fit(X, y)
        
        # With max_depth=1, we should have at most 2 levels
        assert dt.root is not None