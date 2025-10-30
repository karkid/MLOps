import numpy as np
import pytest

from reml.spatial.distance import (
    euclidean,
    squared_euclidean,
    manhattan,
    canberra,
    chebyshev,
    minkowski,
    cosine_distance,
)


class TestDistanceMetricProperties:
    """Test fundamental properties of distance metrics."""
    
    def metrics_test_non_negativity(self, distance_function):
        """Test that distance is non-negative."""
        X1 = np.array([1.0, 2.0, 3.0])
        X2 = np.array([-4.0, -5.0, 6.0])
        dist = distance_function(X1, X2)
        assert dist >= 0, f"{distance_function.__name__} returned negative distance {dist}"

    def metrics_test_symmetry(self, distance_function):
        """Test that distance is symmetric."""
        X1 = np.array([1.0, 2.0, 3.0])
        X2 = np.array([-4.0, -5.0, 6.0])
        dist1 = distance_function(X1, X2)
        dist2 = distance_function(X2, X1)
        assert np.isclose(dist1, dist2), f"{distance_function.__name__} is not symmetric: d(X1, X2)={dist1}, d(X2, X1)={dist2}"

    def metrics_test_triangle_inequality(self, distance_function):
        """Test triangle inequality property."""
        X1 = np.array([1.0, 2.0, 3.0])
        X2 = np.array([-4.0, -5.0, 6.0])
        X3 = np.array([7.0, 8.0, 9.0])
        dist12 = distance_function(X1, X2)
        dist23 = distance_function(X2, X3)
        dist13 = distance_function(X1, X3)
        assert dist13 <= dist12 + dist23, f"{distance_function.__name__} does not satisfy triangle inequality: d(X1, X3)={dist13}, d(X1, X2)+d(X2, X3)={dist12 + dist23}"

    def metrics_test_identical_points(self, distance_function):
        """Test that distance between identical points is zero."""
        X = np.array([1.0, 2.0, 3.0])
        dist = distance_function(X, X)
        assert np.isclose(dist, 0.0), f"{distance_function.__name__} does not return zero distance for identical points: d(X, X)={dist}"

    def metrics_test_all(self, distance_function):
        """Run all metric property tests."""
        self.metrics_test_non_negativity(distance_function)
        self.metrics_test_symmetry(distance_function)
        self.metrics_test_triangle_inequality(distance_function)
        self.metrics_test_identical_points(distance_function)

    @pytest.mark.parametrize("distance_function", [
        euclidean,
        squared_euclidean,
        manhattan,
        canberra,
        chebyshev,
        lambda X1, X2: minkowski(X1, X2, p=3),
        cosine_distance,
    ])
    def test_distance_metric_properties(self, distance_function):
        """Test all distance metrics satisfy basic properties."""
        self.metrics_test_all(distance_function)


class TestEuclideanDistance:
    """Test Euclidean distance function."""
    
    def test_euclidean_basic(self):
        """Test basic Euclidean distance calculation."""
        X1 = np.array([0, 0])
        X2 = np.array([3, 4])
        dist = euclidean(X1, X2)
        assert np.isclose(dist, 5.0), f"Expected 5.0, got {dist}"
    
    def test_squared_euclidean_basic(self):
        """Test basic squared Euclidean distance calculation."""
        X1 = np.array([0, 0])
        X2 = np.array([3, 4])
        dist = squared_euclidean(X1, X2)
        assert np.isclose(dist, 25.0), f"Expected 25.0, got {dist}"


class TestManhattanDistance:
    """Test Manhattan distance function."""
    
    def test_manhattan_basic(self):
        """Test basic Manhattan distance calculation."""
        X1 = np.array([0, 0])
        X2 = np.array([3, 4])
        dist = manhattan(X1, X2)
        assert np.isclose(dist, 7.0), f"Expected 7.0, got {dist}"


class TestMinkowskiDistance:
    """Test Minkowski distance function."""
    
    def test_minkowski_invalid_p(self):
        """Test Minkowski distance with invalid p values."""
        X1 = np.array([1.0, 2.0, 3.0])
        X2 = np.array([4.0, 5.0, 6.0])
        with pytest.raises(ValueError):
            minkowski(X1, X2, p=0)

    def test_minkowski_negative_p(self):
        """Test Minkowski distance with negative p."""
        X1 = np.array([1.0, 2.0, 3.0])
        X2 = np.array([4.0, 5.0, 6.0])
        with pytest.raises(ValueError):
            minkowski(X1, X2, p=-2)

    def test_minkowski_equivalence(self):
        """Test Minkowski distance equivalence to Manhattan and Euclidean."""
        X1 = np.array([1.0, 2.0, 3.0])
        X2 = np.array([4.0, 5.0, 6.0])
        dist_p1 = minkowski(X1, X2, p=1)
        dist_p2 = minkowski(X1, X2, p=2)
        expected_manhattan = manhattan(X1, X2)
        expected_euclidean = euclidean(X1, X2)
        assert np.isclose(dist_p1, expected_manhattan), f"minkowski with p=1 did not match manhattan distance: got {dist_p1}, expected {expected_manhattan}"
        assert np.isclose(dist_p2, expected_euclidean), f"minkowski with p=2 did not match euclidean distance: got {dist_p2}, expected {expected_euclidean}"


class TestChebyshevDistance:
    """Test Chebyshev distance function."""
    
    def test_chebyshev_basic(self):
        """Test basic Chebyshev distance calculation."""
        X1 = np.array([1, 2, 3])
        X2 = np.array([4, 1, 6])
        dist = chebyshev(X1, X2)
        assert np.isclose(dist, 3.0), f"Expected 3.0, got {dist}"


class TestCanberraDistance:
    """Test Canberra distance function."""
    
    def test_canberra_with_zeros(self):
        """Test Canberra distance with zero vectors."""
        X1 = np.array([0.0, 0.0, 0.0])
        X2 = np.array([0.0, 0.0, 0.0])
        dist = canberra(X1, X2)
        assert np.isclose(dist, 0.0), f"canberra distance with zero vectors did not return 0.0, got {dist}"
    
    def test_canberra_basic(self):
        """Test basic Canberra distance calculation."""
        X1 = np.array([1.0, 2.0])
        X2 = np.array([3.0, 4.0])
        dist = canberra(X1, X2)
        expected = 2.0 / (1.0 + 3.0) + 2.0 / (2.0 + 4.0)  # |1-3|/(|1|+|3|) + |2-4|/(|2|+|4|)
        assert np.isclose(dist, expected), f"Expected {expected}, got {dist}"


class TestCosineDistance:
    """Test Cosine distance function."""
    
    def test_cosine_distance_zero_vector(self):
        """Test cosine distance with zero vector."""
        X1 = np.array([0.0, 0.0, 0.0])
        X2 = np.array([1.0, 2.0, 3.0])
        dist = cosine_distance(X1, X2)
        assert np.isclose(dist, 1.0), f"cosine_distance with zero vector did not return 1.0, got {dist}"

    def test_cosine_distance_identical_vectors(self):
        """Test cosine distance for identical vectors."""
        X = np.array([1.0, 2.0, 3.0])
        dist = cosine_distance(X, X)
        assert np.isclose(dist, 0.0), f"cosine_distance for identical vectors did not return 0.0, got {dist}"
    
    def test_cosine_distance_orthogonal_vectors(self):
        """Test cosine distance for orthogonal vectors."""
        X1 = np.array([1.0, 0.0])
        X2 = np.array([0.0, 1.0])
        dist = cosine_distance(X1, X2)
        assert np.isclose(dist, 1.0), f"cosine_distance for orthogonal vectors did not return 1.0, got {dist}"

