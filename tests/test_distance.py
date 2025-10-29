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

def metrics_test_non_neagativity(distance_function):
    X1  = np.array([1.0, 2.0, 3.0])
    X2  = np.array([-4.0, -5.0, 6.0])
    dist = distance_function(X1, X2)
    assert dist >= 0, f"{distance_function.__name__} returned negative distance {dist}"

def metrics_test_symmetry(distance_function):
    X1 = np.array([1.0, 2.0, 3.0])
    X2 = np.array([-4.0, -5.0, 6.0])
    dist1 = distance_function(X1, X2)
    dist2 = distance_function(X2, X1)
    assert np.isclose(dist1, dist2), f"{distance_function.__name__} is not symmetric: d(X1, X2)={dist1}, d(X2, X1)={dist2}"

def metrics_test_triangle_inequality(distance_function):
    X1 = np.array([1.0, 2.0, 3.0])
    X2 = np.array([-4.0, -5.0, 6.0])
    X3 = np.array([7.0, 8.0, 9.0])
    dist12 = distance_function(X1, X2)
    dist23 = distance_function(X2, X3)
    dist13 = distance_function(X1, X3)
    assert dist13 <= dist12 + dist23, f"{distance_function.__name__} does not satisfy triangle inequality: d(X1, X3)={dist13}, d(X1, X2)+d(X2, X3)={dist12 + dist23}"

def metrics_test_identical_points(distance_function):
    X = np.array([1.0, 2.0, 3.0])
    dist = distance_function(X, X)
    assert np.isclose(dist, 0.0), f"{distance_function.__name__} does not return zero distance for identical points: d(X, X)={dist}"

def metrics_test(distance_function):
    metrics_test_non_neagativity(distance_function)
    metrics_test_symmetry(distance_function)
    metrics_test_triangle_inequality(distance_function)
    metrics_test_identical_points(distance_function)

@pytest.mark.parametrize("distance_function", [
    euclidean,
    squared_euclidean,
    manhattan,
    canberra,
    chebyshev,
    lambda X1, X2: minkowski(X1, X2, p=3),
    cosine_distance,
])

def test_distance_metrics(distance_function):
    metrics_test(distance_function)

def test_canberra_with_zeros():
    X1 = np.array([0.0, 0.0, 0.0])
    X2 = np.array([0.0, 0.0, 0.0])
    dist = canberra(X1, X2)
    assert np.isclose(dist, 0.0), f"canberra distance with zero vectors did not return 0.0, got {dist}"

def test_minkowski_invalid_p():
    X1 = np.array([1.0, 2.0, 3.0])
    X2 = np.array([4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        minkowski(X1, X2, p=0)

def test_minkowski_negative_p():
    X1 = np.array([1.0, 2.0, 3.0])
    X2 = np.array([4.0, 5.0, 6.0])
    with pytest.raises(ValueError):
        minkowski(X1, X2, p=-2)

def test_minkowski_p_equals_1_and_2():
    X1 = np.array([1.0, 2.0, 3.0])
    X2 = np.array([4.0, 5.0, 6.0])
    dist_p1 = minkowski(X1, X2, p=1)
    dist_p2 = minkowski(X1, X2, p=2)
    expected_manhattan = manhattan(X1, X2)
    expected_euclidean = euclidean(X1, X2)
    assert np.isclose(dist_p1, expected_manhattan), f"minkowski with p=1 did not match manhattan distance: got {dist_p1}, expected {expected_manhattan}"
    assert np.isclose(dist_p2, expected_euclidean), f"minkowski with p=2 did not match euclidean distance: got {dist_p2}, expected {expected_euclidean}"

def test_cosine_distance_zero_vector():
    X1 = np.array([0.0, 0.0, 0.0])
    X2 = np.array([1.0, 2.0, 3.0])
    dist = cosine_distance(X1, X2)
    assert np.isclose(dist, 1.0), f"cosine_distance with zero vector did not return 1.0, got {dist}"

def test_cosine_distance_identical_vectors():
    X = np.array([1.0, 2.0, 3.0])
    dist = cosine_distance(X, X)
    assert np.isclose(dist, 0.0), f"cosine_distance for identical vectors did not return 0.0, got {dist}"

