import numpy as np
import pytest

from reml.neighbors.KNeighborsClassifier import KNeighborsClassifier

@pytest.fixture
def toy_data():
    # Two well-separated clusters in 2D
    X_train = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],   # class 0 cluster near origin
        [5.0, 5.0],
        [5.1, 5.0],
        [5.0, 5.1],   # class 1 cluster near (5,5)
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test_near_0 = np.array([
        [0.05, 0.02],
        [0.2, -0.1],
    ])
    X_test_near_1 = np.array([
        [5.05, 5.02],
        [4.9, 5.2],
    ])

    return X_train, y_train, X_test_near_0, X_test_near_1


def test_fit_returns_self(toy_data):
    X_train, y_train, *_ = toy_data
    clf = KNeighborsClassifier(k=3)
    out = clf.fit(X_train, y_train)
    assert out is clf


def test_predict_shape(toy_data):
    X_train, y_train, X0, X1 = toy_data
    clf = KNeighborsClassifier(k=3).fit(X_train, y_train)

    X_test = np.vstack([X0, X1])  # 4 samples
    y_pred = clf.predict(X_test)

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (X_test.shape[0],)


def test_k1_is_nearest_neighbor(toy_data):
    X_train, y_train, X0, X1 = toy_data
    clf = KNeighborsClassifier(k=1).fit(X_train, y_train)

    y_pred0 = clf.predict(X0)
    y_pred1 = clf.predict(X1)

    assert np.all(y_pred0 == 0)
    assert np.all(y_pred1 == 1)


def test_k3_majority_vote(toy_data):
    X_train, y_train, X0, X1 = toy_data
    clf = KNeighborsClassifier(k=3, weights="uniform").fit(X_train, y_train)

    y_pred0 = clf.predict(X0)
    y_pred1 = clf.predict(X1)

    # In both clusters, 3 nearest neighbors share the same label
    assert np.all(y_pred0 == 0)
    assert np.all(y_pred1 == 1)


def test_repr():
    clf = KNeighborsClassifier(k=7)
    assert repr(clf) == "KNeighborsClassifier(k=7)"

def test_invalid_k():
    with pytest.raises(ValueError):
        KNeighborsClassifier(k=0)
    with pytest.raises(ValueError):
        KNeighborsClassifier(k=-1)


def test_predict_proba():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    clf = KNeighborsClassifier(k=3)
    clf.fit(X, y)
    
    proba = clf.predict_proba(np.array([[2.5]]))
    np.testing.assert_array_almost_equal(proba, np.array([[0.33333333, 0.66666667]]))


def test_empty_training_data():
    clf = KNeighborsClassifier()
    with pytest.raises(ValueError):
        clf.fit(np.array([]), np.array([]))


def test_predict_single_class():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 1, 1])
    clf = KNeighborsClassifier()
    clf.fit(X, y)
    
    pred = clf.predict(np.array([[2]]))
    assert pred[0] == 1


def test_weights_distance():
    """Test distance-based weighting."""
    X = np.array([[1], [2], [10]])
    y = np.array([0, 0, 1])
    clf = KNeighborsClassifier(k=3, weights="distance")
    clf.fit(X, y)
    
    # Point at 1.9 should be classified as 0 due to two close neighbors
    pred = clf.predict([[1.9]])
    assert pred[0] == 0


def test_k_larger_than_samples():
    """Test behavior when k > n_samples."""
    X = np.array([[1], [2]])
    y = np.array([0, 1])
    clf = KNeighborsClassifier(k=3)
    clf.fit(X, y)
    
    pred = clf.predict([[1.5]])
    # Should still work, using all available samples
    assert pred.shape == (1,)


def test_multiclass():
    """Test with more than two classes."""
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 2, 2])
    clf = KNeighborsClassifier(k=2)
    clf.fit(X, y)
    
    proba = clf.predict_proba([[2.5]])
    assert proba.shape == (1, 3)  # Three classes
    np.testing.assert_array_almost_equal(np.sum(proba, axis=1), [1.0])
