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
