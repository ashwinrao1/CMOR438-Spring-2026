"""
Unit tests for knn.py.

KNNClassifier and KNNRegressor are tested on small datasets where the
expected output can be derived by hand.
"""

import numpy as np
import pytest
from mlpackage import KNNClassifier, KNNRegressor


# =========================================================================
# KNNClassifier
# =========================================================================

def test_1nn_recovers_training_labels():
    """
    With k=1, every training point is its own nearest neighbor and must
    receive its own label back.
    """
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 1, 2, 3])

    clf = KNNClassifier(n_neighbors=1).fit(X, y)
    pred = clf.predict(X)

    np.testing.assert_array_equal(pred, y)


def test_majority_vote_selects_most_common_neighbor():
    """
    With k=3 and two neighbors of class 0 and one of class 1, the prediction
    must be class 0.
    """
    X_train = np.array([[0.0], [0.1], [0.2], [10.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.05]])

    clf = KNNClassifier(n_neighbors=3).fit(X_train, y_train)
    pred = clf.predict(X_test)

    assert pred[0] == 0


def test_predict_proba_shape_and_sums():
    """
    predict_proba must return shape (n_query, n_classes) with rows summing to 1.
    """
    X_train = np.array([[0.0], [1.0], [5.0], [6.0]])
    y_train = np.array([0, 0, 1, 1])

    clf = KNNClassifier(n_neighbors=2).fit(X_train, y_train)
    proba = clf.predict_proba(np.array([[0.5], [5.5]]))

    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(2), atol=1e-12)


def test_distance_weighting_classifier():
    """
    Distance weighting must give more influence to the closer neighbor.
    The query point 0.1 is much closer to class-0 neighbors than to the
    class-1 neighbor at 10.0, so the prediction must be class 0.
    """
    X_train = np.array([[0.0], [0.2], [10.0]])
    y_train = np.array([0, 0, 1])
    X_test = np.array([[0.1]])

    clf = KNNClassifier(n_neighbors=3, weights="distance").fit(X_train, y_train)
    pred = clf.predict(X_test)

    assert pred[0] == 0


def test_manhattan_metric_classifier():
    """KNNClassifier must work correctly with the Manhattan distance metric."""
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    y_train = np.array([0, 0, 1])

    clf = KNNClassifier(n_neighbors=2, metric="manhattan").fit(X_train, y_train)
    pred = clf.predict(np.array([[0.5, 0.5]]))

    assert pred[0] == 0


# -------------------------------------------------------------------------
# Error handling — classifier
# -------------------------------------------------------------------------

def test_classifier_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    clf = KNNClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(np.array([[1.0]]))


def test_invalid_n_neighbors_raises():
    """Constructor must raise ValueError when n_neighbors < 1."""
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=0)


def test_n_neighbors_exceeds_training_size_raises():
    """fit must raise ValueError when n_neighbors > n_training_samples."""
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=5).fit(X, y)


# =========================================================================
# KNNRegressor
# =========================================================================

def test_1nn_regressor_returns_exact_training_targets():
    """
    With k=1 the regressor must return the exact target for each training point.
    """
    X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_train = np.array([10.0, 20.0, 30.0, 40.0])

    reg = KNNRegressor(n_neighbors=1).fit(X_train, y_train)
    pred = reg.predict(X_train)

    np.testing.assert_allclose(pred, y_train, atol=1e-10)


def test_k2_regressor_averages_two_nearest():
    """
    With k=2 the prediction for a midpoint query must be the average of the
    two equidistant training targets.
    """
    X_train = np.array([[1.0], [3.0]])
    y_train = np.array([10.0, 20.0])
    X_test = np.array([[2.0]])  # equidistant from both

    reg = KNNRegressor(n_neighbors=2).fit(X_train, y_train)
    pred = reg.predict(X_test)

    np.testing.assert_allclose(pred, [15.0], atol=1e-10)


def test_distance_weighting_regressor():
    """
    With distance weighting, the closer neighbor must dominate the prediction.
    Query at 1.1 is very close to training point 1.0 (y=10) and far from 10.0
    (y=100), so the prediction must be much closer to 10 than to 100.
    """
    X_train = np.array([[1.0], [10.0]])
    y_train = np.array([10.0, 100.0])
    X_test = np.array([[1.1]])

    reg = KNNRegressor(n_neighbors=2, weights="distance").fit(X_train, y_train)
    pred = reg.predict(X_test)

    assert pred[0] < 20.0


def test_regressor_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    reg = KNNRegressor()
    with pytest.raises(RuntimeError):
        reg.predict(np.array([[1.0]]))
