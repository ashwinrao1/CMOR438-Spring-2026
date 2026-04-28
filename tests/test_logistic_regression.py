"""
Unit tests for logistic_regression.py.

Tests use small synthetic datasets where correct binary classification
is deterministic and the decision boundary is clear.
"""

import numpy as np
import pytest
from mlpackage import LogisticRegression


# -------------------------------------------------------------------------
# Correctness
# -------------------------------------------------------------------------

def test_linearly_separable_binary_classification():
    """
    Perfect classification should be achievable on well-separated 1-D data.
    """
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = LogisticRegression(learning_rate=0.5, n_iterations=1000).fit(X, y)
    pred = clf.predict(X)

    np.testing.assert_array_equal(pred, y)


def test_predict_proba_rows_sum_to_one():
    """
    predict_proba must return probabilities whose rows each sum to 1.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    clf = LogisticRegression(learning_rate=0.5, n_iterations=500).fit(X, y)
    proba = clf.predict_proba(X)

    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-12)


def test_predict_proba_shape():
    """predict_proba must return shape (n_samples, 2) for binary classification."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])

    clf = LogisticRegression(learning_rate=0.5, n_iterations=300).fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (3, 2)


def test_score_returns_float_in_unit_interval():
    """score must return a float between 0 and 1."""
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    clf = LogisticRegression(learning_rate=0.5, n_iterations=500).fit(X, y)
    s = clf.score(X, y)

    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_coef_and_intercept_shapes():
    """coef_ must be 1-D with length n_features; intercept_ must be a scalar."""
    X = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    y = np.array([0, 0, 1, 1])

    clf = LogisticRegression(learning_rate=0.1, n_iterations=500).fit(X, y)

    assert clf.coef_.shape == (2,)
    assert isinstance(clf.intercept_, float)


def test_auc_is_between_zero_and_one():
    """AUC on separable data must be between 0 and 1."""
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    clf = LogisticRegression(learning_rate=0.5, n_iterations=500).fit(X, y)
    auc = clf.auc(X, y)

    assert 0.0 <= auc <= 1.0


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    clf = LogisticRegression()
    with pytest.raises(RuntimeError):
        clf.predict(np.array([[1.0]]))


def test_negative_alpha_raises():
    """Constructor must raise ValueError for alpha < 0."""
    with pytest.raises(ValueError):
        LogisticRegression(alpha=-0.1)


def test_threshold_out_of_range_raises():
    """Constructor must raise ValueError when threshold is outside (0, 1)."""
    with pytest.raises(ValueError):
        LogisticRegression(threshold=1.5)


def test_non_binary_labels_raise():
    """fit must raise ValueError when y contains more than two classes."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 2])

    clf = LogisticRegression()
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_mismatched_X_y_shapes_raise():
    """fit must raise ValueError when X and y have different sample counts."""
    clf = LogisticRegression()
    with pytest.raises(ValueError):
        clf.fit(np.ones((4, 1)), np.ones(3))
