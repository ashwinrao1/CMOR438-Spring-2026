"""
Unit tests for perceptron.py.

Tests use small linearly separable problems where the perceptron is
guaranteed to converge to a zero-training-error solution.
"""

import numpy as np
import pytest
from mlpackage import Perceptron


# -------------------------------------------------------------------------
# Correctness
# -------------------------------------------------------------------------

def test_or_gate():
    """
    OR gate is linearly separable; the perceptron must achieve perfect accuracy.
    """
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 1])

    clf = Perceptron(n_epochs=100).fit(X, y)

    np.testing.assert_array_equal(clf.predict(X), y)


def test_and_gate():
    """
    AND gate is linearly separable; the perceptron must achieve perfect accuracy.
    """
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 0, 0, 1])

    clf = Perceptron(n_epochs=200).fit(X, y)

    np.testing.assert_array_equal(clf.predict(X), y)


def test_score_returns_one_on_separable_data():
    """
    Accuracy score must equal 1.0 on a linearly separable training set.
    """
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    y = np.array([0, 0, 1, 1])

    clf = Perceptron(n_epochs=100).fit(X, y)

    assert clf.score(X, y) == 1.0


def test_coef_shape_matches_n_features():
    """coef_ must be 1-D with length equal to the number of input features."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1, 1, 1])

    clf = Perceptron().fit(X, y)

    assert clf.coef_.shape == (2,)


def test_no_intercept_when_fit_intercept_false():
    """
    With fit_intercept=False, the intercept must remain 0.0 after fitting.
    """
    X = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    y = np.array([0, 0, 1, 1])

    clf = Perceptron(fit_intercept=False).fit(X, y)

    assert clf.intercept_ == 0.0


def test_decision_function_shape():
    """decision_function must return a 1-D array with one score per sample."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 1])

    clf = Perceptron().fit(X, y)
    scores = clf.decision_function(X)

    assert scores.shape == (3,)


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    clf = Perceptron()
    with pytest.raises(RuntimeError):
        clf.predict(np.array([[1.0]]))


def test_non_binary_labels_raise():
    """fit must raise ValueError when y contains more than two distinct values."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        Perceptron().fit(X, y)


def test_invalid_learning_rate_raises():
    """Constructor must raise ValueError when learning_rate <= 0."""
    with pytest.raises(ValueError):
        Perceptron(learning_rate=0.0)


def test_mismatched_X_y_shapes_raise():
    """fit must raise ValueError when X and y have different sample counts."""
    with pytest.raises(ValueError):
        Perceptron().fit(np.ones((3, 1)), np.ones(4))
