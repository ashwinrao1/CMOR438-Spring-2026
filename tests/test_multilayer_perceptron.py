"""
Unit tests for multilayer_perceptron.py.

MLPClassifier and MLPRegressor are tested on simple synthetic problems.
All tests use random_state=42 for reproducibility and a small network to
keep runtime short.
"""

import numpy as np
import pytest
from mlpackage import MLPClassifier, MLPRegressor


# =========================================================================
# MLPClassifier
# =========================================================================

def test_classifier_learns_linearly_separable_data():
    """
    A small MLP should achieve perfect accuracy on well-separated binary data.
    """
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    clf = MLPClassifier(
        hidden_layer_sizes=(8,),
        learning_rate=0.05,
        n_iterations=2000,
        random_state=42,
    ).fit(X, y)

    assert clf.score(X, y) == 1.0


def test_classifier_predict_proba_shape():
    """predict_proba must return shape (n_samples, 2)."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1])

    clf = MLPClassifier(hidden_layer_sizes=(4,), n_iterations=500, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (3, 2)


def test_classifier_predict_proba_rows_sum_to_one():
    """Each row of predict_proba must sum to 1.0."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    clf = MLPClassifier(hidden_layer_sizes=(4,), n_iterations=500, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4), atol=1e-12)


def test_classifier_score_in_unit_interval():
    """score must always return a value in [0, 1]."""
    X = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    clf = MLPClassifier(hidden_layer_sizes=(4,), n_iterations=200, random_state=42)
    clf.fit(X, y)
    s = clf.score(X, y)

    assert 0.0 <= s <= 1.0


# -------------------------------------------------------------------------
# Error handling — classifier
# -------------------------------------------------------------------------

def test_classifier_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    clf = MLPClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(np.array([[1.0]]))


def test_classifier_non_binary_labels_raise():
    """fit must raise ValueError when y has more than two unique classes."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        MLPClassifier().fit(X, y)


def test_classifier_invalid_threshold_raises():
    """Constructor must raise ValueError when threshold is outside (0, 1)."""
    with pytest.raises(ValueError):
        MLPClassifier(threshold=1.0)


def test_classifier_negative_alpha_raises():
    """Constructor must raise ValueError when alpha < 0."""
    with pytest.raises(ValueError):
        MLPClassifier(alpha=-0.1)


# =========================================================================
# MLPRegressor
# =========================================================================

def test_regressor_approximates_linear_function():
    """
    An MLP regressor should learn a close approximation of y = 2x + 1.
    """
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = 2.0 * X.ravel() + 1.0

    reg = MLPRegressor(
        hidden_layer_sizes=(16,),
        learning_rate=0.02,
        n_iterations=3000,
        random_state=42,
    ).fit(X, y)

    assert reg.score(X, y) > 0.95


def test_regressor_predict_shape():
    """predict must return a 1-D array with one value per sample."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.0, 2.0])

    reg = MLPRegressor(hidden_layer_sizes=(4,), n_iterations=100, random_state=42)
    reg.fit(X, y)
    pred = reg.predict(X)

    assert pred.shape == (3,)


def test_regressor_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    reg = MLPRegressor()
    with pytest.raises(RuntimeError):
        reg.predict(np.array([[1.0]]))
