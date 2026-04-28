"""
Unit tests for decision_tree.py.

Tests verify that the tree correctly learns from small datasets with
deterministic structure and that all stopping and splitting criteria behave
as documented.
"""

import numpy as np
import pytest
from mlpackage import DecisionTreeClassifier


# -------------------------------------------------------------------------
# Correctness
# -------------------------------------------------------------------------

def test_all_same_class_is_leaf():
    """
    When all labels are identical the tree should have depth 0 (a single leaf).
    """
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0])

    tree = DecisionTreeClassifier().fit(X, y)

    assert tree.get_depth() == 0


def test_two_class_perfect_split():
    """
    A linearly separable two-class problem must be classified without error.
    """
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = DecisionTreeClassifier(random_state=42).fit(X, y)
    pred = tree.predict(X)

    np.testing.assert_array_equal(pred, y)


def test_max_depth_respected():
    """
    Tree depth must not exceed max_depth regardless of dataset size.
    """
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.tile([0, 1], 10)

    tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)

    assert tree.get_depth() <= 2


def test_gini_criterion_produces_correct_predictions():
    """
    Gini splitting criterion must also achieve perfect accuracy on separable data.
    """
    X = np.array([[0.0], [1.0], [5.0], [6.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(criterion="gini", random_state=42).fit(X, y)

    np.testing.assert_array_equal(tree.predict(X), y)


def test_predict_proba_rows_sum_to_one():
    """
    predict_proba must produce non-negative values whose rows sum to 1.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(random_state=42).fit(X, y)
    proba = tree.predict_proba(X)

    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4), atol=1e-12)
    assert np.all(proba >= 0)


def test_feature_importances_sum_to_one():
    """
    feature_importances_ must be non-negative and sum to 1 when any split occurred.
    """
    X = np.array([[0.0, 1.0], [1.0, 0.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(random_state=42).fit(X, y)

    assert np.all(tree.feature_importances_ >= 0)
    np.testing.assert_allclose(tree.feature_importances_.sum(), 1.0, atol=1e-12)


def test_score_perfect_on_training_data():
    """score must return 1.0 on a problem the tree can solve exactly."""
    X = np.array([[0.0], [1.0], [8.0], [9.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTreeClassifier(random_state=42).fit(X, y)

    assert tree.score(X, y) == 1.0


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    tree = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        tree.predict(np.array([[1.0]]))


def test_invalid_criterion_raises():
    """Constructor must raise ValueError for an unrecognized criterion."""
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="bad_criterion")
