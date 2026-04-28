"""
Unit tests for regression_trees.py.

Tests verify that the RegressionTree produces leaf predictions that match
known averages and that depth-limiting and error handling work correctly.
"""

import numpy as np
import pytest
from mlpackage import RegressionTree


# -------------------------------------------------------------------------
# Correctness
# -------------------------------------------------------------------------

def test_step_function_split():
    """
    A step function in 1-D should be captured exactly by a single split:
    one leaf predicts 0, the other predicts 10.
    """
    X = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    y = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])

    tree = RegressionTree(max_depth=1).fit(X, y)
    pred = tree.predict(X)

    np.testing.assert_allclose(pred, y, atol=1e-10)


def test_single_sample_group_predicts_mean():
    """
    When min_samples_split prevents any further splitting, a leaf must predict
    the mean of all samples that reached it.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 5.0, 9.0])

    tree = RegressionTree(min_samples_split=10).fit(X, y)  # no split possible
    pred = tree.predict(X)

    expected_mean = y.mean()
    np.testing.assert_allclose(pred, np.full(3, expected_mean), atol=1e-10)


def test_max_depth_limits_splits():
    """
    With max_depth=1 the tree can make at most one split, so the number of
    distinct predicted values must be at most 2.
    """
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 10, 10, 10, 10, 10], dtype=float)

    tree = RegressionTree(max_depth=1).fit(X, y)
    pred = tree.predict(X)

    assert len(np.unique(pred)) <= 2


def test_perfect_r2_on_step_function():
    """
    R² must be 1.0 when the tree can represent the target function exactly.
    """
    X = np.array([[0.0], [1.0], [5.0], [6.0]])
    y = np.array([0.0, 0.0, 100.0, 100.0])

    tree = RegressionTree(max_depth=1).fit(X, y)

    np.testing.assert_allclose(tree.score(X, y), 1.0, atol=1e-10)


def test_predictions_are_leaf_means():
    """
    For a two-group dataset, leaf predictions must equal the group means.
    """
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y = np.array([1.0, 2.0, 3.0, 7.0, 8.0, 9.0])

    tree = RegressionTree(max_depth=1).fit(X, y)
    pred = tree.predict(X)

    left_mean = y[:3].mean()  # 2.0
    right_mean = y[3:].mean()  # 8.0

    np.testing.assert_allclose(pred[:3], left_mean, atol=1e-10)
    np.testing.assert_allclose(pred[3:], right_mean, atol=1e-10)


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    tree = RegressionTree()
    with pytest.raises(RuntimeError):
        tree.predict(np.array([[1.0]]))


def test_invalid_max_depth_raises():
    """Constructor must raise ValueError when max_depth < 1."""
    with pytest.raises(ValueError):
        RegressionTree(max_depth=0)


def test_invalid_min_samples_split_raises():
    """Constructor must raise ValueError when min_samples_split < 2."""
    with pytest.raises(ValueError):
        RegressionTree(min_samples_split=1)


def test_mismatched_X_y_shapes_raise():
    """fit must raise ValueError when X and y have different sample counts."""
    tree = RegressionTree()
    with pytest.raises(ValueError):
        tree.fit(np.ones((3, 1)), np.ones(4))
