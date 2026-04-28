"""
Unit tests for linear_regression.py.

Tests verify correctness on small synthetic datasets where the true solution
is known analytically. All three solvers (OLS, Ridge, GD) are covered.
"""

import numpy as np
import pytest
from mlpackage import LinearRegression


# -------------------------------------------------------------------------
# OLS
# -------------------------------------------------------------------------

def test_ols_recovers_exact_coefficients():
    """
    OLS on y = 3x + 2 should recover the exact slope and intercept.
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = 3.0 * X.ravel() + 2.0

    model = LinearRegression(solver="ols")
    model.fit(X, y)

    assert abs(model.coef_[0] - 3.0) < 1e-8
    assert abs(model.intercept_ - 2.0) < 1e-8


def test_ols_predictions_match_true_values():
    """
    OLS predictions on training data should be exact for a linear signal.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([5.0, 7.0, 9.0])

    model = LinearRegression(solver="ols").fit(X, y)
    pred = model.predict(X)

    np.testing.assert_allclose(pred, y, atol=1e-8)


def test_ols_r2_is_one_on_training_data():
    """
    R² must equal 1.0 when OLS fits the training data exactly.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])

    model = LinearRegression(solver="ols").fit(X, y)

    assert abs(model.r2(X, y) - 1.0) < 1e-8


# -------------------------------------------------------------------------
# Ridge
# -------------------------------------------------------------------------

def test_ridge_predictions_close_to_true_signal():
    """
    Ridge with small alpha should still produce predictions close to y = 2x.
    """
    X = np.linspace(0, 5, 20).reshape(-1, 1)
    y = 2.0 * X.ravel()

    model = LinearRegression(solver="ridge", alpha=0.01).fit(X, y)
    pred = model.predict(X)

    np.testing.assert_allclose(pred, y, atol=0.2)


def test_ridge_intercept_not_penalized():
    """
    Ridge should still fit a non-zero intercept when the data calls for it.
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([10.0, 11.0, 12.0, 13.0])  # intercept = 9, slope = 1

    model = LinearRegression(solver="ridge", alpha=1e-6).fit(X, y)

    # With very small alpha the ridge solution is close to OLS.
    assert abs(model.intercept_ - 9.0) < 0.5


# -------------------------------------------------------------------------
# Gradient Descent
# -------------------------------------------------------------------------

def test_gd_converges_on_linear_data():
    """
    GD should converge close to the OLS solution on a clean linear dataset.
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = 2.0 * X.ravel() + 1.0

    model = LinearRegression(
        solver="gd",
        learning_rate=0.01,
        n_iterations=5000,
        tol=1e-8,
    ).fit(X, y)

    np.testing.assert_allclose(model.predict(X), y, atol=0.1)


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    model = LinearRegression()
    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0]]))


def test_invalid_solver_raises():
    """Constructor must raise ValueError for an unknown solver string."""
    with pytest.raises(ValueError):
        LinearRegression(solver="bad_solver")


def test_negative_alpha_raises():
    """Constructor must raise ValueError when alpha < 0."""
    with pytest.raises(ValueError):
        LinearRegression(solver="ridge", alpha=-1.0)


def test_mismatched_X_y_shapes_raises():
    """fit must raise ValueError when X and y have different sample counts."""
    model = LinearRegression()
    with pytest.raises(ValueError):
        model.fit(np.ones((3, 1)), np.ones(4))


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------

def test_mse_is_zero_on_perfect_fit():
    """MSE on the training set of a perfect fit must be 0."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])

    model = LinearRegression(solver="ols").fit(X, y)

    assert model.mse(X, y) < 1e-20


def test_residuals_sum_to_zero_for_ols_with_intercept():
    """OLS residuals must sum to (approximately) zero when an intercept is fitted."""
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.1, 3.9, 6.2, 7.8])

    model = LinearRegression(solver="ols", fit_intercept=True).fit(X, y)
    residuals = model.residuals(X, y)

    assert abs(residuals.sum()) < 1e-8


def test_fit_intercept_false_no_bias():
    """With fit_intercept=False, the intercept property must return 0.0."""
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 4.0, 6.0])

    model = LinearRegression(solver="ols", fit_intercept=False).fit(X, y)

    assert model.intercept_ == 0.0
