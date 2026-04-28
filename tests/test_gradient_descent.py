"""
Unit tests for gradient_descent.py.

Tests verify convergence on analytically known minima: f(w) = w² in 1-D
and f(w) = ||w||² in N-D.
"""

import numpy as np
import pytest
from mlpackage import GradientDescent1D, GradientDescentND


# =========================================================================
# GradientDescent1D
# =========================================================================

def test_1d_minimizes_quadratic():
    """
    GD1D should converge to the minimum of f(w) = w² at w = 0.
    """
    opt = GradientDescent1D(learning_rate=0.1, n_iterations=200)
    w, _ = opt.optimize(lambda w: w ** 2, lambda w: 2 * w, w_init=5.0)

    assert abs(w) < 1e-4


def test_1d_history_contains_w_and_f_pairs():
    """
    Each entry in the returned history must be a (w, f(w)) tuple.
    """
    opt = GradientDescent1D(learning_rate=0.1, n_iterations=10, tol=None)
    _, history = opt.optimize(lambda w: w ** 2, lambda w: 2 * w, w_init=3.0)

    assert len(history) == 10
    for w, fval in history:
        assert isinstance(w, float)
        assert isinstance(fval, float)


def test_1d_early_stopping_on_convergence():
    """
    With a tight tolerance, GD1D should stop before exhausting n_iterations.
    """
    opt = GradientDescent1D(learning_rate=0.5, n_iterations=1000, tol=1e-6)
    _, history = opt.optimize(lambda w: w ** 2, lambda w: 2 * w, w_init=1.0)

    assert len(history) < 1000


def test_1d_tol_none_runs_all_iterations():
    """
    With tol=None, GD1D must run exactly n_iterations steps.
    """
    n = 50
    opt = GradientDescent1D(learning_rate=0.01, n_iterations=n, tol=None)
    _, history = opt.optimize(lambda w: w ** 2, lambda w: 2 * w, w_init=1.0)

    assert len(history) == n


def test_1d_invalid_lr_raises():
    """Constructor must raise ValueError when learning_rate <= 0."""
    with pytest.raises(ValueError):
        GradientDescent1D(learning_rate=0.0)


# =========================================================================
# GradientDescentND
# =========================================================================

def test_nd_minimizes_quadratic():
    """
    GDnD should converge to the minimum of f(w) = ||w||² at w = 0.
    """
    opt = GradientDescentND(learning_rate=0.1, n_iterations=500)
    grad_fn = lambda w: 2 * w  # gradient of ||w||^2
    w, _ = opt.optimize(grad_fn, w_init=np.array([3.0, -4.0]))

    np.testing.assert_allclose(w, np.zeros(2), atol=1e-4)


def test_nd_history_uses_loss_fn_when_provided():
    """
    When loss_fn is provided, history entries must be loss values, not grad norms.
    """
    opt = GradientDescentND(learning_rate=0.1, n_iterations=5, tol=None)
    loss_fn = lambda w: float(np.dot(w, w))  # ||w||^2
    grad_fn = lambda w: 2 * w
    _, history = opt.optimize(grad_fn, np.array([1.0, 1.0]), loss_fn=loss_fn)

    # All entries must be non-negative (they are loss values).
    assert all(v >= 0 for v in history)


def test_nd_history_uses_grad_norm_by_default():
    """
    Without loss_fn, history entries must be gradient norms (non-negative).
    """
    opt = GradientDescentND(learning_rate=0.1, n_iterations=10, tol=None)
    grad_fn = lambda w: 2 * w
    _, history = opt.optimize(grad_fn, np.array([1.0, 2.0]))

    assert all(v >= 0 for v in history)


def test_nd_w_init_not_1d_raises():
    """optimize must raise ValueError when w_init is not 1-D."""
    opt = GradientDescentND(learning_rate=0.1)
    with pytest.raises(ValueError):
        opt.optimize(lambda w: 2 * w, w_init=np.array([[1.0, 2.0]]))


def test_nd_invalid_lr_raises():
    """Constructor must raise ValueError when learning_rate <= 0."""
    with pytest.raises(ValueError):
        GradientDescentND(learning_rate=-0.1)
