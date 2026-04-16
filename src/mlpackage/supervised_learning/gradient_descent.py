"""
Gradient Descent Optimizers

Implemented Optimizers
----------------------
GradientDescent1D
    Gradient descent for scalar-valued parameters w ∈ ℝ, given an
    explicit derivative df/dw.

GradientDescentND
    Gradient descent for vector-valued parameters w ∈ ℝⁿ, given a
    gradient function ∇f(w).
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np


# ------------------------------------------------------------------
# 1-D Gradient Descent
# ------------------------------------------------------------------

class GradientDescent1D:
    """
    Gradient descent for a scalar parameter w ∈ ℝ.

    Update rule:
        w_{t+1} = w_t - lr * df(w_t)

    Parameters
    ----------
    learning_rate : float
        Step size. Too large causes divergence; too small slows convergence.
    n_iterations : int
        Maximum number of update steps.
    tol : float or None
        Stop when |df(w)| < tol. None disables early stopping.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: Optional[float] = 1e-6,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol

    def optimize(
        self,
        f: Callable[[float], float],
        df: Callable[[float], float],
        w_init: float,
    ) -> tuple[float, list[tuple[float, float]]]:
        """
        Minimize f using its derivative df.

        Parameters
        ----------
        f : callable w -> scalar
            Objective function (used only for history tracking).
        df : callable w -> scalar
            Derivative of f with respect to w.
        w_init : float
            Starting value.

        Returns
        -------
        w : float
            Value of w at convergence or after n_iterations steps.
        history : list of (w, f(w)) tuples, one per iteration.
        """
        w = float(w_init)
        history: list[tuple[float, float]] = []

        for _ in range(self.n_iterations):
            grad = df(w)
            w -= self.learning_rate * grad
            history.append((w, f(w)))

            if self.tol is not None and abs(grad) < self.tol:
                break

        return w, history


# ------------------------------------------------------------------
# N-D Gradient Descent
# ------------------------------------------------------------------

class GradientDescentND:
    """
    Gradient descent for a vector parameter w ∈ ℝⁿ.

    Update rule:
        w_{t+1} = w_t - lr * ∇f(w_t)

    Parameters
    ----------
    learning_rate : float
        Step size applied uniformly to all parameters.
    n_iterations : int
        Maximum number of update steps.
    tol : float or None
        Stop when ||∇f(w)||_2 < tol. None disables early stopping.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: Optional[float] = 1e-6,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol

    def optimize(
        self,
        grad_fn: Callable[[np.ndarray], np.ndarray],
        w_init: np.ndarray,
        loss_fn: Optional[Callable[[np.ndarray], float]] = None,
    ) -> tuple[np.ndarray, list[float]]:
        """
        Minimize a function using its gradient.

        Parameters
        ----------
        grad_fn : callable w -> gradient array (same shape as w)
            Returns ∇f at the current parameter vector.
        w_init : 1-D array
            Initial parameter vector.
        loss_fn : optional callable w -> scalar
            If provided, its value is recorded in history each iteration.
            If None, history contains the gradient norm ||∇f(w)||_2 instead.

        Returns
        -------
        w : 1-D array
            Parameter vector at convergence or after n_iterations steps.
        history : list of floats, one per iteration.
        """
        w = np.array(w_init, dtype=float)
        if w.ndim != 1:
            raise ValueError("w_init must be a 1-D array.")

        history: list[float] = []

        for _ in range(self.n_iterations):
            grad = grad_fn(w)
            w -= self.learning_rate * grad

            grad_norm = float(np.linalg.norm(grad))
            history.append(float(loss_fn(w)) if loss_fn is not None else grad_norm)

            if self.tol is not None and grad_norm < self.tol:
                break

        return w, history
