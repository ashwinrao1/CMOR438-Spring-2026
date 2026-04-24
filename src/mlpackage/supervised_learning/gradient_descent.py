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

__all__ = ["GradientDescent1D", "GradientDescentND"]


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
        f : callable
            Objective function ``w -> scalar``. Used only for history
            tracking; not evaluated during the gradient step itself.
        df : callable
            Derivative ``w -> scalar`` of f with respect to w.
        w_init : float
            Starting value.

        Returns
        -------
        w : float
            Value of w at convergence or after n_iterations steps.
        history : list of (float, float)
            One ``(w, f(w))`` tuple per iteration.

        Examples
        --------
        >>> opt = GradientDescent1D(learning_rate=0.1)
        >>> w, hist = opt.optimize(lambda w: w**2, lambda w: 2*w, w_init=5.0)
        >>> round(w, 4)
        0.0
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
        grad_fn : callable
            ``w -> ndarray`` — returns ∇f at the current parameter vector.
            Output must have the same shape as w.
        w_init : array-like of shape (n,)
            Initial parameter vector.
        loss_fn : callable or None
            ``w -> float``. If provided, its value is appended to history
            each iteration. If None, history contains ||∇f(w)||₂ instead.

        Returns
        -------
        w : ndarray of shape (n,)
            Parameter vector at convergence or after n_iterations steps.
        history : list of float
            One value per iteration: loss_fn(w) if provided, else grad norm.

        Examples
        --------
        >>> import numpy as np
        >>> opt = GradientDescentND(learning_rate=0.1)
        >>> grad_fn = lambda w: 2 * w          # gradient of ||w||^2
        >>> w, hist = opt.optimize(grad_fn, np.array([3.0, -4.0]))
        >>> np.allclose(w, 0.0, atol=1e-4)
        True
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
