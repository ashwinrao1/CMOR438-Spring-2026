"""
Linear Regression

Supports ordinary least squares (OLS), ridge regularization, and gradient
descent optimization. All three solvers share the same interface: fit on
(X, y) pairs and predict on new feature matrices.

OLS minimizes the unpenalized squared loss via the normal equations, solved
with least-squares factorization for numerical stability. Ridge adds an L2
penalty on the feature weights to reduce variance at the cost of bias; the
intercept is never penalized. Gradient descent iteratively updates the weight
vector using the MSE gradient and is an alternative when direct matrix
inversion is undesirable.

An intercept term is prepended automatically when fit_intercept=True so
callers never need to add a bias column by hand. Evaluation metrics (R²,
MSE, RMSE, MAE) and raw residuals are available on any fitted model.
"""

from __future__ import annotations

from typing import Literal, Union, Sequence

import numpy as np

ArrayLike = Union[np.ndarray, Sequence]

__all__ = ["LinearRegression"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _as1d_float(y: ArrayLike, name: str = "y") -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    return arr


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LinearRegression:
    """
    Linear regression with OLS, Ridge, and gradient descent solvers.

    Parameters
    ----------
    solver : {'ols', 'ridge', 'gd'}
        Optimization strategy. 'ols' uses the normal equations via least-
        squares factorization. 'ridge' adds an L2 weight penalty. 'gd' runs
        batch gradient descent.
    alpha : float
        L2 regularization strength for solver='ridge'. Ignored otherwise.
        Must be >= 0.
    fit_intercept : bool
        Whether to fit a bias term. When True, a column of ones is prepended
        to X before solving; the intercept is never subject to regularization.
    learning_rate : float
        Step size for solver='gd'. Ignored otherwise.
    n_iterations : int
        Maximum gradient descent steps for solver='gd'. Ignored otherwise.
    tol : float or None
        Gradient norm convergence threshold for solver='gd'. Training stops
        early when the gradient norm falls below this value. None disables
        early stopping. Ignored for other solvers.
    """

    def __init__(
        self,
        solver: Literal["ols", "ridge", "gd"] = "ols",
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: float | None = 1e-6,
    ) -> None:
        if solver not in ("ols", "ridge", "gd"):
            raise ValueError(f"solver must be 'ols', 'ridge', or 'gd', got '{solver}'.")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}.")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}.")
        self.solver = solver
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.tol = tol
        self._weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """
        Fit the model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = _as2d_float(X, "X")
        y = _as1d_float(y, "y")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}."
            )
        X_aug = _add_intercept(X) if self.fit_intercept else X
        if self.solver == "ols":
            self._weights = self._fit_ols(X_aug, y)
        elif self.solver == "ridge":
            self._weights = self._fit_ridge(X_aug, y)
        else:
            self._weights = self._fit_gd(X_aug, y)
        return self

    def _fit_ols(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights, *_ = np.linalg.lstsq(X, y, rcond=None)
        return weights

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_cols = X.shape[1]
        penalty = np.eye(n_cols) * self.alpha
        if self.fit_intercept:
            # intercept is in column 0 — do not penalize it
            penalty[0, 0] = 0.0
        A = X.T @ X + penalty
        return np.linalg.solve(A, X.T @ y)

    def _fit_gd(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        weights = np.zeros(X.shape[1])
        for _ in range(self.n_iterations):
            grad = (2.0 / n_samples) * X.T @ (X @ weights - y)
            weights -= self.learning_rate * grad
            if self.tol is not None and np.linalg.norm(grad) < self.tol:
                break
        return weights

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return predicted target values for each row in X."""
        self._check_fitted()
        X = _as2d_float(X, "X")
        X_aug = _add_intercept(X) if self.fit_intercept else X
        if X_aug.shape[1] != self._weights.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was fitted on "
                f"{self._weights.shape[0] - int(self.fit_intercept)} features."
            )
        return X_aug @ self._weights

    def residuals(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Return y - y_hat for each sample."""
        return _as1d_float(y, "y") - self.predict(X)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the R² coefficient of determination on (X, y)."""
        return self.r2(X, y)

    def r2(self, X: ArrayLike, y: ArrayLike) -> float:
        """R² coefficient of determination."""
        y_true = _as1d_float(y, "y")
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        if ss_tot == 0.0:
            raise ValueError("R² is undefined when y is constant.")
        ss_res = (self.residuals(X, y_true) ** 2).sum()
        return float(1.0 - ss_res / ss_tot)

    def mse(self, X: ArrayLike, y: ArrayLike) -> float:
        """Mean squared error."""
        r = self.residuals(X, y)
        return float((r ** 2).mean())

    def rmse(self, X: ArrayLike, y: ArrayLike) -> float:
        """Root mean squared error."""
        return float(np.sqrt(self.mse(X, y)))

    def mae(self, X: ArrayLike, y: ArrayLike) -> float:
        """Mean absolute error."""
        return float(np.abs(self.residuals(X, y)).mean())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def coef_(self) -> np.ndarray:
        """Fitted feature weights, excluding the intercept."""
        self._check_fitted()
        return self._weights[1:] if self.fit_intercept else self._weights.copy()

    @property
    def intercept_(self) -> float:
        """Fitted intercept. Returns 0.0 when fit_intercept=False."""
        self._check_fitted()
        return float(self._weights[0]) if self.fit_intercept else 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._weights is None:
            raise RuntimeError("Call fit before using this method.")
