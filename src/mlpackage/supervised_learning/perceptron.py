"""
Perceptron

The classic Rosenblatt perceptron for binary classification. Weights are
updated online — one sample at a time — using the perceptron learning rule:
whenever a prediction is wrong, the weight vector is nudged in the direction
of the true label. No update is made for correctly classified samples.

Training halts when a full pass over the data produces no weight updates
(convergence), or when the maximum number of epochs is reached. Convergence
is only guaranteed when the classes are linearly separable.

An intercept term is maintained separately from the feature weights and is
updated by the same rule. Labels need not be {0, 1} — any two distinct values
are accepted and mapped internally.
"""

from __future__ import annotations

from typing import Union, Sequence

import numpy as np

ArrayLike = Union[np.ndarray, Sequence]

__all__ = ["Perceptron"]


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


def _as1d(y: ArrayLike, name: str = "y") -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    return arr


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Perceptron:
    """
    Binary perceptron classifier with online weight updates.

    Parameters
    ----------
    learning_rate : float
        Step size applied to each weight update. Values in (0, 1] are typical.
    n_epochs : int
        Maximum number of full passes over the training data.
    fit_intercept : bool
        Whether to maintain a bias term updated by the same perceptron rule.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 1.0,
        n_epochs: int = 1000,
        fit_intercept: bool = True,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {n_epochs}.")
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "Perceptron":
        """
        Fit the perceptron to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels. Must contain exactly two unique values; they are
            mapped internally to {-1, +1}.

        Returns
        -------
        self
        """
        X = _as2d_float(X, "X")
        y = _as1d(y, "y")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}."
            )
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError(
                f"Perceptron requires exactly 2 classes, "
                f"got {classes.shape[0]}: {classes}."
            )
        self.classes_ = classes
        # map to {-1, +1} for the update rule
        y_signed = np.where(y == classes[1], 1, -1).astype(float)

        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0

        for _ in range(self.n_epochs):
            n_errors = 0
            for xi, yi in zip(X, y_signed):
                prediction = self._predict_signed(xi)
                if prediction != yi:
                    self.coef_ += self.learning_rate * yi * xi
                    if self.fit_intercept:
                        self.intercept_ += self.learning_rate * yi
                    n_errors += 1
            if n_errors == 0:
                break

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return the raw linear score for each sample."""
        self._check_fitted()
        X = _as2d_float(X, "X")
        self._check_feature_count(X)
        return X @ self.coef_ + self.intercept_

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the predicted class label for each sample."""
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == _as1d(y, "y")))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict_signed(self, x: np.ndarray) -> float:
        return 1.0 if np.dot(self.coef_, x) + self.intercept_ >= 0 else -1.0

    def _check_fitted(self) -> None:
        if self.coef_ is None:
            raise RuntimeError("Call fit before using this method.")

    def _check_feature_count(self, X: np.ndarray) -> None:
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was "
                f"fitted on {self.coef_.shape[0]} features."
            )
