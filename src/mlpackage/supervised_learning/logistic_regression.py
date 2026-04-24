"""
Logistic Regression

Binary classifier that models the probability of the positive class using the
logistic function applied to a linear combination of input features. Parameters
are learned by minimizing binary cross-entropy loss via batch gradient descent.

L2 regularization penalizes the feature weights to reduce overfitting; the
regularization strength is controlled by alpha and the intercept is never
penalized. An intercept term is prepended automatically when fit_intercept=True.

After fitting, the model exposes predicted probabilities via predict_proba,
hard class predictions via predict, and the signed distance from the decision
boundary via decision_function. A manually computed ROC curve and AUC are also
available for threshold-free evaluation.
"""

from __future__ import annotations

from typing import Union, Sequence

import numpy as np

from ._utils import _as2d_float, _as1d, _sigmoid, _add_intercept

ArrayLike = Union[np.ndarray, Sequence]

__all__ = ["LogisticRegression"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LogisticRegression:
    """
    Binary logistic regression trained with batch gradient descent.

    Parameters
    ----------
    alpha : float
        L2 regularization strength. Must be >= 0. Set to 0.0 to disable
        regularization.
    fit_intercept : bool
        Whether to fit a bias term. The intercept is never regularized.
    learning_rate : float
        Gradient descent step size.
    n_iterations : int
        Maximum number of gradient descent steps.
    tol : float or None
        Gradient norm convergence threshold. Training stops early when the
        gradient norm falls below this value. None disables early stopping.
    threshold : float
        Decision threshold for converting probabilities to class labels.
        Defaults to 0.5.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        tol: float | None = 1e-6,
        threshold: float = 0.5,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}.")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}.")
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.tol = tol
        self.threshold = float(threshold)
        self._weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """
        Fit the classifier to binary training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels. Must contain exactly two unique values; they are
            mapped internally to {0, 1}.

        Returns
        -------
        self : LogisticRegression

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
        >>> y = np.array([0, 0, 1, 1])
        >>> clf = LogisticRegression(learning_rate=0.5, n_iterations=500)
        >>> clf.fit(X, y).predict(X)
        array([0, 0, 1, 1])
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
                f"LogisticRegression requires exactly 2 classes, "
                f"got {classes.shape[0]}: {classes}."
            )
        self.classes_ = classes
        y_bin = (y == classes[1]).astype(float)

        X_aug = _add_intercept(X) if self.fit_intercept else X
        n_samples, n_cols = X_aug.shape
        weights = np.zeros(n_cols)

        for _ in range(self.n_iterations):
            p = _sigmoid(X_aug @ weights)
            error = p - y_bin
            grad = X_aug.T @ error / n_samples
            if self.alpha > 0.0:
                reg = self.alpha * weights.copy()
                if self.fit_intercept:
                    reg[0] = 0.0
                grad += reg / n_samples
            weights -= self.learning_rate * grad
            if self.tol is not None and np.linalg.norm(grad) < self.tol:
                break

        self._weights = weights
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return the raw linear scores (log-odds) for each sample."""
        self._check_fitted()
        X = _as2d_float(X, "X")
        X_aug = _add_intercept(X) if self.fit_intercept else X
        self._check_feature_count(X_aug)
        return X_aug @ self._weights

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Return class probability estimates for each sample.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Columns are [P(negative class), P(positive class)].
        """
        p_pos = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p_pos, p_pos])

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the predicted class label for each sample."""
        p_pos = self.predict_proba(X)[:, 1]
        return self.classes_[(p_pos >= self.threshold).astype(int)]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == _as1d(y, "y")))

    def roc_curve(self, X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the ROC curve by sweeping all unique probability thresholds.

        Returns
        -------
        fpr : ndarray
            False positive rates, sorted by increasing threshold.
        tpr : ndarray
            True positive rates at each threshold.
        thresholds : ndarray
            Probability thresholds used to compute each (fpr, tpr) point.
        """
        y_true = _as1d(y, "y")
        y_score = self.predict_proba(X)[:, 1]
        classes = np.unique(y_true)
        if classes.shape[0] != 2:
            raise ValueError("roc_curve requires binary labels.")
        pos_label = self.classes_[1]
        y_bin = (y_true == pos_label).astype(int)

        thresholds = np.sort(np.unique(y_score))[::-1]
        fpr_list, tpr_list = [], []
        n_pos = y_bin.sum()
        n_neg = y_bin.shape[0] - n_pos

        for t in thresholds:
            y_pred = (y_score >= t).astype(int)
            tp = ((y_pred == 1) & (y_bin == 1)).sum()
            fp = ((y_pred == 1) & (y_bin == 0)).sum()
            tpr_list.append(tp / n_pos if n_pos > 0 else 0.0)
            fpr_list.append(fp / n_neg if n_neg > 0 else 0.0)

        # Prepend (0, 0): the "predict nothing positive" boundary point.
        # No need to append (1, 1) explicitly: at the minimum threshold every
        # sample is predicted positive, so the last loop point is already (1, 1).
        fpr = np.array([0.0] + fpr_list)
        tpr = np.array([0.0] + tpr_list)
        thresholds = np.concatenate([[np.inf], thresholds])
        return fpr, tpr, thresholds

    def auc(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the area under the ROC curve using the trapezoidal rule."""
        fpr, tpr, _ = self.roc_curve(X, y)
        # np.trapz is available in all supported NumPy versions; np.trapezoid
        # was added in NumPy 2.0 and is not backward-compatible.
        return float(np.trapz(tpr, fpr))

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

    def _check_feature_count(self, X_aug: np.ndarray) -> None:
        if X_aug.shape[1] != self._weights.shape[0]:
            n_fit = self._weights.shape[0] - int(self.fit_intercept)
            raise ValueError(
                f"X has {X_aug.shape[1] - int(self.fit_intercept)} features "
                f"but the model was fitted on {n_fit} features."
            )
