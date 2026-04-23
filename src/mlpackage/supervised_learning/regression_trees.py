"""
Regression Tree

A CART-style regression tree that recursively partitions the feature space
by finding the binary split that minimises mean squared error within each
resulting partition. Every internal node stores the feature index and
threshold of its split; every leaf stores the mean target value of the
training samples that reached it.

Splits are chosen by an exhaustive search over all features and all unique
midpoint thresholds derived from sorted feature values. The tree grows
top-down and stops when a node's depth reaches max_depth, when a node
contains fewer samples than min_samples_split, or when no split reduces
the MSE. No pruning is applied beyond these stopping conditions.
"""

from __future__ import annotations

from typing import Union, Sequence

import numpy as np

ArrayLike = Union[np.ndarray, Sequence]

__all__ = ["RegressionTree"]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(
        self,
        *,
        feature: int | None = None,
        threshold: float | None = None,
        left: "_Node | None" = None,
        right: "_Node | None" = None,
        value: float | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


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


def _mse(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.var(y))


def _best_split(
    X: np.ndarray,
    y: np.ndarray,
    min_samples_split: int,
) -> tuple[int | None, float | None, float]:
    """
    Search all features and thresholds for the split that minimises the
    weighted MSE of the two child partitions.

    Returns the best (feature_index, threshold, weighted_mse). If no
    beneficial split exists, feature_index and threshold are None.
    """
    n_samples, n_features = X.shape
    best_feature, best_threshold, best_mse = None, None, _mse(y)

    for feature in range(n_features):
        values = np.sort(np.unique(X[:, feature]))
        if values.size < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0

        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask
            n_left, n_right = left_mask.sum(), right_mask.sum()
            if n_left < min_samples_split or n_right < min_samples_split:
                continue

            weighted = (
                n_left * _mse(y[left_mask]) + n_right * _mse(y[right_mask])
            ) / n_samples

            if weighted < best_mse:
                best_mse = weighted
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_mse


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RegressionTree:
    """
    CART-style regression tree.

    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree. None allows unlimited growth, subject
        to the other stopping conditions.
    min_samples_split : int
        Minimum number of samples required in a node to attempt a split.
        Also applied as the minimum size of each child partition.
    """

    def __init__(
        self,
        *,
        max_depth: int | None = None,
        min_samples_split: int = 2,
    ) -> None:
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be >= 1 or None, got {max_depth}.")
        if min_samples_split < 2:
            raise ValueError(f"min_samples_split must be >= 2, got {min_samples_split}.")
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self._root: _Node | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RegressionTree":
        """
        Fit the regression tree to training data.

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
        self._root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or X.shape[0] < self.min_samples_split
        ):
            return _Node(value=float(y.mean()))

        feature, threshold, _ = _best_split(X, y, self.min_samples_split)
        if feature is None:
            return _Node(value=float(y.mean()))

        left_mask = X[:, feature] <= threshold
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._build(X[left_mask], y[left_mask], depth + 1),
            right=self._build(X[~left_mask], y[~left_mask], depth + 1),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return predicted target values for each row in X."""
        self._check_fitted()
        X = _as2d_float(X, "X")
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> float:
        node = self._root
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the R² coefficient of determination on (X, y)."""
        y_true = _as1d_float(y, "y")
        y_pred = self.predict(X)
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        if ss_tot == 0.0:
            raise ValueError("R² is undefined when y is constant.")
        return float(1.0 - ((y_true - y_pred) ** 2).sum() / ss_tot)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._root is None:
            raise RuntimeError("Call fit before using this method.")
