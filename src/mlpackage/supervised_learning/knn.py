"""
k-Nearest Neighbors classification and regression.

Predictions are made by finding the k training examples closest to a query
point and aggregating their labels. No model parameters are learned during
training — the full dataset is stored and searched at prediction time
(brute-force). This makes kNN simple to understand but expensive at scale.

Classifiers aggregate neighbor labels by majority vote and expose class
probability estimates via predict_proba. Regressors average neighbor target
values. Both support distance-based weighting, where closer neighbors carry
more influence than distant ones.

Euclidean and Manhattan distances are both supported. The number of neighbors
and weighting scheme are configured at construction time and do not change
after fitting.
"""

from __future__ import annotations

from typing import Literal, Union, Sequence

import numpy as np

from ._utils import _as2d_float, _as1d

ArrayLike = Union[np.ndarray, Sequence]

__all__ = ["KNNClassifier", "KNNRegressor"]


def _pairwise_distances(XA: np.ndarray, XB: np.ndarray, metric: str) -> np.ndarray:
    """Return (n_query, n_train) distance matrix."""
    if metric == "euclidean":
        aa = np.sum(XA * XA, axis=1)[:, None]
        bb = np.sum(XB * XB, axis=1)[None, :]
        return np.sqrt(np.maximum(aa + bb - 2.0 * XA @ XB.T, 0.0))
    # manhattan
    return np.sum(np.abs(XA[:, None, :] - XB[None, :, :]), axis=2)


def _neighbor_weights(dist: np.ndarray, scheme: str, eps: float = 1e-12) -> np.ndarray:
    """
    Return per-neighbor weight matrix matching the shape of dist.

    Rows where any neighbor has distance <= eps are treated as exact ties:
    those neighbors each receive weight 1 and all others receive weight 0.
    """
    if scheme == "uniform":
        return np.ones_like(dist)
    zero_mask = dist <= eps
    row_has_zero = zero_mask.any(axis=1)[:, None]
    return np.where(row_has_zero, zero_mask.astype(float), 1.0 / np.maximum(dist, eps))


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _KNNBase:
    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError(f"metric must be 'euclidean' or 'manhattan', got '{metric}'.")
        if weights not in ("uniform", "distance"):
            raise ValueError(f"weights must be 'uniform' or 'distance', got '{weights}'.")
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.weights = weights
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "_KNNBase":
        X = _as2d_float(X, "X")
        y = _as1d(y, "y")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}."
            )
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) cannot exceed "
                f"the number of training samples ({X.shape[0]})."
            )
        self._X = X
        self._y = y
        return self

    def kneighbors(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the k nearest training neighbors for each query point.

        Returns
        -------
        distances : ndarray of shape (n_query, n_neighbors)
            Sorted distances to each neighbor, closest first.
        indices : ndarray of shape (n_query, n_neighbors)
            Corresponding indices into the training set.
        """
        if self._X is None:
            raise RuntimeError("Call fit before kneighbors.")
        X = _as2d_float(X, "X")
        if X.shape[1] != self._X.shape[1]:
            raise ValueError(
                f"Query has {X.shape[1]} features but the model was "
                f"trained on {self._X.shape[1]} features."
            )
        D = _pairwise_distances(X, self._X, self.metric)
        # partial sort: argpartition is O(n) vs argsort's O(n log n)
        part_idx = np.argpartition(D, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
        part_dist = np.take_along_axis(D, part_idx, axis=1)
        order = part_dist.argsort(axis=1)
        idx = np.take_along_axis(part_idx, order, axis=1)
        dist = np.take_along_axis(part_dist, order, axis=1)
        return dist, idx


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class KNNClassifier(_KNNBase):
    """
    Classifies each query point by aggregating the labels of its k nearest
    training neighbors.

    With weights='uniform', all neighbors vote equally (majority vote). With
    weights='distance', each neighbor's vote is weighted by the inverse of its
    distance so that closer neighbors exert more influence.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to consider. Must be >= 1.
    metric : {'euclidean', 'manhattan'}
        Distance metric used to locate neighbors.
    weights : {'uniform', 'distance'}
        Neighbor weighting scheme.

    Examples
    --------
    >>> import numpy as np
    >>> X_train = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0]])
    >>> y_train = np.array([0, 0, 1, 1])
    >>> clf = KNNClassifier(n_neighbors=2)
    >>> clf.fit(X_train, y_train).predict(np.array([[0.5, 0.5]]))
    array([0])
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNClassifier":
        super().fit(X, y)
        self.classes_: np.ndarray = np.unique(self._y)
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Return class probability estimates for each query point.

        Probabilities are computed as the normalized sum of neighbor weights
        per class, so they reflect both vote counts and distance weighting.

        Returns
        -------
        proba : ndarray of shape (n_query, n_classes)
            Columns correspond to classes in sorted order (self.classes_).
        """
        if self._X is None:
            raise RuntimeError("Call fit before predict_proba.")
        dist, idx = self.kneighbors(X)
        w = _neighbor_weights(dist, self.weights)           # (nq, k)
        neighbor_labels = self._y[idx]                      # (nq, k)
        class_idx = np.searchsorted(self.classes_, neighbor_labels)  # (nq, k)

        n_classes = len(self.classes_)
        # one-hot encode neighbor labels then weight-sum across neighbors
        one_hot = np.eye(n_classes)[class_idx]              # (nq, k, n_classes)
        proba = (one_hot * w[..., None]).sum(axis=1)        # (nq, n_classes)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the predicted class label for each row in X."""
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == _as1d(y)))


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------

class KNNRegressor(_KNNBase):
    """
    Predicts a continuous value for each query point by averaging the target
    values of its k nearest training neighbors.

    With weights='uniform', a simple mean is computed. With
    weights='distance', a distance-weighted mean is computed so that closer
    neighbors contribute more to the prediction.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to consider. Must be >= 1.
    metric : {'euclidean', 'manhattan'}
        Distance metric used to locate neighbors.
    weights : {'uniform', 'distance'}
        Neighbor weighting scheme.

    Examples
    --------
    >>> import numpy as np
    >>> X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> y_train = np.array([1.0, 2.0, 3.0, 4.0])
    >>> reg = KNNRegressor(n_neighbors=2)
    >>> reg.fit(X_train, y_train).predict(np.array([[2.5]]))
    array([2.5])
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNRegressor":
        # Cast y to float before handing off to the base class so that
        # integer targets (e.g., [1, 2, 3]) produce float predictions.
        super().fit(X, _as1d(y, "y").astype(float))
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return predicted target values for each row in X."""
        dist, idx = self.kneighbors(X)
        w = _neighbor_weights(dist, self.weights)
        y_neighbors = self._y[idx]
        return (w * y_neighbors).sum(axis=1) / w.sum(axis=1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return the R^2 coefficient of determination on (X, y).

        Raises ValueError if y is constant, as R^2 is undefined in that case.
        """
        y_true = _as1d(y).astype(float)
        y_pred = self.predict(X)
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        if ss_tot == 0:
            raise ValueError("R^2 is undefined when y is constant.")
        return float(1.0 - ((y_true - y_pred) ** 2).sum() / ss_tot)
