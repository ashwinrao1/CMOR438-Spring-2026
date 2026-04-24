"""
K-Means Clustering

This module provides a from-scratch implementation of K-Means using NumPy.

K-Means partitions n points into k clusters by iteratively assigning each
point to its nearest centroid and recomputing centroids as cluster means
(Lloyd's algorithm).

Key properties:
- Requires the number of clusters to be specified in advance
- Assumes spherical, similarly-sized clusters
- Sensitive to initialization and outliers
- Guaranteed to converge, but may find a local minimum

No sklearn dependencies are used.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

__all__ = ["KMeans"]


# ==========================================================
# Utility
# ==========================================================

def _validate_inputs(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return X


# ==========================================================
# KMeans Class
# ==========================================================

class KMeans:
    """
    K-Means clustering via Lloyd's algorithm.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    max_iter : int
        Maximum number of iterations of Lloyd's algorithm.
    tol : float
        Convergence threshold. Stops when the maximum centroid shift
        across all clusters falls below this value.
    random_state : int or None
        Seed for the random number generator used during initialization.
    init : ndarray of shape (n_clusters, n_features) or None
        User-supplied initial centroids. If None, centroids are chosen
        uniformly at random from the data points.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centroids after fitting.
    labels_ : ndarray of shape (n_samples,)
        Cluster index assigned to each training point.
    inertia_ : float
        Within-cluster sum of squared distances to the nearest centroid.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        init: Optional[np.ndarray] = None,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")
        if tol < 0:
            raise ValueError("tol must be non-negative.")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    # ======================================================
    # Public API
    # ======================================================

    def fit(self, X):
        """
        Fit K-Means clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self : KMeans

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[1,1],[1,2],[2,1],[9,9],[9,10],[10,9]])
        >>> km = KMeans(n_clusters=2, random_state=0)
        >>> km.fit(X)
        KMeans(n_clusters=2, random_state=0)
        >>> len(set(km.labels_))
        2
        """
        X = _validate_inputs(X)
        n_samples = X.shape[0]

        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} must be >= n_clusters={self.n_clusters}."
            )

        centroids = self._initialize_centroids(X)

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._recompute_centroids(X, labels, centroids)

            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids

            if shift < self.tol:
                break

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels, centroids)
        self.n_iter_ = iteration
        return self

    def predict(self, X):
        """Assign each point in X to the nearest centroid."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit before predict.")
        X = _validate_inputs(X)
        return self._assign_clusters(X, self.cluster_centers_)

    def fit_predict(self, X):
        """Fit K-Means and return cluster labels."""
        return self.fit(X).labels_

    # ======================================================
    # Internal helpers
    # ======================================================

    def _initialize_centroids(self, X):
        """Choose initial centroids from the data or from user-supplied values."""
        if self.init is not None:
            centroids = np.asarray(self.init, dtype=float)
            if centroids.shape != (self.n_clusters, X.shape[1]):
                raise ValueError(
                    f"init must have shape ({self.n_clusters}, {X.shape[1]}), "
                    f"got {centroids.shape}."
                )
            return centroids.copy()

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _assign_clusters(self, X, centroids):
        """Return the index of the nearest centroid for each point in X."""
        distances = self._pairwise_distances(X, centroids)
        return np.argmin(distances, axis=1)

    def _recompute_centroids(self, X, labels, old_centroids):
        """Compute new centroids as the mean of each cluster's points."""
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))

        for k in range(self.n_clusters):
            mask = labels == k
            if mask.any():
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                # Empty cluster: keep the old centroid in place.
                new_centroids[k] = old_centroids[k]

        return new_centroids

    def _compute_inertia(self, X, labels, centroids):
        """Sum of squared distances from each point to its assigned centroid."""
        diffs = X - centroids[labels]
        return float(np.sum(diffs * diffs))

    @staticmethod
    def _pairwise_distances(X, centroids):
        """Return (n_samples, n_clusters) Euclidean distance matrix."""
        # (n, 1, d) - (1, k, d) broadcasts to (n, k, d)
        diff = X[:, None, :] - centroids[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))
