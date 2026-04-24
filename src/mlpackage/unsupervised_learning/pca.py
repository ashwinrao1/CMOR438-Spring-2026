"""
Principal Component Analysis (PCA)

This module provides a from-scratch implementation of PCA using NumPy.

PCA finds the directions of maximum variance in high-dimensional data and
projects the data onto a lower-dimensional subspace spanned by those directions.

Key properties:
- Linear dimensionality reduction
- Components are orthogonal and ordered by explained variance
- Requires centering the data before projection
- Sensitive to feature scale; standardize inputs when features have different units

No sklearn dependencies are used.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

__all__ = ["PCA"]


# ==========================================================
# Utility
# ==========================================================

def _validate_inputs(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return X


# ==========================================================
# PCA Class
# ==========================================================

class PCA:
    """
    Principal Component Analysis via eigendecomposition of the covariance matrix.

    Parameters
    ----------
    n_components : int or None
        Number of principal components to keep. If None, all components
        are retained.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space, ordered by descending explained variance.
        Each row is a unit eigenvector of the covariance matrix.
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each component (eigenvalues of the covariance matrix).
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total variance explained by each component.
    mean_ : ndarray of shape (n_features,)
        Per-feature means computed during fit, used to center data.
    n_components_ : int
        Actual number of components kept after fitting.
    """

    def __init__(self, n_components: Optional[int] = None):
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be >= 1.")

        self.n_components = n_components

        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None

    # ======================================================
    # Public API
    # ======================================================

    def fit(self, X):
        """
        Compute principal components from X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self : PCA

        Examples
        --------
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((100, 5))
        >>> pca = PCA(n_components=2)
        >>> X_reduced = pca.fit_transform(X)
        >>> X_reduced.shape
        (100, 2)
        >>> pca.explained_variance_ratio_.sum() <= 1.0
        True
        """
        X = _validate_inputs(X)
        n_samples, n_features = X.shape

        n_components = self._resolve_n_components(n_features)

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        cov = (X_centered.T @ X_centered) / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns eigenvalues in ascending order; reverse for descending.
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        self.components_ = eigenvectors[:, :n_components].T
        self.explained_variance_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / eigenvalues.sum()
        self.n_components_ = n_components
        return self

    def transform(self, X):
        """
        Project X onto the principal components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise RuntimeError("Call fit before transform.")
        X = _validate_inputs(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        """Fit PCA and return the projected data."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed):
        """
        Map projected data back to the original feature space.

        Parameters
        ----------
        X_transformed : ndarray of shape (n_samples, n_components)

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
        """
        if self.components_ is None:
            raise RuntimeError("Call fit before inverse_transform.")
        return X_transformed @ self.components_ + self.mean_

    # ======================================================
    # Internal helpers
    # ======================================================

    def _resolve_n_components(self, n_features):
        """Return the number of components to keep."""
        if self.n_components is None:
            return n_features
        if self.n_components > n_features:
            raise ValueError(
                f"n_components={self.n_components} must be <= n_features={n_features}."
            )
        return self.n_components
