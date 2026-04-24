"""
Preprocessing

Common transformations applied to raw data before training a model.
All operations are implemented from scratch using NumPy.

Classes
-------
StandardScaler
    Zero mean, unit variance scaling (z-score normalization).
MinMaxScaler
    Scale features to a fixed range, default [0, 1].
LabelEncoder
    Encode a categorical label column as integers 0, 1, 2, ...
OneHotEncoder
    Encode integer class labels as binary indicator rows.

Functions
---------
train_test_split
    Split arrays into random train and test subsets.

No sklearn dependencies are used.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

__all__ = [
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",
    "OneHotEncoder",
    "train_test_split",
]


# ==========================================================
# Utility
# ==========================================================

def _validate_2d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    return X


def _validate_1d(y):
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    return y


# ==========================================================
# StandardScaler
# ==========================================================

class StandardScaler:
    """
    Standardize features to zero mean and unit variance.

    For each feature j:
        X_scaled[:, j] = (X[:, j] - mean_j) / std_j

    Features with zero variance are left unchanged (divided by 1).

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
    scale_ : ndarray of shape (n_features,)
        Standard deviation per feature. Zero-variance features have scale_ = 1.
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X):
        """Compute mean and std from X."""
        X = _validate_2d(X)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        """Standardize X using the fitted mean and std."""
        if self.mean_ is None:
            raise RuntimeError("Call fit before transform.")
        return (_validate_2d(X) - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fit and standardize X."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Reverse standardization back to original scale."""
        if self.mean_ is None:
            raise RuntimeError("Call fit before inverse_transform.")
        return _validate_2d(X_scaled) * self.scale_ + self.mean_


# ==========================================================
# MinMaxScaler
# ==========================================================

class MinMaxScaler:
    """
    Scale each feature to a given range [lo, hi], default [0, 1].

    For each feature j:
        X_scaled[:, j] = (X[:, j] - min_j) / (max_j - min_j) * (hi - lo) + lo

    Constant features (zero range) are mapped to lo.

    Parameters
    ----------
    feature_range : tuple of (float, float)
        Target range after scaling. Default is (0, 1).

    Attributes
    ----------
    data_min_ : ndarray of shape (n_features,)
    data_max_ : ndarray of shape (n_features,)
    data_range_ : ndarray of shape (n_features,)
        data_max_ - data_min_. Zero-range features have data_range_ = 1.
    """

    def __init__(self, feature_range: tuple = (0, 1)):
        lo, hi = feature_range
        if lo >= hi:
            raise ValueError("feature_range must satisfy lo < hi.")
        self.feature_range = feature_range
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None

    def fit(self, X):
        """Compute per-feature min and max from X."""
        X = _validate_2d(X)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        data_range = self.data_max_ - self.data_min_
        self.data_range_ = np.where(data_range == 0, 1.0, data_range)
        return self

    def transform(self, X):
        """Scale X to feature_range using the fitted min and max."""
        if self.data_min_ is None:
            raise RuntimeError("Call fit before transform.")
        lo, hi = self.feature_range
        X_std = (_validate_2d(X) - self.data_min_) / self.data_range_
        return X_std * (hi - lo) + lo

    def fit_transform(self, X):
        """Fit and scale X."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Reverse scaling back to the original range."""
        if self.data_min_ is None:
            raise RuntimeError("Call fit before inverse_transform.")
        lo, hi = self.feature_range
        X_std = (_validate_2d(X_scaled) - lo) / (hi - lo)
        return X_std * self.data_range_ + self.data_min_


# ==========================================================
# LabelEncoder
# ==========================================================

class LabelEncoder:
    """
    Encode a 1-D array of categorical labels as integers 0, 1, 2, ...

    The mapping is determined by sorted unique values seen during fit,
    so it is stable regardless of the order labels appear in the data.

    Attributes
    ----------
    classes_ : ndarray
        Sorted unique class labels seen during fit.
    """

    def __init__(self):
        self.classes_: Optional[np.ndarray] = None

    def fit(self, y):
        """Learn the set of unique labels from y."""
        self.classes_ = np.unique(_validate_1d(y))
        return self

    def transform(self, y):
        """Map labels in y to integer codes."""
        if self.classes_ is None:
            raise RuntimeError("Call fit before transform.")
        y = _validate_1d(y)
        unknown = ~np.isin(y, self.classes_)
        if unknown.any():
            raise ValueError(f"Unknown labels encountered: {np.unique(y[unknown])}")
        return np.searchsorted(self.classes_, y)

    def fit_transform(self, y):
        """Fit and encode y."""
        return self.fit(y).transform(y)

    def inverse_transform(self, codes):
        """Map integer codes back to original labels."""
        if self.classes_ is None:
            raise RuntimeError("Call fit before inverse_transform.")
        codes = np.asarray(codes)
        if codes.min() < 0 or codes.max() >= len(self.classes_):
            raise ValueError("codes contain values outside the fitted range.")
        return self.classes_[codes]


# ==========================================================
# OneHotEncoder
# ==========================================================

class OneHotEncoder:
    """
    Encode integer class labels as binary indicator rows.

    Label k becomes a row with a 1 in column k and 0s everywhere else.

    Attributes
    ----------
    n_classes_ : int
        Number of classes inferred from the maximum label seen during fit.
    """

    def __init__(self):
        self.n_classes_: Optional[int] = None

    def fit(self, y):
        """Learn the number of classes from y."""
        y = _validate_1d(y)
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("OneHotEncoder expects integer labels.")
        if y.min() < 0:
            raise ValueError("Labels must be non-negative integers.")
        self.n_classes_ = int(y.max()) + 1
        return self

    def transform(self, y):
        """Return a (n_samples, n_classes) binary matrix."""
        if self.n_classes_ is None:
            raise RuntimeError("Call fit before transform.")
        y = _validate_1d(y)
        out = np.zeros((len(y), self.n_classes_), dtype=float)
        out[np.arange(len(y)), y] = 1.0
        return out

    def fit_transform(self, y):
        """Fit and encode y."""
        return self.fit(y).transform(y)

    def inverse_transform(self, one_hot):
        """Return integer class labels from a one-hot matrix."""
        return np.argmax(np.asarray(one_hot), axis=1)


# ==========================================================
# train_test_split
# ==========================================================

def train_test_split(*arrays, test_size: float = 0.2, random_state: Optional[int] = None):
    """
    Split arrays into random train and test subsets.

    All arrays must have the same number of rows. A single random permutation
    is applied to all of them so corresponding rows stay aligned.

    Parameters
    ----------
    *arrays : sequence of array-like
        Arrays to split. All must share the same first dimension.
    test_size : float
        Fraction of samples reserved for the test set. Must be in (0, 1).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    splits : list
        Alternating train/test pairs: [X_train, X_test, y_train, y_test, ...]

    Example
    -------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1).")
    if len(arrays) == 0:
        raise ValueError("At least one array is required.")

    arrays = [np.asarray(a) for a in arrays]
    n_samples = arrays[0].shape[0]
    if any(a.shape[0] != n_samples for a in arrays):
        raise ValueError("All arrays must have the same number of rows.")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)
    n_test = max(1, int(np.ceil(test_size * n_samples)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    result = []
    for a in arrays:
        result.append(a[train_idx])
        result.append(a[test_idx])
    return result
