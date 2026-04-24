"""
Shared internal helpers for the supervised_learning sub-package.

All functions here are private (prefixed with _) and not part of the
public API. Import them with:

    from ._utils import _as2d_float, _as1d, _as1d_float, _sigmoid, _add_intercept
"""

from __future__ import annotations
from typing import Union, Sequence
import numpy as np

ArrayLike = Union[np.ndarray, Sequence]


def _as2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Cast X to a non-empty 2-D float array or raise a clear ValueError."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _as1d(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Cast y to a 1-D array, preserving its dtype."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    return arr


def _as1d_float(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Cast y to a 1-D float array."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
    return arr


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: clips z to avoid overflow in exp."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to X to represent the bias term."""
    return np.hstack([np.ones((X.shape[0], 1)), X])
