"""
Distance Metrics

This module implements common vector distance functions used in machine
learning, including Euclidean (L2) and Manhattan (L1). All operations
use NumPy for performance and include strict validation to ensure that
inputs are 1-dimensional numeric arrays with matching shapes.

Supported metrics
-----------------
euclidean   L2 norm:      sqrt(sum((a - b)^2))
manhattan   L1 norm:      sum(|a - b|)
chebyshev   L-inf norm:   max(|a - b|)
minkowski   Lp norm:      sum(|a - b|^p)^(1/p)
cosine      angular dist: 1 - dot(a,b) / (||a|| * ||b||)
hamming     mismatch:     fraction of positions where a != b
"""

from __future__ import annotations
from typing import Callable
import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------

def _validate(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a)
    b = np.asarray(b)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            f"Inputs must be 1-D arrays, got shapes {a.shape} and {b.shape}."
        )
    if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
        raise TypeError("Inputs must be numeric arrays.")
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: {a.shape} vs {b.shape}. Inputs must have equal length."
        )
    return a, b


# ------------------------------------------------------------------
# Metric functions
# ------------------------------------------------------------------

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean (L2) distance: sqrt(sum((a - b)^2))

    The standard straight-line distance between two points in R^d.
    """
    a, b = _validate(a, b)
    diff = a - b
    return float(np.sqrt(diff @ diff))


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    """
    Manhattan (L1) distance: sum(|a - b|)

    Measures distance as the sum of absolute differences per coordinate.
    Less sensitive to large individual differences than Euclidean.
    """
    a, b = _validate(a, b)
    return float(np.sum(np.abs(a - b)))


def chebyshev(a: np.ndarray, b: np.ndarray) -> float:
    """
    Chebyshev (L-infinity) distance: max(|a - b|)

    Equals the largest absolute difference across all coordinates.
    Equivalent to Minkowski distance as p -> infinity.
    """
    a, b = _validate(a, b)
    return float(np.max(np.abs(a - b)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance: 1 - dot(a, b) / (||a|| * ||b||)

    Measures the angle between two vectors rather than their magnitude.
    Returns 0 when vectors are parallel, 1 when orthogonal.
    Undefined for zero vectors.
    """
    a, b = _validate(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cosine distance is undefined for zero vectors.")
    return float(1.0 - (a @ b) / (norm_a * norm_b))


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    """
    Hamming distance: fraction of positions where a != b.

    Ranges from 0 (identical) to 1 (all positions differ).
    Intended for categorical or binary feature vectors.
    """
    a, b = _validate(a, b)
    if len(a) == 0:
        raise ValueError("hamming distance requires non-empty vectors.")
    return float(np.mean(a != b))


def minkowski_metric(p: float) -> Metric:
    """
    Return a Minkowski distance function for exponent p.

    sum(|a - b|^p)^(1/p)

    p=1 recovers manhattan, p=2 recovers euclidean.
    p must be >= 1 to satisfy the triangle inequality.
    """
    if p < 1:
        raise ValueError(f"Minkowski p must be >= 1, got {p}.")

    def _minkowski(a: np.ndarray, b: np.ndarray) -> float:
        a, b = _validate(a, b)
        return float(np.sum(np.abs(a - b) ** p) ** (1.0 / p))

    _minkowski.__name__ = f"minkowski(p={p})"
    return _minkowski


# ------------------------------------------------------------------
# Registry and lookup
# ------------------------------------------------------------------

_REGISTRY: dict[str, Metric] = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "chebyshev": chebyshev,
    "cosine":    cosine,
    "hamming":   hamming,
    "minkowski": minkowski_metric(p=2),
}


def get_metric(name: str) -> Metric:
    """
    Return a metric function by name.

    For Minkowski with p != 2, call minkowski_metric(p) directly.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown metric {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


# ------------------------------------------------------------------
# Pairwise distance matrix
# ------------------------------------------------------------------

def pairwise_distances(
    A: np.ndarray,
    B: np.ndarray,
    metric: str | Metric = "euclidean",
) -> np.ndarray:
    """
    Compute an (n, m) distance matrix between rows of A and rows of B.

    result[i, j] = dist(A[i], B[j])

    Euclidean distance is computed via broadcasting to avoid the overhead of
    an explicit Python loop. All other metrics use a row-pair loop.

    Parameters
    ----------
    A : array of shape (n, d)
    B : array of shape (m, d)
    metric : str or callable

    Returns
    -------
    D : np.ndarray of shape (n, m)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must both be 2-D arrays.")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: A has {A.shape[1]} columns, "
            f"B has {B.shape[1]}."
        )

    fn: Metric = get_metric(metric) if isinstance(metric, str) else metric

    if fn is euclidean:
        return _euclidean_matrix(A, B)

    n, m = len(A), len(B)
    D = np.empty((n, m))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            D[i, j] = fn(a, b)
    return D


def _euclidean_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # ||a - b||^2 = ||a||^2 - 2*a@b^T + ||b||^2 avoids the (n, m, d) diff array.
    sq_A = np.sum(A * A, axis=1, keepdims=True)
    sq_B = np.sum(B * B, axis=1, keepdims=True)
    D_sq = sq_A + sq_B.T - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(D_sq, 0.0))
