"""
Unit tests for distance_metrics.py.

All expected distances are computed analytically from the metric definitions
so the tests serve as ground-truth checks independent of the implementation.
"""

import numpy as np
import pytest
from mlpackage import (
    euclidean,
    manhattan,
    chebyshev,
    cosine,
    hamming,
    minkowski_metric,
    get_metric,
    pairwise_distances,
)


# =========================================================================
# euclidean
# =========================================================================

def test_euclidean_identical_vectors():
    """Distance between identical vectors must be 0."""
    a = np.array([1.0, 2.0, 3.0])
    assert euclidean(a, a) == pytest.approx(0.0)


def test_euclidean_known_value():
    """
    a=[0,0], b=[3,4] -> sqrt(9+16) = 5.
    """
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean(a, b) == pytest.approx(5.0)


def test_euclidean_symmetry():
    """euclidean(a, b) must equal euclidean(b, a)."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert euclidean(a, b) == pytest.approx(euclidean(b, a))


def test_euclidean_non_negative():
    """Distance must always be >= 0."""
    a = np.array([-3.0, 4.0])
    b = np.array([1.0, -2.0])
    assert euclidean(a, b) >= 0.0


# =========================================================================
# manhattan
# =========================================================================

def test_manhattan_identical_vectors():
    """Distance between identical vectors must be 0."""
    a = np.array([1.0, 2.0, 3.0])
    assert manhattan(a, a) == pytest.approx(0.0)


def test_manhattan_known_value():
    """
    a=[1,2], b=[4,6] -> |3|+|4| = 7.
    """
    a = np.array([1.0, 2.0])
    b = np.array([4.0, 6.0])
    assert manhattan(a, b) == pytest.approx(7.0)


def test_manhattan_symmetry():
    """manhattan(a, b) must equal manhattan(b, a)."""
    a = np.array([0.0, 5.0])
    b = np.array([3.0, 1.0])
    assert manhattan(a, b) == pytest.approx(manhattan(b, a))


# =========================================================================
# chebyshev
# =========================================================================

def test_chebyshev_identical_vectors():
    """Distance between identical vectors must be 0."""
    a = np.array([7.0, 3.0])
    assert chebyshev(a, a) == pytest.approx(0.0)


def test_chebyshev_known_value():
    """
    a=[1,5,2], b=[4,1,6] -> |3|, |4|, |4| -> max = 4.
    """
    a = np.array([1.0, 5.0, 2.0])
    b = np.array([4.0, 1.0, 6.0])
    assert chebyshev(a, b) == pytest.approx(4.0)


def test_chebyshev_symmetry():
    """chebyshev(a, b) must equal chebyshev(b, a)."""
    a = np.array([2.0, 8.0])
    b = np.array([5.0, 3.0])
    assert chebyshev(a, b) == pytest.approx(chebyshev(b, a))


# =========================================================================
# cosine
# =========================================================================

def test_cosine_parallel_vectors():
    """Two parallel vectors must have cosine distance 0."""
    a = np.array([1.0, 2.0, 3.0])
    b = 5.0 * a
    assert cosine(a, b) == pytest.approx(0.0, abs=1e-10)


def test_cosine_orthogonal_vectors():
    """Two orthogonal vectors must have cosine distance 1."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine(a, b) == pytest.approx(1.0)


def test_cosine_zero_vector_raises():
    """cosine distance with a zero vector must raise ValueError."""
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        cosine(a, b)


def test_cosine_symmetry():
    """cosine(a, b) must equal cosine(b, a)."""
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    assert cosine(a, b) == pytest.approx(cosine(b, a))


# =========================================================================
# hamming
# =========================================================================

def test_hamming_identical_vectors():
    """Identical vectors must have Hamming distance 0."""
    a = np.array([0, 1, 1, 0])
    assert hamming(a, a) == pytest.approx(0.0)


def test_hamming_all_different():
    """Vectors that differ in every position must have Hamming distance 1."""
    a = np.array([0, 0, 0])
    b = np.array([1, 1, 1])
    assert hamming(a, b) == pytest.approx(1.0)


def test_hamming_known_value():
    """
    a=[0,1,0,1], b=[1,1,0,0] -> mismatch at positions 0 and 3 -> 2/4 = 0.5.
    """
    a = np.array([0, 1, 0, 1])
    b = np.array([1, 1, 0, 0])
    assert hamming(a, b) == pytest.approx(0.5)


def test_hamming_empty_vector_raises():
    """Passing empty vectors must raise ValueError."""
    with pytest.raises(ValueError):
        hamming(np.array([]), np.array([]))


# =========================================================================
# minkowski_metric
# =========================================================================

def test_minkowski_p1_equals_manhattan():
    """minkowski(p=1) must equal manhattan on the same inputs."""
    a = np.array([1.0, 3.0])
    b = np.array([4.0, 7.0])
    assert minkowski_metric(p=1)(a, b) == pytest.approx(manhattan(a, b))


def test_minkowski_p2_equals_euclidean():
    """minkowski(p=2) must equal euclidean on the same inputs."""
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert minkowski_metric(p=2)(a, b) == pytest.approx(euclidean(a, b))


def test_minkowski_invalid_p_raises():
    """p < 1 violates the triangle inequality and must raise ValueError."""
    with pytest.raises(ValueError):
        minkowski_metric(p=0.5)


# =========================================================================
# get_metric
# =========================================================================

def test_get_metric_returns_callable():
    """get_metric must return a callable for every registered name."""
    for name in ["euclidean", "manhattan", "chebyshev", "cosine", "hamming"]:
        fn = get_metric(name)
        assert callable(fn)


def test_get_metric_unknown_name_raises():
    """An unregistered name must raise ValueError."""
    with pytest.raises(ValueError):
        get_metric("l3_norm")


def test_get_metric_euclidean_gives_correct_distance():
    """Metric retrieved by name must compute the same result as the direct function."""
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert get_metric("euclidean")(a, b) == pytest.approx(5.0)


# =========================================================================
# pairwise_distances
# =========================================================================

def test_pairwise_distances_shape():
    """Result shape must be (n, m) for inputs (n, d) and (m, d)."""
    A = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 1.0], [2.0, 2.0]])
    D = pairwise_distances(A, B)

    assert D.shape == (3, 2)


def test_pairwise_distances_self_distance_is_zero():
    """Diagonal of pairwise_distances(A, A) must be 0."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    D = pairwise_distances(A, A)

    np.testing.assert_allclose(np.diag(D), np.zeros(3), atol=1e-10)


def test_pairwise_distances_known_values():
    """
    A=[[0,0]], B=[[3,4]] -> D=[[5.0]].
    """
    A = np.array([[0.0, 0.0]])
    B = np.array([[3.0, 4.0]])
    D = pairwise_distances(A, B)

    np.testing.assert_allclose(D, np.array([[5.0]]), atol=1e-10)


def test_pairwise_distances_manhattan_metric():
    """pairwise_distances with metric='manhattan' must use Manhattan distance."""
    A = np.array([[1.0, 2.0]])
    B = np.array([[4.0, 6.0]])
    D = pairwise_distances(A, B, metric="manhattan")

    assert D[0, 0] == pytest.approx(7.0)


def test_pairwise_distances_callable_metric():
    """A callable metric must be accepted and applied correctly."""
    A = np.array([[0.0, 0.0]])
    B = np.array([[3.0, 4.0]])
    D = pairwise_distances(A, B, metric=euclidean)

    assert D[0, 0] == pytest.approx(5.0)


def test_pairwise_distances_feature_mismatch_raises():
    """A and B with different column counts must raise ValueError."""
    A = np.array([[1.0, 2.0]])
    B = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        pairwise_distances(A, B)
