"""
Unit tests for dbscan.py.

Tests verify that DBSCAN correctly identifies clusters and noise on
small synthetic datasets where the expected output is known.
"""

import numpy as np
import pytest
from mlpackage import DBSCAN


# -------------------------------------------------------------------------
# Correctness
# -------------------------------------------------------------------------

def test_two_clusters_no_noise():
    """
    Two well-separated dense blobs should form two clusters with no noise.
    """
    X = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],
    ])

    db = DBSCAN(eps=0.5, min_samples=2).fit(X)

    # Exactly two distinct non-noise clusters.
    clusters = set(db.labels_) - {-1}
    assert len(clusters) == 2
    # No noise points.
    assert -1 not in db.labels_


def test_isolated_point_is_noise():
    """
    An isolated point that lies outside the eps neighborhood of any dense region
    must be labeled -1 (noise).
    """
    X = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],  # dense blob
        [100.0, 100.0],                          # isolated
    ])

    db = DBSCAN(eps=0.5, min_samples=2).fit(X)

    assert db.labels_[3] == -1


def test_single_dense_blob_one_cluster():
    """
    A single dense blob should produce exactly one cluster and no noise.
    """
    X = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
    ])

    db = DBSCAN(eps=0.5, min_samples=2).fit(X)

    assert len(set(db.labels_) - {-1}) == 1
    assert -1 not in db.labels_


def test_all_points_noise_when_min_samples_too_high():
    """
    When min_samples exceeds the size of every neighborhood, all points must
    be labeled as noise (-1).
    """
    X = np.array([[0.0], [1.0], [2.0]])  # three isolated points

    db = DBSCAN(eps=0.1, min_samples=10).fit(X)

    assert all(label == -1 for label in db.labels_)


def test_fit_predict_returns_labels():
    """fit_predict must return the same array as fit().labels_."""
    X = np.array([[0.0], [0.1], [10.0], [10.1]])

    db = DBSCAN(eps=0.5, min_samples=2)
    labels_fp = db.fit_predict(X)
    labels_fit = db.labels_

    np.testing.assert_array_equal(labels_fp, labels_fit)


def test_labels_array_length_matches_n_samples():
    """labels_ must have the same length as the input array."""
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])

    db = DBSCAN(eps=0.5, min_samples=2).fit(X)

    assert len(db.labels_) == len(X)


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------

def test_invalid_eps_raises():
    """Constructor must raise ValueError when eps <= 0."""
    with pytest.raises(ValueError):
        DBSCAN(eps=0.0)


def test_invalid_min_samples_raises():
    """Constructor must raise ValueError when min_samples < 1."""
    with pytest.raises(ValueError):
        DBSCAN(min_samples=0)
