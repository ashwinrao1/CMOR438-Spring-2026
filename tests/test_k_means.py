"""
Unit tests for k_means_clustering.py.

Tests use synthetic datasets with well-separated cluster centres where the
correct partition is known in advance.
"""

import numpy as np
import pytest
from mlpackage import KMeans


# -------------------------------------------------------------------------
# Correctness
# -------------------------------------------------------------------------

def test_two_well_separated_clusters_partition_correctly():
    """
    Two tight, far-apart blobs should be assigned to different clusters.
    """
    X = np.array([
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
        [10.0, 10.0], [10.1, 10.0], [10.0, 10.1],
    ])

    km = KMeans(n_clusters=2, random_state=42).fit(X)

    # The first three and last three must share a label.
    assert len(set(km.labels_[:3])) == 1
    assert len(set(km.labels_[3:])) == 1
    # The two groups must be in different clusters.
    assert km.labels_[0] != km.labels_[3]


def test_n_unique_labels_equals_n_clusters():
    """
    The number of distinct labels must equal n_clusters on well-separated data.
    """
    rng = np.random.default_rng(0)
    blobs = [rng.normal(loc=center, scale=0.1, size=(10, 2))
             for center in [0.0, 10.0, 20.0]]
    X = np.vstack(blobs)

    km = KMeans(n_clusters=3, random_state=42).fit(X)

    assert len(np.unique(km.labels_)) == 3


def test_inertia_is_non_negative():
    """inertia_ must be a non-negative float after fitting."""
    X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    km = KMeans(n_clusters=2, random_state=42).fit(X)

    assert km.inertia_ >= 0.0


def test_fit_predict_matches_labels():
    """fit_predict must return the same labels as fit().labels_."""
    X = np.array([[0.0], [0.1], [10.0], [10.1]])

    km = KMeans(n_clusters=2, random_state=42)
    labels_fp = km.fit_predict(X)
    labels_fit = km.labels_

    np.testing.assert_array_equal(labels_fp, labels_fit)


def test_predict_assigns_new_points_to_nearest_centroid():
    """
    After fitting, predict should assign a new point to the centroid closest to it.
    """
    X_train = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    km = KMeans(n_clusters=2, random_state=42).fit(X_train)

    # A point near (0, 0) must land in the same cluster as the origin.
    label_origin = km.predict(np.array([[0.0, 0.0]]))[0]
    label_new = km.predict(np.array([[0.05, 0.0]]))[0]

    assert label_origin == label_new


def test_user_supplied_init_centroids():
    """
    When init centroids are provided the algorithm must start from them and
    converge to the correct partition.
    """
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    init = np.array([[0.0], [10.0]])

    km = KMeans(n_clusters=2, init=init).fit(X)

    assert km.labels_[0] == km.labels_[1]
    assert km.labels_[2] == km.labels_[3]
    assert km.labels_[0] != km.labels_[2]


def test_n_iter_recorded():
    """n_iter_ must be a positive integer after fitting."""
    X = np.array([[0.0], [1.0], [5.0], [6.0]])
    km = KMeans(n_clusters=2, random_state=42).fit(X)

    assert isinstance(km.n_iter_, int)
    assert km.n_iter_ >= 1


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------

def test_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    km = KMeans(n_clusters=2)
    with pytest.raises(RuntimeError):
        km.predict(np.array([[0.0]]))


def test_n_clusters_exceeds_n_samples_raises():
    """fit must raise ValueError when n_clusters > n_samples."""
    X = np.array([[0.0], [1.0]])
    km = KMeans(n_clusters=5)
    with pytest.raises(ValueError):
        km.fit(X)


def test_invalid_n_clusters_raises():
    """Constructor must raise ValueError when n_clusters < 1."""
    with pytest.raises(ValueError):
        KMeans(n_clusters=0)
