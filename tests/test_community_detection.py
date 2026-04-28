"""
Unit tests for community_detection.py.

These tests validate correctness on small graphs where community structure
is known in advance.
"""

import numpy as np
from mlpackage import LabelPropagation


def test_two_communities():
    """
    Two disconnected components should form two communities.
    """
    # adjacency matrix for two disconnected triangles
    A = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ])

    model = LabelPropagation(n_iterations=50, random_state=42)
    labels = model.fit(A).labels_

    # first three nodes should share a label
    assert len(set(labels[:3])) == 1

    # last three nodes should share a label
    assert len(set(labels[3:])) == 1

    # the two groups should be different
    assert labels[0] != labels[3]


def test_single_component():
    """
    Fully connected graph should yield a single community.
    """
    n = 5
    A = np.ones((n, n)) - np.eye(n)

    model = LabelPropagation(n_iterations=50, random_state=42)
    labels = model.fit(A).labels_

    assert len(set(labels)) == 1


def test_isolated_nodes():
    """
    Isolated nodes should retain unique labels.
    """
    A = np.zeros((3, 3))

    model = LabelPropagation(n_iterations=10, random_state=42)
    labels = model.fit(A).labels_

    assert len(set(labels)) == 3


def test_labels_array_length_matches_n_nodes():
    """
    The labels array must have one entry per node in the graph.
    """
    n = 8
    A = np.zeros((n, n))

    model = LabelPropagation(n_iterations=10, random_state=42)
    labels = model.fit(A).labels_

    assert len(labels) == n


def test_labels_are_non_negative_integers():
    """
    Community labels must be non-negative integers.
    """
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ])

    model = LabelPropagation(n_iterations=20, random_state=42)
    labels = model.fit(A).labels_

    assert all(isinstance(int(l), int) for l in labels)
    assert all(l >= 0 for l in labels)
