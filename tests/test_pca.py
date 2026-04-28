"""
Unit tests for pca.py.

Tests verify dimensionality reduction shape, explained variance properties,
component orthogonality, and round-trip reconstruction accuracy.
"""

import numpy as np
import pytest
from mlpackage import PCA


# -------------------------------------------------------------------------
# Correctness
# -------------------------------------------------------------------------

def test_output_shape_matches_n_components():
    """
    fit_transform must return an array of shape (n_samples, n_components).
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))

    X_reduced = PCA(n_components=2).fit_transform(X)

    assert X_reduced.shape == (50, 2)


def test_all_components_explained_variance_ratio_sums_to_one():
    """
    When n_components equals n_features the ratios must sum to exactly 1.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4))

    pca = PCA(n_components=4).fit(X)

    np.testing.assert_allclose(pca.explained_variance_ratio_.sum(), 1.0, atol=1e-10)


def test_partial_components_explained_variance_ratio_less_than_one():
    """
    Keeping fewer than all components must yield a ratio sum strictly < 1.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 6))

    pca = PCA(n_components=2).fit(X)

    assert pca.explained_variance_ratio_.sum() < 1.0


def test_components_are_orthonormal():
    """
    The principal component matrix must satisfy components @ components.T ≈ I.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 5))

    pca = PCA(n_components=3).fit(X)
    gram = pca.components_ @ pca.components_.T

    np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)


def test_inverse_transform_reconstructs_original():
    """
    When n_components equals n_features, inverse_transform should recover X.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 3))

    pca = PCA(n_components=3).fit(X)
    X_reconstructed = pca.inverse_transform(pca.transform(X))

    np.testing.assert_allclose(X_reconstructed, X, atol=1e-10)


def test_explained_variance_is_descending():
    """
    Components must be ordered by descending explained variance.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 5))

    pca = PCA(n_components=4).fit(X)
    ev = pca.explained_variance_

    assert all(ev[i] >= ev[i + 1] for i in range(len(ev) - 1))


def test_n_components_attribute_after_fit():
    """n_components_ must equal the requested number of components."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5))

    pca = PCA(n_components=3).fit(X)

    assert pca.n_components_ == 3


def test_mean_subtracted_correctly():
    """
    mean_ must match the column-wise mean of the training data.
    """
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    pca = PCA(n_components=1).fit(X)

    np.testing.assert_allclose(pca.mean_, X.mean(axis=0), atol=1e-12)


# -------------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------------

def test_transform_before_fit_raises():
    """transform must raise RuntimeError when called before fit."""
    pca = PCA(n_components=2)
    with pytest.raises(RuntimeError):
        pca.transform(np.ones((5, 3)))


def test_n_components_exceeds_n_features_raises():
    """fit must raise ValueError when n_components > n_features."""
    X = np.ones((10, 3))
    with pytest.raises(ValueError):
        PCA(n_components=5).fit(X)


def test_invalid_n_components_raises():
    """Constructor must raise ValueError when n_components < 1."""
    with pytest.raises(ValueError):
        PCA(n_components=0)
