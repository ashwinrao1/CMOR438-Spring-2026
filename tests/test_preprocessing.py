"""
Unit tests for preprocessing.py.

Tests use small arrays whose expected output can be verified by hand.
No external data sources are required.
"""

import numpy as np
import pytest
from mlpackage import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
    train_test_split,
)


# =========================================================================
# StandardScaler
# =========================================================================

def test_standard_scaler_zero_mean():
    """After fit_transform, every feature column must have mean ~ 0."""
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    X_scaled = StandardScaler().fit_transform(X)

    np.testing.assert_allclose(X_scaled.mean(axis=0), np.zeros(2), atol=1e-10)


def test_standard_scaler_unit_variance():
    """After fit_transform, every feature column must have std ~ 1."""
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    X_scaled = StandardScaler().fit_transform(X)

    np.testing.assert_allclose(X_scaled.std(axis=0), np.ones(2), atol=1e-10)


def test_standard_scaler_constant_feature_unchanged():
    """A constant feature (zero variance) must be left as zeros, not NaN or inf."""
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    X_scaled = StandardScaler().fit_transform(X)

    np.testing.assert_array_equal(X_scaled[:, 0], np.zeros(3))


def test_standard_scaler_inverse_transform_roundtrip():
    """inverse_transform must recover the original values."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_recovered = scaler.inverse_transform(X_scaled)

    np.testing.assert_allclose(X_recovered, X, atol=1e-10)


def test_standard_scaler_transform_before_fit_raises():
    """transform called before fit must raise RuntimeError."""
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.array([[1.0, 2.0]]))


def test_standard_scaler_fit_on_train_transform_on_test():
    """Scaler fitted on train data must apply the same shift to test data."""
    X_train = np.array([[0.0], [2.0], [4.0]])
    X_test = np.array([[6.0]])
    scaler = StandardScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    expected = (6.0 - 2.0) / X_train.std()
    np.testing.assert_allclose(X_test_scaled[0, 0], expected, atol=1e-10)


# =========================================================================
# MinMaxScaler
# =========================================================================

def test_minmax_scaler_output_range():
    """After fit_transform with default range, all values must be in [0, 1]."""
    X = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
    X_scaled = MinMaxScaler().fit_transform(X)

    assert X_scaled.min() >= 0.0
    assert X_scaled.max() <= 1.0


def test_minmax_scaler_min_maps_to_zero():
    """The original minimum of each feature must map to 0."""
    X = np.array([[2.0, 5.0], [4.0, 15.0], [6.0, 25.0]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    np.testing.assert_allclose(X_scaled.min(axis=0), np.zeros(2), atol=1e-10)


def test_minmax_scaler_max_maps_to_one():
    """The original maximum of each feature must map to 1."""
    X = np.array([[2.0, 5.0], [4.0, 15.0], [6.0, 25.0]])
    X_scaled = MinMaxScaler().fit_transform(X)

    np.testing.assert_allclose(X_scaled.max(axis=0), np.ones(2), atol=1e-10)


def test_minmax_scaler_custom_range():
    """feature_range=(-1, 1) must map min to -1 and max to 1."""
    X = np.array([[0.0], [5.0], [10.0]])
    X_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)

    np.testing.assert_allclose(X_scaled[0, 0], -1.0, atol=1e-10)
    np.testing.assert_allclose(X_scaled[-1, 0], 1.0, atol=1e-10)


def test_minmax_scaler_inverse_transform_roundtrip():
    """inverse_transform must recover the original values."""
    X = np.array([[1.0, 2.0], [3.0, 8.0], [5.0, 14.0]])
    scaler = MinMaxScaler()
    X_recovered = scaler.inverse_transform(scaler.fit_transform(X))

    np.testing.assert_allclose(X_recovered, X, atol=1e-10)


def test_minmax_scaler_invalid_range_raises():
    """Constructing with lo >= hi must raise ValueError."""
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 0))


# =========================================================================
# LabelEncoder
# =========================================================================

def test_label_encoder_integer_codes_are_contiguous():
    """Encoded labels must be 0, 1, 2, ... with no gaps."""
    le = LabelEncoder()
    codes = le.fit_transform(["cat", "dog", "bird", "cat"])

    assert set(codes) == {0, 1, 2}


def test_label_encoder_sorted_order():
    """classes_ must be sorted; 'bird' < 'cat' < 'dog' alphabetically."""
    le = LabelEncoder().fit(["dog", "cat", "bird"])

    np.testing.assert_array_equal(le.classes_, ["bird", "cat", "dog"])


def test_label_encoder_inverse_transform_roundtrip():
    """inverse_transform must recover original labels from integer codes."""
    labels = ["a", "b", "a", "c"]
    le = LabelEncoder()
    codes = le.fit_transform(labels)
    recovered = le.inverse_transform(codes)

    np.testing.assert_array_equal(recovered, labels)


def test_label_encoder_unknown_label_raises():
    """transform must raise ValueError when an unseen label is passed."""
    le = LabelEncoder().fit(["a", "b"])

    with pytest.raises(ValueError):
        le.transform(["a", "z"])


def test_label_encoder_transform_before_fit_raises():
    """transform before fit must raise RuntimeError."""
    with pytest.raises(RuntimeError):
        LabelEncoder().transform(["a", "b"])


# =========================================================================
# OneHotEncoder
# =========================================================================

def test_one_hot_encoder_output_shape():
    """Output must be (n_samples, n_classes)."""
    y = np.array([0, 1, 2, 1])
    ohe = OneHotEncoder().fit_transform(y)

    assert ohe.shape == (4, 3)


def test_one_hot_encoder_rows_sum_to_one():
    """Each row must contain exactly one 1 and sum to 1."""
    y = np.array([0, 1, 2, 0])
    ohe = OneHotEncoder().fit_transform(y)

    np.testing.assert_array_equal(ohe.sum(axis=1), np.ones(4))


def test_one_hot_encoder_correct_column_is_set():
    """Label k must produce a 1 only in column k."""
    y = np.array([2])
    ohe = OneHotEncoder().fit_transform(y)

    assert ohe[0, 2] == 1.0
    assert ohe[0, 0] == 0.0
    assert ohe[0, 1] == 0.0


def test_one_hot_encoder_inverse_transform_roundtrip():
    """inverse_transform must recover the original integer labels."""
    y = np.array([0, 1, 2, 1, 0])
    ohe = OneHotEncoder()
    recovered = ohe.inverse_transform(ohe.fit_transform(y))

    np.testing.assert_array_equal(recovered, y)


def test_one_hot_encoder_non_integer_raises():
    """Fitting on float labels must raise ValueError."""
    with pytest.raises(ValueError):
        OneHotEncoder().fit(np.array([0.5, 1.5]))


# =========================================================================
# train_test_split
# =========================================================================

def test_train_test_split_sizes():
    """Train and test subsets together must contain all original samples."""
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    assert len(X_train) + len(X_test) == 10
    assert len(y_train) + len(y_test) == 10


def test_train_test_split_no_overlap():
    """No row index may appear in both train and test sets."""
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_rows = set(map(tuple, X_train.tolist()))
    test_rows = set(map(tuple, X_test.tolist()))
    assert len(train_rows & test_rows) == 0


def test_train_test_split_X_y_stay_aligned():
    """Rows of X and y must remain paired after splitting."""
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    np.testing.assert_array_equal(X_train.ravel(), y_train)
    np.testing.assert_array_equal(X_test.ravel(), y_test)


def test_train_test_split_reproducible_with_seed():
    """Two calls with the same random_state must produce identical splits."""
    X = np.arange(50).reshape(25, 2)
    y = np.arange(25)
    split_a = train_test_split(X, y, test_size=0.2, random_state=7)
    split_b = train_test_split(X, y, test_size=0.2, random_state=7)

    for a, b in zip(split_a, split_b):
        np.testing.assert_array_equal(a, b)


def test_train_test_split_invalid_test_size_raises():
    """test_size outside (0, 1) must raise ValueError."""
    X = np.arange(10).reshape(5, 2)
    y = np.arange(5)

    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=1.5)
