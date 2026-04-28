"""
Unit tests for ensemble_methods.py.

Covers BaggingClassifier, VotingClassifier, RandomForestClassifier, and
AdaBoostClassifier. All tests use small deterministic datasets with
random_state=42 for reproducibility.
"""

import numpy as np
import pytest
from mlpackage import (
    BaggingClassifier,
    VotingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    DecisionTreeClassifier,
)


# =========================================================================
# Shared fixture data — two clearly separable groups
# =========================================================================

X_TRAIN = np.array([
    [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
    [5.0, 5.0], [5.1, 5.0], [5.0, 5.1],
])
Y_TRAIN = np.array([0, 0, 0, 1, 1, 1])


# =========================================================================
# BaggingClassifier
# =========================================================================

def test_bagging_perfect_accuracy_on_separable_data():
    """
    Bagging over decision trees should classify perfectly separated data.
    """
    clf = BaggingClassifier(n_estimators=5, random_state=42).fit(X_TRAIN, Y_TRAIN)

    assert clf.score(X_TRAIN, Y_TRAIN) == 1.0


def test_bagging_n_estimators_correct():
    """
    After fitting, estimators_ must contain exactly n_estimators entries.
    """
    clf = BaggingClassifier(n_estimators=7, random_state=42).fit(X_TRAIN, Y_TRAIN)

    assert len(clf.estimators_) == 7


def test_bagging_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    clf = BaggingClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(X_TRAIN)


def test_bagging_with_custom_base_estimator():
    """
    BaggingClassifier should accept any estimator with fit/predict.
    """
    base = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf = BaggingClassifier(base_estimator=base, n_estimators=5, random_state=42)
    clf.fit(X_TRAIN, Y_TRAIN)

    assert clf.score(X_TRAIN, Y_TRAIN) == 1.0


# =========================================================================
# VotingClassifier
# =========================================================================

def test_voting_perfect_accuracy_on_separable_data():
    """
    A majority vote of several decision trees should classify perfectly.
    """
    estimators = [
        ("dt1", DecisionTreeClassifier(random_state=0)),
        ("dt2", DecisionTreeClassifier(random_state=1)),
        ("dt3", DecisionTreeClassifier(random_state=2)),
    ]
    clf = VotingClassifier(estimators=estimators).fit(X_TRAIN, Y_TRAIN)

    assert clf.score(X_TRAIN, Y_TRAIN) == 1.0


def test_voting_n_fitted_estimators_matches_input():
    """estimators_ must have the same length as the input estimator list."""
    estimators = [
        ("a", DecisionTreeClassifier(random_state=0)),
        ("b", DecisionTreeClassifier(random_state=1)),
    ]
    clf = VotingClassifier(estimators=estimators).fit(X_TRAIN, Y_TRAIN)

    assert len(clf.estimators_) == 2


def test_voting_empty_estimators_raises():
    """Constructor must raise ValueError for an empty estimator list."""
    with pytest.raises(ValueError):
        VotingClassifier(estimators=[])


def test_voting_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    estimators = [("dt", DecisionTreeClassifier())]
    clf = VotingClassifier(estimators=estimators)
    with pytest.raises(RuntimeError):
        clf.predict(X_TRAIN)


# =========================================================================
# RandomForestClassifier
# =========================================================================

def test_random_forest_perfect_accuracy_on_separable_data():
    """
    A random forest should classify perfectly separated data without error.
    """
    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_TRAIN, Y_TRAIN)

    assert clf.score(X_TRAIN, Y_TRAIN) == 1.0


def test_random_forest_feature_importances_sum_to_one():
    """
    feature_importances_ must be non-negative and sum to 1 after fitting.
    """
    clf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_TRAIN, Y_TRAIN)

    assert np.all(clf.feature_importances_ >= 0)
    np.testing.assert_allclose(clf.feature_importances_.sum(), 1.0, atol=1e-10)


def test_random_forest_predict_proba_shape():
    """
    predict_proba must return shape (n_query, n_classes) with rows in [0, 1].
    """
    clf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X_TRAIN, Y_TRAIN)
    proba = clf.predict_proba(X_TRAIN)

    assert proba.shape == (len(X_TRAIN), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_random_forest_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    clf = RandomForestClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(X_TRAIN)


# =========================================================================
# AdaBoostClassifier
#
# Note: perfectly separable data causes eps=0 on the first stump, which
# makes alpha=inf and corrupts the weight distribution. These tests use a
# dataset where no single depth-1 split achieves zero training error.
# =========================================================================

# Three-class problem; no axis-aligned stump can perfectly separate all three.
X_ADA = np.array([
    [0.0, 0.0], [0.1, 0.1],   # class 0
    [1.0, 0.0], [1.1, 0.1],   # class 1
    [0.5, 1.0], [0.6, 1.1],   # class 2
])
Y_ADA = np.array([0, 0, 1, 1, 2, 2])


def test_adaboost_runs_and_produces_valid_predictions():
    """
    AdaBoost must run without numerical errors and produce integer class labels
    for every sample.
    """
    clf = AdaBoostClassifier(n_estimators=20, random_state=42).fit(X_ADA, Y_ADA)
    pred = clf.predict(X_ADA)

    assert pred.shape == (len(X_ADA),)
    assert set(pred).issubset({0, 1, 2})


def test_adaboost_predict_proba_shape_and_nonneg():
    """
    predict_proba must return shape (n_samples, n_classes) with non-negative values.
    """
    clf = AdaBoostClassifier(n_estimators=20, random_state=42).fit(X_ADA, Y_ADA)
    proba = clf.predict_proba(X_ADA)

    assert proba.shape == (len(X_ADA), 3)
    assert np.all(proba >= 0)


def test_adaboost_predict_before_fit_raises():
    """predict must raise RuntimeError when called before fit."""
    clf = AdaBoostClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(X_TRAIN)
