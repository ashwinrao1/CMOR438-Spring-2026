"""
Unit tests for postprocessing.py.

All expected values are computed by hand or from the mathematical definitions
so the tests do not depend on any external library for ground truth.
"""

import numpy as np
import pytest
from mlpackage import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# =========================================================================
# confusion_matrix
# =========================================================================

def test_confusion_matrix_perfect_predictions():
    """Perfect predictions must produce a diagonal matrix."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    cm = confusion_matrix(y_true, y_pred)

    np.testing.assert_array_equal(cm, np.eye(3, dtype=int))


def test_confusion_matrix_known_values():
    """Verify entry [i, j] counts true-i predicted-as-j correctly."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    cm = confusion_matrix(y_true, y_pred)

    # true-0 predicted-0: 2, true-0 predicted-1: 0
    # true-1 predicted-0: 1, true-1 predicted-1: 1
    expected = np.array([[2, 0], [1, 1]])
    np.testing.assert_array_equal(cm, expected)


def test_confusion_matrix_shape():
    """Output shape must be (n_classes, n_classes)."""
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 2, 1, 2])
    cm = confusion_matrix(y_true, y_pred)

    assert cm.shape == (3, 3)


def test_confusion_matrix_sum_equals_n_samples():
    """Sum of all entries must equal the number of samples."""
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2])
    cm = confusion_matrix(y_true, y_pred)

    assert cm.sum() == len(y_true)


# =========================================================================
# accuracy_score
# =========================================================================

def test_accuracy_score_all_correct():
    """Perfect predictions must yield accuracy 1.0."""
    y = np.array([0, 1, 2])
    assert accuracy_score(y, y) == 1.0


def test_accuracy_score_all_wrong():
    """All wrong predictions on a two-class problem must yield 0.0."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    assert accuracy_score(y_true, y_pred) == 0.0


def test_accuracy_score_known_value():
    """Three correct out of four must yield 0.75."""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)


# =========================================================================
# precision_score
# =========================================================================

def test_precision_score_perfect():
    """Perfect predictions must yield precision 1.0."""
    y = np.array([0, 1, 0, 1])
    assert precision_score(y, y) == pytest.approx(1.0)


def test_precision_score_macro_known_value():
    """
    y_true = [0,0,1,1], y_pred = [0,1,1,1]
    class 0: TP=1, FP=0 -> precision=1.0
    class 1: TP=2, FP=1 -> precision=2/3
    macro = (1.0 + 2/3) / 2 = 5/6
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    expected = (1.0 + 2.0 / 3.0) / 2.0
    assert precision_score(y_true, y_pred, average="macro") == pytest.approx(expected)


def test_precision_score_per_class_shape():
    """average=None must return an array with one entry per class."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    per_class = precision_score(y_true, y_pred, average=None)

    assert per_class.shape == (3,)


# =========================================================================
# recall_score
# =========================================================================

def test_recall_score_perfect():
    """Perfect predictions must yield recall 1.0."""
    y = np.array([0, 1, 2, 0])
    assert recall_score(y, y) == pytest.approx(1.0)


def test_recall_score_macro_known_value():
    """
    y_true = [0,0,1,1], y_pred = [0,1,0,1]
    class 0: TP=1, FN=1 -> recall=0.5
    class 1: TP=1, FN=1 -> recall=0.5
    macro = 0.5
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    assert recall_score(y_true, y_pred, average="macro") == pytest.approx(0.5)


# =========================================================================
# f1_score
# =========================================================================

def test_f1_score_perfect():
    """Perfect predictions must yield F1 = 1.0."""
    y = np.array([0, 1, 0, 1])
    assert f1_score(y, y) == pytest.approx(1.0)


def test_f1_score_balanced_two_class_perfect():
    """
    A balanced two-class problem with perfect predictions must yield macro F1 = 1.0.
    Both classes have P=R=F1=1, so the macro average is also 1.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert f1_score(y_true, y_pred) == pytest.approx(1.0)


def test_f1_score_weighted_differs_from_macro():
    """Weighted and macro averages must differ on an imbalanced dataset."""
    y_true = np.array([0] * 9 + [1])
    y_pred = np.array([0] * 9 + [0])
    macro = f1_score(y_true, y_pred, average="macro")
    weighted = f1_score(y_true, y_pred, average="weighted")

    assert macro != weighted


# =========================================================================
# roc_curve / auc
# =========================================================================

def test_roc_curve_starts_at_origin():
    """The first point of the ROC curve must be (FPR=0, TPR=0)."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, _ = roc_curve(y_true, y_score)

    assert fpr[0] == pytest.approx(0.0)
    assert tpr[0] == pytest.approx(0.0)


def test_roc_curve_ends_at_one_one():
    """The last point of the ROC curve must be (FPR=1, TPR=1)."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    fpr, tpr, _ = roc_curve(y_true, y_score)

    assert fpr[-1] == pytest.approx(1.0)
    assert tpr[-1] == pytest.approx(1.0)


def test_roc_curve_perfect_classifier_auc_is_one():
    """A classifier that perfectly separates classes must have AUC = 1."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    fpr, tpr, _ = roc_curve(y_true, y_score)

    assert auc(fpr, tpr) == pytest.approx(1.0)


def test_roc_curve_requires_both_classes():
    """roc_curve must raise ValueError when only one class is present."""
    with pytest.raises(ValueError):
        roc_curve(np.array([1, 1, 1]), np.array([0.8, 0.9, 0.7]))


def test_auc_known_triangle():
    """AUC of a triangle with base 1 and height 1 must be 0.5."""
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])

    assert auc(x, y) == pytest.approx(0.5)


def test_auc_unsorted_x_gives_correct_result():
    """auc must sort x before integrating, so order of inputs should not matter."""
    x = np.array([1.0, 0.0])
    y = np.array([1.0, 0.0])

    assert auc(x, y) == pytest.approx(0.5)


# =========================================================================
# Regression metrics
# =========================================================================

def test_mse_perfect_predictions():
    """MSE must be 0 when predictions match targets exactly."""
    y = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y, y) == pytest.approx(0.0)


def test_mse_known_value():
    """
    y_true=[1,2,3], y_pred=[1,2,4]
    errors=[0,0,1], squared=[0,0,1], mean=1/3
    """
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    assert mean_squared_error(y_true, y_pred) == pytest.approx(1.0 / 3.0)


def test_mae_perfect_predictions():
    """MAE must be 0 when predictions match targets exactly."""
    y = np.array([1.0, 2.0, 3.0])
    assert mean_absolute_error(y, y) == pytest.approx(0.0)


def test_mae_known_value():
    """
    y_true=[1,2,3], y_pred=[2,2,2]
    abs errors=[1,0,1], mean=2/3
    """
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])

    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2.0 / 3.0)


def test_r2_score_perfect():
    """R² must be 1.0 for perfect predictions."""
    y = np.array([1.0, 2.0, 3.0])
    assert r2_score(y, y) == pytest.approx(1.0)


def test_r2_score_mean_predictor_is_zero():
    """R² of the mean-constant predictor is 0 by definition."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.full(3, y_true.mean())

    assert r2_score(y_true, y_pred) == pytest.approx(0.0)


def test_r2_score_can_be_negative():
    """A predictor worse than the mean must produce R² < 0."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([10.0, 10.0, 10.0])

    assert r2_score(y_true, y_pred) < 0.0
