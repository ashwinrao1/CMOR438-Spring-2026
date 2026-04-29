"""
Postprocessing

Common operations applied to model predictions and outputs after inference.
All operations are implemented from scratch using NumPy.

Functions
---------
confusion_matrix
    Count true/false positives and negatives in a (n_classes, n_classes) matrix.
accuracy_score
    Fraction of correctly predicted labels.
precision_score
    Precision per class: TP / (TP + FP).
recall_score
    Recall per class: TP / (TP + FN).
f1_score
    Harmonic mean of precision and recall per class.
classification_report
    Print precision, recall, and F1 for every class.
roc_curve
    Compute TPR and FPR at every threshold (binary classification).
auc
    Area under a curve via the trapezoidal rule.
mean_squared_error
    Average squared difference between predictions and targets.
mean_absolute_error
    Average absolute difference between predictions and targets.
r2_score
    Coefficient of determination R².

No sklearn dependencies are used.
"""

from __future__ import annotations
import numpy as np

__all__ = [
    "confusion_matrix",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "classification_report",
    "roc_curve",
    "auc",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
]


# ==========================================================
# Utility
# ==========================================================

def _validate_1d_pair(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    return y_true, y_pred


# ==========================================================
# Classification metrics
# ==========================================================

def confusion_matrix(y_true, y_pred):
    """
    Compute a confusion matrix of shape (n_classes, n_classes).

    Entry [i, j] is the number of samples with true label i predicted as j.

    Parameters
    ----------
    y_true : array-like of int, shape (n_samples,)
    y_pred : array-like of int, shape (n_samples,)

    Returns
    -------
    cm : ndarray of shape (n_classes, n_classes)

    Examples
    --------
    >>> confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])
    array([[2, 0],
           [1, 1]])
    """
    y_true, y_pred = _validate_1d_pair(y_true, y_pred)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    # np.add.at accumulates counts without a Python loop over samples.
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy_score(y_true, y_pred):
    """
    Fraction of samples where y_true == y_pred.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    accuracy : float in [0, 1]

    Examples
    --------
    >>> accuracy_score([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    y_true, y_pred = _validate_1d_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision_score(y_true, y_pred, average: str = "macro"):
    """
    Precision per class: TP / (TP + FP).

    Parameters
    ----------
    average : {"macro", "weighted", None}
        "macro"    — unweighted mean across classes.
        "weighted" — mean weighted by true support per class.
        None       — return per-class array.

    Returns
    -------
    precision : float or ndarray
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    predicted_positive = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(predicted_positive == 0, 0.0, tp / predicted_positive)
    return _aggregate(per_class, cm.sum(axis=1), average)


def recall_score(y_true, y_pred, average: str = "macro"):
    """
    Recall per class: TP / (TP + FN).

    Parameters
    ----------
    average : {"macro", "weighted", None}

    Returns
    -------
    recall : float or ndarray
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    actual_positive = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(actual_positive == 0, 0.0, tp / actual_positive)
    return _aggregate(per_class, actual_positive, average)


def f1_score(y_true, y_pred, average: str = "macro"):
    """
    F1 score per class: 2 * precision * recall / (precision + recall).

    Parameters
    ----------
    average : {"macro", "weighted", None}

    Returns
    -------
    f1 : float or ndarray
    """
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    actual_positive = cm.sum(axis=1)
    predicted_positive = cm.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(predicted_positive == 0, 0.0, tp / predicted_positive)
        r = np.where(actual_positive == 0, 0.0, tp / actual_positive)
        denom = p + r
        per_class = np.where(denom == 0, 0.0, 2 * p * r / denom)

    return _aggregate(per_class, actual_positive, average)


def _aggregate(per_class, support, average):
    """Apply macro / weighted averaging or return raw per-class values."""
    if average is None:
        return per_class
    if average == "macro":
        return float(per_class.mean())
    if average == "weighted":
        total = support.sum()
        return float(0.0 if total == 0 else np.dot(per_class, support) / total)
    raise ValueError(f"average must be 'macro', 'weighted', or None, got {average!r}.")


def classification_report(y_true, y_pred, digits: int = 2):
    """
    Print precision, recall, F1, and support for every class.

    Parameters
    ----------
    digits : int
        Number of decimal places to display.
    """
    y_true, y_pred = _validate_1d_pair(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    tp = np.diag(cm)
    support = cm.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_positive = cm.sum(axis=0)
        precision = np.where(predicted_positive == 0, 0.0, tp / predicted_positive)
        recall = np.where(support == 0, 0.0, tp / support)
        denom = precision + recall
        f1 = np.where(denom == 0, 0.0, 2 * precision * recall / denom)

    col_width = max(digits + 6, 10)
    header = f"{'class':>8}  {'precision':>{col_width}}  {'recall':>{col_width}}  {'f1-score':>{col_width}}  {'support':>8}"
    print(header)
    print("-" * len(header))

    for k in range(n_classes):
        print(
            f"{k:>8}  {precision[k]:>{col_width}.{digits}f}  "
            f"{recall[k]:>{col_width}.{digits}f}  "
            f"{f1[k]:>{col_width}.{digits}f}  "
            f"{support[k]:>8}"
        )

    print("-" * len(header))
    total = support.sum()
    w_p = float(np.dot(precision, support) / total) if total else 0.0
    w_r = float(np.dot(recall, support) / total) if total else 0.0
    w_f = float(np.dot(f1, support) / total) if total else 0.0
    print(
        f"{'weighted avg':>8}  {w_p:>{col_width}.{digits}f}  "
        f"{w_r:>{col_width}.{digits}f}  "
        f"{w_f:>{col_width}.{digits}f}  "
        f"{total:>8}"
    )


# ==========================================================
# ROC / AUC (binary classification)
# ==========================================================

def roc_curve(y_true, y_score):
    """
    Compute the ROC curve for binary classification.

    Thresholds are swept from highest to lowest predicted score.
    At each threshold, a sample is predicted positive if its score >= threshold.

    Parameters
    ----------
    y_true : array-like of int, shape (n_samples,)
        Binary ground-truth labels (0 or 1).
    y_score : array-like of float, shape (n_samples,)
        Predicted probability or decision score for the positive class.

    Returns
    -------
    fpr : ndarray — false positive rates
    tpr : ndarray — true positive rates
    thresholds : ndarray — score thresholds used
    """
    y_true, y_score = _validate_1d_pair(y_true, y_score)
    y_true = y_true.astype(int)

    desc_idx = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc_idx]
    y_true_sorted = y_true[desc_idx]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve requires at least one positive and one negative sample.")

    tp_cumsum = np.cumsum(y_true_sorted)
    fp_cumsum = np.cumsum(1 - y_true_sorted)

    tpr = tp_cumsum / n_pos
    fpr = fp_cumsum / n_neg
    thresholds = y_score_sorted

    # Prepend the (0, 0) origin point.
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])

    return fpr, tpr, thresholds


def auc(x, y):
    """
    Compute the area under the curve (x, y) using the trapezoidal rule.

    Parameters
    ----------
    x : array-like, shape (n,)
    y : array-like, shape (n,)

    Returns
    -------
    area : float
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    order = np.argsort(x)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(_trapz(y[order], x[order]))


# ==========================================================
# Regression metrics
# ==========================================================

def mean_squared_error(y_true, y_pred):
    """
    Mean squared error: mean((y_true - y_pred)^2).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    mse : float

    Examples
    --------
    >>> mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    0.3333333333333333
    """
    y_true, y_pred = _validate_1d_pair(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true, y_pred):
    """
    Mean absolute error: mean(|y_true - y_pred|).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    mae : float

    Examples
    --------
    >>> mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    0.3333333333333333
    """
    y_true, y_pred = _validate_1d_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    """
    Coefficient of determination R².

    R² = 1 - SS_res / SS_tot
    where SS_res = sum((y_true - y_pred)^2)
      and SS_tot = sum((y_true - mean(y_true))^2).

    Returns 0.0 when SS_tot == 0 (constant target).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    r2 : float

    Examples
    --------
    >>> r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0
    """
    y_true, y_pred = _validate_1d_pair(y_true, y_pred)
    y_true = y_true.astype(float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(0.0 if ss_tot == 0 else 1 - ss_res / ss_tot)
