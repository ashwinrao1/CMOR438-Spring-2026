"""
Decision Tree Classifier (CART, entropy / Gini)

Binary classification tree built from scratch with NumPy. Supports both
entropy-based information gain and Gini impurity as splitting criteria.

Candidate split thresholds are midpoints between consecutive sorted unique
feature values. Any threshold strictly between v_i and v_{i+1} produces the
same partition, so the midpoint is the conventional representative choice.

After fitting, feature_importances_ holds the normalized mean decrease in
impurity across all splits that used each feature.

Public API: fit, predict, predict_proba, score, get_depth
"""

from __future__ import annotations
from typing import Optional
import numpy as np


# ------------------------------------------------------------------
# Internal node
# ------------------------------------------------------------------

class _Node:
    """Single node in the tree. Leaves have feature=None."""

    __slots__ = ("feature", "threshold", "left", "right", "proba", "impurity", "n_samples")

    def __init__(
        self,
        *,
        proba: np.ndarray,
        impurity: float,
        n_samples: int,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_Node"] = None,
        right: Optional["_Node"] = None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.proba = proba
        self.impurity = impurity
        self.n_samples = n_samples

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


# ------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------

class DecisionTreeClassifier:
    """
    Decision tree classifier using information gain or Gini impurity.

    criterion="entropy" selects splits by maximizing:
        IG = H(parent) - (n_L/n)*H(left) - (n_R/n)*H(right)
    where H(S) = -sum_k p_k * log2(p_k)

    criterion="gini" selects splits by minimizing:
        (n_L/n)*G(left) + (n_R/n)*G(right)
    where G(S) = 1 - sum_k p_k^2

    Parameters
    ----------
    criterion : {"entropy", "gini"}
        Splitting criterion. Default is "entropy".
    max_depth : int or None
        Maximum tree depth. None grows the tree until all other stopping
        conditions are met.
    min_samples_split : int
        Minimum samples needed at a node to attempt a split.
    min_samples_leaf : int
        Minimum samples required in each resulting child node.
    max_features : int, float, or None
        Features to consider at each split. A float is treated as a fraction
        of n_features. None considers all features.
    random_state : int or None
        Seed for the feature-subsampling RNG.
    """

    def __init__(
        self,
        criterion: str = "entropy",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int | float] = None,
        random_state: Optional[int] = None,
    ):
        if criterion not in ("entropy", "gini"):
            raise ValueError(f"criterion must be 'entropy' or 'gini', got {criterion!r}")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.tree_: Optional[_Node] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self._rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features).")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be 1-D with one label per row of X.")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Class labels must be integers.")
        if y.min() < 0:
            raise ValueError("Class labels must be non-negative.")

        self.n_features_ = X.shape[1]
        self.n_classes_ = int(y.max()) + 1
        self._rng = np.random.default_rng(self.random_state)

        # Importance accumulator — filled during tree growth, normalized after.
        self._importance_acc = np.zeros(self.n_features_)
        self._n_total = len(y)

        self.tree_ = self._build(X, y, depth=0)

        total = self._importance_acc.sum()
        self.feature_importances_ = (
            self._importance_acc / total if total > 0 else self._importance_acc.copy()
        )
        del self._importance_acc, self._n_total

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Call fit before predicting.")
        X = np.asarray(X, dtype=float)
        return np.array([self._reach_leaf(x, self.tree_).proba for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of correctly classified samples."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_depth(self) -> int:
        """Maximum depth of the fitted tree (root = depth 0)."""
        if self.tree_ is None:
            raise RuntimeError("Call fit before get_depth.")
        return self._tree_depth(self.tree_)

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        proba = self._class_proba(y)
        impurity = self._impurity(proba)
        node = _Node(proba=proba, impurity=impurity, n_samples=len(y))

        if self._stop(y, depth):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feat, thresh, left_mask, right_mask = split

        # Mean decrease in impurity for this split, weighted by node sample fraction.
        n, n_l, n_r = len(y), left_mask.sum(), right_mask.sum()
        child_impurity = (
            n_l * self._impurity(self._class_proba(y[left_mask]))
            + n_r * self._impurity(self._class_proba(y[right_mask]))
        ) / n
        self._importance_acc[feat] += (n / self._n_total) * (impurity - child_impurity)

        node.feature = feat
        node.threshold = thresh
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _stop(self, y: np.ndarray, depth: int) -> bool:
        """True when this node should become a leaf."""
        if len(np.unique(y)) == 1:
            return True
        if len(y) < self.min_samples_split:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    # ------------------------------------------------------------------
    # Split selection
    # ------------------------------------------------------------------

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n = len(y)
        parent_impurity = self._impurity(self._class_proba(y))
        best_gain = 0.0
        best = None

        for feat in self._candidate_features(X.shape[1]):
            unique_vals = np.unique(X[:, feat])
            if len(unique_vals) < 2:
                continue

            # Midpoints are the only distinct candidates — see module docstring.
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask

                n_l, n_r = left_mask.sum(), right_mask.sum()
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                gain = parent_impurity - (
                    n_l * self._impurity(self._class_proba(y[left_mask]))
                    + n_r * self._impurity(self._class_proba(y[right_mask]))
                ) / n

                if gain > best_gain:
                    best_gain = gain
                    best = (feat, float(t), left_mask, right_mask)

        return best

    def _candidate_features(self, n_features: int) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features)
        if isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        else:
            k = min(int(self.max_features), n_features)
        return self._rng.choice(n_features, k, replace=False)

    # ------------------------------------------------------------------
    # Impurity measures
    # ------------------------------------------------------------------

    def _impurity(self, proba: np.ndarray) -> float:
        return self._entropy(proba) if self.criterion == "entropy" else self._gini(proba)

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        # Ignore zero-probability classes to avoid log(0).
        nz = p[p > 0]
        return float(-np.sum(nz * np.log2(nz)))

    @staticmethod
    def _gini(p: np.ndarray) -> float:
        return float(1.0 - np.sum(p * p))

    def _class_proba(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes_)
        return counts / counts.sum()

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def _reach_leaf(self, x: np.ndarray, node: _Node) -> _Node:
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def _tree_depth(self, node: _Node) -> int:
        if node.is_leaf:
            return 0
        return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))


# Convenience alias for cross-imports in ensemble_methods.py
DecisionTree = DecisionTreeClassifier
