"""
Ensemble Methods

BaggingClassifier
    Trains independent copies of any base estimator, each on a bootstrap
    sample of the training data. Predictions are made by majority vote across
    all estimators. Works with any estimator that implements fit and predict.

VotingClassifier
    Combines a fixed set of heterogeneous estimators via hard voting. Each
    estimator casts one vote per sample; the class with the most votes wins.

RandomForestClassifier
    Trains an independent ensemble of trees, each on a bootstrap sample of
    the training data. Predictions are made by averaging class probabilities
    across all trees. Feature subsampling at each split (max_features) is the
    primary mechanism that decorrelates the trees.

AdaBoostClassifier
    Trains trees sequentially using the SAMME algorithm. Each round, sample
    weights are updated to focus on previously misclassified examples. The
    final prediction is a weighted majority vote across all rounds.

All classes expose the same fit / predict / score interface.
"""

from __future__ import annotations
import copy
from typing import Any, Optional
import numpy as np

from .decision_tree import DecisionTreeClassifier


# ------------------------------------------------------------------
# Bagging
# ------------------------------------------------------------------

class BaggingClassifier:
    """
    Bootstrap Aggregating (Bagging) over any base estimator.

    Each estimator is trained on a bootstrap sample — n_samples rows drawn
    with replacement. Predictions are made by majority vote: each estimator
    contributes one vote per sample and the class with the most votes wins.

    Unlike RandomForestClassifier, BaggingClassifier does not subsample
    features at each split. It is the base estimator's responsibility to
    handle any internal randomness.

    Parameters
    ----------
    base_estimator : object with fit(X, y) and predict(X) methods
        Template estimator. A deep copy is made for each member of the
        ensemble so the original object is never modified.
        Defaults to DecisionTreeClassifier().
    n_estimators : int
        Number of estimators to train.
    random_state : int or None
        Seed for bootstrap sampling.
    """

    def __init__(
        self,
        base_estimator: Any = None,
        n_estimators: int = 10,
        random_state: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.estimators_: list[Any] = []
        self.n_classes_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_samples = len(y)
        self.n_classes_ = int(y.max()) + 1
        self.estimators_ = []

        template = self.base_estimator if self.base_estimator is not None else DecisionTreeClassifier()
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            indices = rng.integers(0, n_samples, size=n_samples)
            estimator = copy.deepcopy(template)
            estimator.fit(X[indices], y[indices])
            self.estimators_.append(estimator)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Call fit before predicting.")
        X = np.asarray(X, dtype=float)

        # Collect one prediction per estimator, then take the majority class.
        votes = np.array([est.predict(X) for est in self.estimators_])  # (n_estimators, n_samples)
        return np.array([
            np.bincount(votes[:, i], minlength=self.n_classes_).argmax()
            for i in range(X.shape[0])
        ])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


# ------------------------------------------------------------------
# Voting
# ------------------------------------------------------------------

class VotingClassifier:
    """
    Hard voting ensemble over a fixed set of heterogeneous estimators.

    Each estimator casts one vote per sample. The class that receives the
    most votes is the final prediction. Ties are broken by the lowest class
    index.

    Parameters
    ----------
    estimators : list of (name, estimator) tuples
        Each estimator must implement fit(X, y) and predict(X).
    """

    def __init__(self, estimators: list[tuple[str, Any]]):
        if not estimators:
            raise ValueError("estimators must be a non-empty list of (name, estimator) tuples.")
        self.estimators = estimators

        self.estimators_: list[Any] = []
        self.n_classes_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.n_classes_ = int(y.max()) + 1
        self.estimators_ = []

        for _, estimator in self.estimators:
            est = copy.deepcopy(estimator)
            est.fit(X, y)
            self.estimators_.append(est)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Call fit before predicting.")
        X = np.asarray(X, dtype=float)

        # Each fitted estimator votes; majority class wins per sample.
        votes = np.array([est.predict(X) for est in self.estimators_])  # (n_estimators, n_samples)
        return np.array([
            np.bincount(votes[:, i], minlength=self.n_classes_).argmax()
            for i in range(X.shape[0])
        ])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))


# ------------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------------

class RandomForestClassifier:
    """
    Ensemble of decision trees trained on bootstrap samples.

    Each tree is trained on a bootstrap sample (n samples drawn with
    replacement) and considers only `max_features` candidate features at
    every split. These two sources of randomness decorrelate the trees so
    that the average prediction has lower variance than any single tree.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum depth per tree.
    min_samples_split : int
        Passed through to each DecisionTreeClassifier.
    min_samples_leaf : int
        Passed through to each DecisionTreeClassifier.
    max_features : int, float, or {"sqrt", "log2"}
        Features to consider at each split.
        "sqrt"  -> int(sqrt(n_features))  [default, standard for classification]
        "log2"  -> int(log2(n_features))
        int     -> exact count
        float   -> fraction of n_features
    criterion : {"entropy", "gini"}
        Splitting criterion passed to each tree.
    bootstrap : bool
        If False, each tree is trained on the full dataset (pasting).
    random_state : int or None
        Seed for the ensemble RNG. Each tree gets a derived seed so results
        are fully reproducible.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | float | str = "sqrt",
        criterion: str = "entropy",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators_: list[DecisionTreeClassifier] = []
        self.n_classes_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.n_classes_ = int(y.max()) + 1
        self.estimators_ = []

        rng = np.random.default_rng(self.random_state)
        max_feat = self._resolve_max_features(n_features)

        importance_sum = np.zeros(n_features)

        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 2**31))

            if self.bootstrap:
                indices = rng.integers(0, n_samples, size=n_samples)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_feat,
                random_state=seed,
            )
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)
            importance_sum += tree.feature_importances_

        self.feature_importances_ = importance_sum / self.n_estimators
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Call fit before predicting.")
        X = np.asarray(X, dtype=float)
        # Average class probabilities across all trees.
        proba_sum = sum(tree.predict_proba(X) for tree in self.estimators_)
        return proba_sum / self.n_estimators

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        return min(int(self.max_features), n_features)


# ------------------------------------------------------------------
# AdaBoost (SAMME)
# ------------------------------------------------------------------

class AdaBoostClassifier:
    """
    Adaptive boosting using the SAMME algorithm.

    SAMME (Stagewise Additive Modeling using a Multiclass Exponential loss)
    extends the classic binary AdaBoost to K > 2 classes. Each round:

        1. Train a weak classifier h_t on the current sample weights.
        2. Compute weighted error: ε_t = Σ w_i * 1[h_t(x_i) != y_i]
        3. Compute round weight: α_t = lr * (log((1 - ε_t) / ε_t) + log(K - 1))
        4. Update sample weights: w_i *= exp(α_t * 1[h_t(x_i) != y_i])
        5. Renormalize weights to sum to 1.

    Final prediction: argmax over classes of Σ_t α_t * 1[h_t(x) == k]

    Rounds where ε_t >= 1 - 1/K (no better than chance) are skipped.

    Parameters
    ----------
    n_estimators : int
        Maximum number of boosting rounds.
    learning_rate : float
        Shrinks the contribution of each round. Lower values require more
        estimators to achieve the same fit.
    max_depth : int
        Depth of each weak learner. Depth-1 stumps are the standard choice.
    criterion : {"entropy", "gini"}
        Splitting criterion for each weak learner.
    random_state : int or None
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        criterion: str = "entropy",
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state

        self.estimators_: list[DecisionTreeClassifier] = []
        self.alphas_: list[float] = []
        self.n_classes_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_samples = len(y)
        self.n_classes_ = int(y.max()) + 1
        K = self.n_classes_
        self.estimators_ = []
        self.alphas_ = []

        # Uniform weights to start.
        weights = np.full(n_samples, 1.0 / n_samples)
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            # Sample the dataset according to current weights.
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=weights)
            X_w, y_w = X[indices], y[indices]

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                random_state=int(rng.integers(0, 2**31)),
            )
            tree.fit(X_w, y_w)
            predictions = tree.predict(X)

            incorrect = (predictions != y).astype(float)
            eps = np.dot(weights, incorrect)

            # Skip rounds that are no better than random guessing for K classes.
            if eps >= 1.0 - (1.0 / K):
                continue

            # SAMME alpha includes the log(K-1) term for multi-class correction.
            alpha = self.learning_rate * (np.log((1.0 - eps) / eps) + np.log(K - 1))

            # Misclassified samples get higher weight next round.
            weights *= np.exp(alpha * incorrect)
            weights /= weights.sum()

            self.estimators_.append(tree)
            self.alphas_.append(alpha)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Call fit before predicting.")
        X = np.asarray(X, dtype=float)

        # Accumulate weighted votes: score[i, k] = sum of alphas where h_t(x_i) == k
        scores = np.zeros((len(X), self.n_classes_))
        for tree, alpha in zip(self.estimators_, self.alphas_):
            preds = tree.predict(X)
            for k in range(self.n_classes_):
                scores[:, k] += alpha * (preds == k)

        return np.argmax(scores, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Soft probability estimates via normalized SAMME scores.

        These are not calibrated probabilities. Use predict() for hard labels
        and treat predict_proba() as a ranking signal only.
        """
        if not self.estimators_:
            raise RuntimeError("Call fit before predicting.")
        X = np.asarray(X, dtype=float)

        scores = np.zeros((len(X), self.n_classes_))
        for tree, alpha in zip(self.estimators_, self.alphas_):
            preds = tree.predict(X)
            for k in range(self.n_classes_):
                scores[:, k] += alpha * (preds == k)

        # Shift and normalize to [0, 1] so rows sum to 1.
        scores -= scores.min(axis=1, keepdims=True)
        totals = scores.sum(axis=1, keepdims=True)
        return np.where(totals == 0, 1.0 / self.n_classes_, scores / totals)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
