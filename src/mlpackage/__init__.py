"""
mlpackage — a from-scratch machine learning library built with NumPy.

All public classes and functions are importable directly from this top-level
package, so you can write:

    from mlpackage import KMeans, StandardScaler, accuracy_score

instead of navigating into the sub-packages.
"""

# Supervised learning — models
from .supervised_learning import (
    LinearRegression,
    LogisticRegression,
    Perceptron,
    MLPClassifier,
    MLPRegressor,
    DecisionTreeClassifier,
    RegressionTree,
    BaggingClassifier,
    VotingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    KNNClassifier,
    KNNRegressor,
)

# Supervised learning — utilities
from .supervised_learning import (
    GradientDescent1D,
    GradientDescentND,
    euclidean,
    manhattan,
    chebyshev,
    cosine,
    hamming,
    minkowski_metric,
    get_metric,
    pairwise_distances,
)

# Unsupervised learning
from .unsupervised_learning import (
    KMeans,
    DBSCAN,
    PCA,
    LabelPropagation,
)

# Processing
from .processing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
    train_test_split,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

__all__ = [
    # supervised — models
    "LinearRegression",
    "LogisticRegression",
    "Perceptron",
    "MLPClassifier",
    "MLPRegressor",
    "DecisionTreeClassifier",
    "RegressionTree",
    "BaggingClassifier",
    "VotingClassifier",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "KNNClassifier",
    "KNNRegressor",
    # supervised — utilities
    "GradientDescent1D",
    "GradientDescentND",
    "euclidean",
    "manhattan",
    "chebyshev",
    "cosine",
    "hamming",
    "minkowski_metric",
    "get_metric",
    "pairwise_distances",
    # unsupervised
    "KMeans",
    "DBSCAN",
    "PCA",
    "LabelPropagation",
    # preprocessing
    "StandardScaler",
    "MinMaxScaler",
    "LabelEncoder",
    "OneHotEncoder",
    "train_test_split",
    # postprocessing
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
