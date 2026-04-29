# CMOR 438 / INDE 577 — Machine Learning

[![CI](https://github.com/ashwinrao1/CMOR438-Spring-2026/actions/workflows/tests.yml/badge.svg)](https://github.com/ashwinrao1/CMOR438-Spring-2026/actions/workflows/tests.yml)

From-scratch implementations of core machine learning algorithms built with
NumPy, demonstrated on real datasets through Jupyter notebooks, and validated
by a pytest unit test suite.

## Repository layout

```
CMOR438-Spring-2026/
├── src/mlpackage/          # Python package — all algorithm implementations
├── notebooks/              # Jupyter notebooks organized by topic
│   ├── supervised_learning/
│   └── unsupervised_learning/
├── tests/                  # pytest unit test suite
├── data/                   # Local datasets
├── pyproject.toml          # Package build and dependency config
└── requirements.txt        # Full environment for running notebooks
```

## Setup

**Minimum (package + tests only):**

```bash
pip install -e ".[dev]"
```

**Full environment (notebooks + visualizations):**

```bash
pip install -r requirements.txt
pip install -e .
```

Requires Python 3.10 or later.

## Quick start

```python
from mlpackage import (
    RandomForestClassifier, StandardScaler, train_test_split, accuracy_score
)
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print(f"Train accuracy: {accuracy_score(y_train, clf.predict(X_train)):.4f}")
print(f"Test  accuracy: {accuracy_score(y_test,  clf.predict(X_test)):.4f}")
```

## Running the tests

```bash
pytest
```

CI runs the same command on every push via `.github/workflows/tests.yml`.

## mlpackage

A from-scratch machine learning library. scikit-learn is never used for
models or preprocessing; it appears only for dataset loading
(`load_breast_cancer` in the quick-start example) and `adjusted_rand_score`.

All public classes and functions are importable from the top level:

```python
from mlpackage import (
    LinearRegression, LogisticRegression, DecisionTreeClassifier,
    RandomForestClassifier, KNNClassifier, MLPClassifier,
    KMeans, DBSCAN, PCA, LabelPropagation,
    StandardScaler, train_test_split, accuracy_score,
)
```

See [src/mlpackage/README.md](src/mlpackage/README.md) for full API
documentation of every class and function.

### Supervised learning

| Class | Source file | Algorithm |
|---|---|---|
| `LinearRegression` | `linear_regression.py` | OLS, Ridge, gradient descent |
| `LogisticRegression` | `logistic_regression.py` | Binary cross-entropy, batch GD, L2 reg |
| `Perceptron` | `perceptron.py` | Rosenblatt online update rule |
| `MLPClassifier`, `MLPRegressor` | `multilayer_perceptron.py` | Backprop, ReLU hidden layers, He init |
| `DecisionTreeClassifier` | `decision_tree.py` | CART, entropy / Gini, feature importances |
| `RegressionTree` | `regression_trees.py` | CART, variance reduction, mean-leaf prediction |
| `BaggingClassifier` | `ensemble_methods.py` | Bootstrap aggregation, majority vote |
| `VotingClassifier` | `ensemble_methods.py` | Hard vote over heterogeneous estimators |
| `RandomForestClassifier` | `ensemble_methods.py` | Bootstrap + feature subsampling, probability averaging |
| `AdaBoostClassifier` | `ensemble_methods.py` | Sequential SAMME boosting |
| `KNNClassifier`, `KNNRegressor` | `knn.py` | Brute-force k-NN, uniform / distance weighting |
| `GradientDescent1D`, `GradientDescentND` | `gradient_descent.py` | Scalar and vector GD with early stopping |

### Unsupervised learning

| Class | Source file | Algorithm |
|---|---|---|
| `KMeans` | `k_means_clustering.py` | Lloyd's algorithm, random centroid init |
| `DBSCAN` | `dbscan.py` | Density-based clustering, noise label -1 |
| `PCA` | `pca.py` | Eigendecomposition of covariance matrix |
| `LabelPropagation` | `community_detection.py` | Iterative majority-label graph clustering |

### Processing utilities

| Class / Function | Purpose |
|---|---|
| `StandardScaler` | Zero-mean, unit-variance normalization |
| `MinMaxScaler` | Scale features to [0, 1] |
| `LabelEncoder`, `OneHotEncoder` | Categorical encoding |
| `train_test_split` | Random train/test partition |
| `confusion_matrix`, `classification_report` | Classification evaluation |
| `accuracy_score`, `precision_score`, `recall_score`, `f1_score` | Classification metrics |
| `roc_curve`, `auc` | ROC analysis |
| `mean_squared_error`, `mean_absolute_error`, `r2_score` | Regression metrics |

## Notebooks

Each notebook runs top-to-bottom without errors, includes EDA, trains the
corresponding `mlpackage` class, and produces labeled plots with
`sns.set_style("whitegrid")`.

### Supervised learning

| Notebook | Dataset | Topic |
|---|---|---|
| `Linear_Regression/linear_regression_example.ipynb` | UCI Combined Cycle Power Plant (9,568 samples) | OLS, Ridge, gradient descent solvers; residual analysis |
| `Logistic_Regression/logistic_regression_example.ipynb` | UCI Bank Marketing (10,000 samples) | Sigmoid decision boundary, ROC curve, AUC |
| `Perceptron/perceptron_example.ipynb` | UCI Bank Marketing (5,000 samples) | Online weight update, convergence visualization |
| `Multilayer_Perceptron/multilayer_perceptron_example.ipynb` | UCI Dry Bean (classification) + UCI CCPP (regression) | Backprop MLP, architecture sweep, loss curves |
| `Decision_Tree/decision_tree_example.ipynb` | UCI Adult Census (32,561 samples) | Depth sweep, feature importances, PCA decision boundary |
| `Regression_Trees/regression_trees_example.ipynb` | UCI Combined Cycle Power Plant (9,568 samples) | Depth vs. MSE, comparison to OLS |
| `Ensemble_Models/ensemble_models_example.ipynb` | UCI Adult Census (32,561 samples) | Bagging, Random Forest, AdaBoost; accuracy vs. estimator count |
| `KNN/knn_example.ipynb` | UCI Dry Bean (13,611 samples) | k sweep, distance metric comparison, PCA visualization |
| `Gradient_Descent/gradient_descent_example.ipynb` | UCI Combined Cycle Power Plant (9,568 samples) | 1-D and N-D loss surface, convergence history, GD vs. OLS |

### Unsupervised learning

| Notebook | Dataset | Topic |
|---|---|---|
| `K-Means_Clustering/k_means_clustering_example.ipynb` | UCI Dry Bean (13,611 samples) | Elbow method, ARI vs. ground truth, cluster profiles |
| `DBScan/dbscan_example.ipynb` | USGS Global Earthquakes 2023 (1,780 events) | Noise detection, eps sensitivity, tectonic arc geometry |
| `PCA/pca_example.ipynb` | UCI Dry Bean (13,611 samples) | Scree plot, feature loadings, reconstruction error |
| `Community_Detection/community_detection_example.ipynb` | Zachary's Karate Club (34 nodes, 78 edges) | Label propagation, modularity, ARI vs. ground-truth factions |

## License

MIT License. Copyright (c) 2026 Ashwin Rao. See [LICENSE](LICENSE) for details.
