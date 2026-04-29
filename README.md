# CMOR 438 / INDE 577 — Machine Learning

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
├── data/                   # Local datasets (mall_customers.csv)
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

## Running the tests

```bash
pytest
```

CI runs the same command on every push via `.github/workflows/tests.yml`.

## mlpackage

A from-scratch machine learning library. scikit-learn is never used for
models or preprocessing; it appears only for loading built-in datasets
(`load_digits`, `load_breast_cancer`) and `adjusted_rand_score`.

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
| `Linear_Regression/linear_regression_example.ipynb` | Synthetic / real | OLS, Ridge, gradient descent solvers; residual analysis |
| `Logistic_Regression/logistic_regression.ipynb` | Binary classification | Sigmoid decision boundary, ROC curve, AUC |
| `Perceptron/perceptron_example.ipynb` | Synthetic binary | Online weight update, convergence visualization |
| `Multilayer_Perceptron/multilayer_perceptron_example.ipynb` | Digits | Backprop MLP, loss curves, confusion matrix |
| `Decision_Tree/decision_tree_example.ipynb` | UCI Adult Census | Depth sweep, feature importances, PCA decision boundary |
| `Regression_Trees/regression_trees_example.ipynb` | Real-valued target | Depth vs. MSE, leaf mean prediction |
| `Ensemble_Models/ensemble_models_example.ipynb` | Classification | Bagging, Random Forest, AdaBoost; accuracy vs. estimator count |
| `KNN/knn_ex.ipynb` | Digits / regression | k sweep, distance weighting, regression targets |
| `Gradient_Descent/gradient_descent_example.ipynb` | Synthetic | 1-D and N-D loss surface, convergence history |

### Unsupervised learning

| Notebook | Dataset | Topic |
|---|---|---|
| `K-Means_Clustering/k_means_clustering_example.ipynb` | Mall Customers | Elbow method, cluster profiles, inertia |
| `DBScan/dbscan_example.ipynb` | Synthetic shapes | Noise detection, eps / min_samples sensitivity |
| `PCA/pca_example.ipynb` | Breast Cancer | Scree plot, biplot, reconstruction error |
| `Community_Detection/community_detection_example.ipynb` | Graph data | Label propagation, modularity, community structure |

## License

MIT License. Copyright (c) 2026 Ashwin Rao. See [LICENSE](LICENSE) for details.
