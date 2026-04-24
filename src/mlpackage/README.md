# mlpackage

A from-scratch machine learning library built with NumPy. No scikit-learn dependencies.

---

## Supervised Learning

### `linear_regression.py` — `LinearRegression`

Supports three solvers sharing a common `fit` / `predict` interface:

- **OLS** — normal equations via least-squares factorization
- **Ridge** — L2-regularized least squares; intercept is never penalized
- **Gradient descent** — iterative MSE minimization

Optional intercept prepended automatically (`fit_intercept=True`). Evaluation metrics: R², MSE, RMSE, MAE.

---

### `logistic_regression.py` — `LogisticRegression`

Binary classifier using the logistic function and binary cross-entropy loss, trained by batch gradient descent. L2 regularization on feature weights. Exposes `predict_proba`, `predict`, `decision_function`, and a manually computed ROC curve and AUC.

---

### `perceptron.py` — `Perceptron`

Rosenblatt perceptron for binary classification. Online weight updates — only misclassified samples trigger a weight change. Halts on convergence or epoch limit. Accepts any two distinct label values.

---

### `multilayer_perceptron.py` — `MLPClassifier`, `MLPRegressor`

Fully connected network trained by backpropagation and batch gradient descent. ReLU hidden activations, He initialization, optional L2 regularization on weight matrices. Architecture specified as a list of hidden layer widths at construction.

---

### `decision_tree.py` — `DecisionTreeClassifier`

CART-style binary classification tree. Supports entropy (information gain) and Gini impurity as splitting criteria. Exposes `predict_proba`, `score`, `get_depth`, and normalized `feature_importances_`. Optional feature subsampling per split (`max_features`).

---

### `regression_trees.py` — `RegressionTree`

CART regression tree that minimizes mean squared error at each split. Leaves predict the mean target value of their training samples. Stops on `max_depth`, `min_samples_split`, or when no split reduces MSE.

---

### `ensemble_methods.py` — `BaggingClassifier`, `VotingClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`

| Class | Method |
|---|---|
| `BaggingClassifier` | Bootstrap aggregation over any base estimator; majority vote |
| `VotingClassifier` | Hard vote over a fixed set of heterogeneous estimators |
| `RandomForestClassifier` | Bootstrap + feature subsampling; averages class probabilities |
| `AdaBoostClassifier` | Sequential boosting via SAMME; weighted majority vote |

All expose `fit` / `predict` / `score`.

---

### `knn.py` — `KNNClassifier`, `KNNRegressor`

Brute-force k-nearest neighbors. Classifiers use majority vote and expose `predict_proba`; regressors average neighbor targets. Supports uniform and distance-based weighting, and Euclidean or Manhattan distance.

---

### `gradient_descent.py` — `GradientDescent1D`, `GradientDescentND`

Standalone gradient descent optimizers.

- `GradientDescent1D` — scalar parameter w ∈ ℝ, given an explicit derivative df/dw
- `GradientDescentND` — vector parameter w ∈ ℝⁿ, given a gradient function ∇f(w)

---

### `distance_metrics.py`

Standalone distance functions operating on 1-D NumPy arrays.

| Function | Formula |
|---|---|
| `euclidean` | √∑(a − b)² |
| `manhattan` | ∑\|a − b\| |
| `chebyshev` | max\|a − b\| |
| `minkowski` | (∑\|a − b\|ᵖ)^(1/p) |
| `cosine` | 1 − a·b / (‖a‖‖b‖) |
| `hamming` | fraction of positions where a ≠ b |

---

## Unsupervised Learning

### `k_means_clustering.py` — `KMeans`

Lloyd's algorithm. Centroids initialized randomly from the data or user-supplied. Convergence detected by maximum centroid shift across clusters. Exposes `cluster_centers_`, `labels_`, `inertia_`, and `n_iter_`. Also accepts new points via `predict`.

---

### `dbscan.py` — `DBSCAN`

Density-based clustering. Points with at least `min_samples` neighbors within radius `eps` are core points; reachable non-core points are border points; everything else is noise (label `-1`). No preset number of clusters required. Exposes `labels_` and `core_sample_indices_`.

---

### `pca.py` — `PCA`

Linear dimensionality reduction via eigendecomposition of the covariance matrix. Components are ordered by descending explained variance. Exposes `components_`, `explained_variance_`, `explained_variance_ratio_`, and `mean_`. Supports `transform`, `fit_transform`, and `inverse_transform`.

---

### `community_detection.py` — `LabelPropagation`

Label propagation for community detection in undirected graphs. Each node iteratively adopts the majority label of its neighbors. Supports weighted edges (votes summed by edge weight). Update order is shuffled each iteration to reduce bias. Output communities are relabelled 0, 1, 2, ... in order of first appearance.

---

## Dependencies

- `numpy` — all numerical computation
