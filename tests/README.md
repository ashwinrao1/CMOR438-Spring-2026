# tests/

Unit tests for `mlpackage`. All tests are written with `pytest` and use
small synthetic datasets where the correct output is known in advance.
No network access or external data files are required to run the suite.

## Running the tests

From the repository root:

```bash
pip install -e ".[dev]"
pytest
```

The CI workflow (`.github/workflows/tests.yml`) runs the same command on
every push and pull request against Python 3.11.

## Test modules

| File | Module under test | What is verified |
|---|---|---|
| `test_linear_regression.py` | `linear_regression.py` | OLS recovers exact coefficients and R² = 1 on training data; Ridge shrinks toward zero; gradient descent solver converges |
| `test_logistic_regression.py` | `logistic_regression.py` | Perfect accuracy on linearly separable binary data; `predict_proba` rows sum to 1; `score` stays in [0, 1] |
| `test_perceptron.py` | `perceptron.py` | OR and AND gates classified without error; `coef_` shape matches input dimensionality; raises before fit |
| `test_decision_tree.py` | `decision_tree.py` | Pure-class input produces depth-0 leaf; `max_depth` is never exceeded; both entropy and Gini criteria classify separable data correctly |
| `test_regression_trees.py` | `regression_trees.py` | Step-function split yields exact predictions; leaf predicts mean when splitting is blocked; `max_depth=1` produces at most two distinct predictions |
| `test_ensemble_methods.py` | `ensemble_methods.py` | Bagging, VotingClassifier, RandomForest, and AdaBoost all achieve perfect accuracy on clearly separable data; `n_estimators` attribute is respected; unfit models raise `RuntimeError` |
| `test_knn.py` | `knn.py` | 1-NN recovers training labels exactly; majority vote selects the correct class; `predict_proba` sums to 1; distance weighting and Manhattan distance produce valid predictions |
| `test_multilayer_perceptron.py` | `multilayer_perceptron.py` | `MLPClassifier` learns a linearly separable binary problem; `predict_proba` shape and row sums are correct; `MLPRegressor` converges on a synthetic linear signal |
| `test_gradient_descent.py` | `gradient_descent.py` | `GradientDescent1D` reaches the minimum of f(w) = w²; history tuples have correct types and length; early stopping fires when tolerance is met; `GradientDescentND` minimizes a 3-D quadratic |
| `test_k_means.py` | `k_means_clustering.py` | Two blobs are assigned to different clusters; number of unique labels equals `n_clusters`; inertia is non-negative; `fit_predict` agrees with `labels_` |
| `test_dbscan.py` | `dbscan.py` | Two separated blobs produce two clusters with no noise; isolated point is labeled -1; single dense blob yields one cluster; `eps` and `min_samples` edge cases |
| `test_pca.py` | `pca.py` | Output shape matches `n_components`; explained variance ratios sum to 1 when all components are kept; principal components are orthonormal; `inverse_transform` round-trip error is below tolerance |
| `test_community_detection.py` | `community_detection.py` | Two disconnected triangles form two communities; fully connected graph yields one community; isolated nodes retain unique labels |
| `test_preprocessing.py` | `preprocessing.py` | `StandardScaler` produces zero mean and unit variance, handles constant features, and round-trips via `inverse_transform`; `MinMaxScaler` maps min to 0 and max to 1 and supports custom ranges; `LabelEncoder` sorts classes and raises on unseen labels; `OneHotEncoder` shape and row-sum invariants; `train_test_split` preserves alignment and is reproducible with a seed |
| `test_postprocessing.py` | `postprocessing.py` | `confusion_matrix` entry counts and diagonal shape; `accuracy_score`, `precision_score`, `recall_score`, `f1_score` on known cases with macro and weighted averaging; `roc_curve` starts at origin, ends at (1, 1), and yields AUC = 1 for a perfect classifier; `auc` via the trapezoidal rule; MSE, MAE, and R² on analytically known values including the zero-error and mean-predictor edge cases |
| `test_distance_metrics.py` | `distance_metrics.py` | Each metric (`euclidean`, `manhattan`, `chebyshev`, `cosine`, `hamming`) is tested for identity-zero, known value, and symmetry; `minkowski_metric(p=1)` and `minkowski_metric(p=2)` recover Manhattan and Euclidean respectively; `get_metric` returns callables for all registered names; `pairwise_distances` produces the correct shape, zero diagonal, and delegates to callable metrics |

## Conventions

- `random_state=42` everywhere randomness is involved.
- Each test function has exactly one logical assertion (one reason to fail).
- Tests target behavior, not implementation details: no access to private attributes beyond what the public API exposes.
- No mocking; all tests run against the real implementation.
