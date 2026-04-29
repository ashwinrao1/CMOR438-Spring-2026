# Supervised Learning Notebooks

Nine notebooks covering regression, classification, ensemble methods, and
gradient-based optimization. Each uses a real dataset and reports both train
and test metrics side-by-side.

| Notebook | Dataset | Key experiment |
|---|---|---|
| `Linear_Regression/linear_regression_example.ipynb` | UCI Combined Cycle Power Plant (9,568 samples) | OLS vs Ridge coefficient shrinkage; test R² 0.9311 |
| `Logistic_Regression/logistic_regression_example.ipynb` | UCI Bank Marketing (10,000 samples) | Class imbalance effect; minority-class F1 = 0.30 |
| `Perceptron/perceptron_example.ipynb` | UCI Bank Marketing (5,000 samples) | Convergence plateau at epoch 10; learning rate insensitivity |
| `Multilayer_Perceptron/multilayer_perceptron_example.ipynb` | UCI Dry Bean + CCPP | Architecture sweep; regression test R² 0.9382 vs OLS 0.9311 |
| `Decision_Tree/decision_tree_example.ipynb` | UCI Adult Census (32,561 samples) | Depth sweep; overfit gap 0.9990 train vs 0.7725 test at depth None |
| `Regression_Trees/regression_trees_example.ipynb` | UCI CCPP (9,568 samples) | Tree R² 0.9433 vs OLS 0.9311; non-linear AT–PE curvature |
| `Ensemble_Models/ensemble_models_example.ipynb` | UCI Adult Census | Random Forest 0.8348 vs single tree 0.8287; AdaBoost minority-class reweighting |
| `KNN/knn_example.ipynb` | UCI Dry Bean (13,611 samples) | k sweep; k=10 optimal at 0.9291; BOMBAY 100%, SIRA 87.2% |
| `Gradient_Descent/gradient_descent_example.ipynb` | UCI CCPP (single feature) | Learning rate effects; GD solution matches OLS exactly |
