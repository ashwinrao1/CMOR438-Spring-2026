# Ensemble Methods

## Model Architecture

Ensemble methods combine multiple base learners to produce a prediction more accurate than any single model. Four ensembles are demonstrated alongside a single Decision Tree baseline:

**Random Forest** trains an independent decision tree on each of $B$ bootstrap samples. At each split only a random subset of features is considered, which decorrelates the trees. Final predictions are by majority vote. Averaging uncorrelated estimators reduces variance without increasing bias.

**AdaBoost** builds trees sequentially. After each round, misclassified training samples receive higher weights so the next tree focuses on the hard cases. The final prediction is a weighted vote of all trees.

**Bagging** applies bootstrap aggregation using a single base estimator type without random feature subsets, so trees are more correlated than in a Random Forest and the variance-reduction benefit is smaller.

**Voting** combines heterogeneous classifiers (Random Forest, AdaBoost, Decision Tree) by majority vote, leveraging the diversity of their error patterns.

## Dataset

**UCI Adult Census Income** — 32,561 records from the 1994 US Census. Six numeric features; binary income target (>$50K vs ≤$50K); 75%/25% class imbalance. Same preprocessing as the Decision Tree notebook.

## What the Notebook Covers

- Training all four ensembles and a single Decision Tree baseline; reporting train and test accuracy for all five models
- Side-by-side accuracy bar chart (Random Forest best at test 0.8348; AdaBoost 0.8282 and Bagging 0.8253 fall below the baseline of 0.8287)
- Random Forest feature importances confirming capital-gain as the dominant variable
- n_estimators sweep for Random Forest showing accuracy stabilising after ~50 trees
- Confusion matrix and classification report for the best ensemble (Random Forest)
