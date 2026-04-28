# Decision Tree Classifier

## Model Architecture

A decision tree classifier partitions the feature space through a sequence of binary splits. At each internal node the algorithm selects the feature and threshold that maximise information gain — the reduction in entropy after the split. Splitting continues recursively until the tree reaches the specified `max_depth` or no further gain is possible. Each leaf assigns the majority class of the training samples it contains.

The key regularisation parameter is `max_depth`. Shallow trees underfit (too few splits to capture the signal); deep trees overfit (too many splits memorise noise). Feature importances are derived directly from the weighted information gain each variable contributes across all nodes where it is used as the split criterion.

## Dataset

**UCI Adult Census Income** — 32,561 records from the 1994 US Census, after dropping rows with missing values. Six numeric features are used: age, final sampling weight (fnlwgt), education years (education-num), capital-gain, capital-loss, and hours-per-week. The binary target is whether annual income exceeds $50K. The dataset is class-imbalanced: approximately 75% of records fall in the ≤$50K class.

## What the Notebook Covers

- Training a decision tree with entropy criterion and evaluating with a confusion matrix and classification report
- Sweeping `max_depth` from 1 to unconstrained to observe the bias-variance trade-off (best at depth 12, test accuracy 0.8291; unconstrained tree: train 0.9990, test 0.7725)
- Feature importance bar chart showing capital-gain as the dominant predictor, followed by education-num and age
- Decision boundary visualisation in 2D via PCA projection, showing the axis-aligned, piecewise-rectangular structure of tree splits
