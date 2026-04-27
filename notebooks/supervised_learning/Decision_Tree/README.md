# Decision Tree — Adult Income Dataset

## Algorithm

A decision tree classifier recursively partitions the feature space by selecting the feature
and threshold that maximise Gini impurity reduction at each node. Each leaf assigns the
majority class label of its training samples. Tree depth is the primary hyperparameter
controlling model complexity.

## Dataset

- **Source:** UCI Adult Income (census data)
- **Task:** Binary classification — predict whether annual income exceeds $50K
- **Samples:** 32,561 after dropping rows with missing values
- **Features:** 15 (mix of continuous and categorical; categorical features are one-hot encoded)
- **Class balance:** ~75% ≤50K, ~25% >50K (imbalanced)
- **Split:** 26,048 train / 6,513 test

## Results

| max_depth | Train Accuracy | Test Accuracy |
|---|---|---|
| 1 | 0.8017 | 0.7979 |
| 2 | 0.8017 | 0.7979 |
| 3 | 0.8017 | 0.7979 |
| 5 | 0.8291 | 0.8245 |
| 8 | 0.8372 | 0.8287 |
| **12** | **0.8484** | **0.8291** |
| None | 0.9990 | 0.7725 |

Best test accuracy: **0.8291** at max_depth=12.

### Classification Report (max_depth=12)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ≤50K (0) | 0.84 | 0.95 | 0.89 | 4,926 |
| >50K (1) | 0.74 | 0.46 | 0.57 | 1,587 |
| Weighted avg | 0.82 | 0.83 | 0.81 | 6,513 |

## Key Findings

**Performance and why:** The tree achieves 82.9% test accuracy by learning axis-aligned splits
on the 15 income-related features. Depths 1–3 all produce the same accuracy (0.7979) because
the first several splits converge on the same high-gain feature; meaningful separation of
additional subgroups only begins at depth 5. The unlimited-depth tree memorises the training
set (99.9% accuracy) but drops to 77.3% test accuracy — the clearest demonstration in this
notebook of a model overfitting by fitting noise rather than signal.

**Feature importance:** `capital-gain` is the dominant feature by Gini importance. This makes
sense: large capital gains are rare but nearly exclusive to high earners, providing an extremely
high-purity split at or near the root. `education-num` and `age` contribute the next largest
shares, consistent with the well-established labour-market relationship between attainment,
experience, and income.

**Strengths of the architecture:** Decision trees naturally handle mixed feature types, require
no feature scaling, and produce directly interpretable rules. The piecewise rectangular decision
boundary in the PCA projection shows the model learning conditional thresholds rather than a
global linear separator, which suits a dataset where income jumps at categorical boundaries
(e.g. a professional degree vs. high school).

**Limitations grounded in these results:** The low recall for the minority class (0.46) reveals
a core weakness: trees trained with Gini impurity on imbalanced data are biased toward the
majority class — a wrong prediction on a ≤50K sample costs less impurity than a wrong prediction
on a >50K sample. Without class-weight balancing or pruning, the tree sacrifices recall for
the class that matters most in a targeted-assistance context. The depth sweep also illustrates
that the model has no built-in regularisation; depth must be tuned explicitly or test accuracy
collapses.
