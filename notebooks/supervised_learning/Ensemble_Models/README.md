# Ensemble Methods — UCI Adult Census Income Dataset

## Algorithm

Ensemble methods combine multiple base learners to produce a stronger aggregate predictor.
Four variants are demonstrated alongside a single Decision Tree baseline:

- **Random Forest:** Trains B decision trees, each on a bootstrap sample and a random subset
  of features at every split. Predictions are made by majority vote. Random feature subsets
  decorrelate the trees, so averaging them reduces variance without increasing bias.
- **AdaBoost:** Trains decision stumps sequentially. Each round reweights training samples by
  exponentially upweighting the misclassified ones, forcing the next learner to focus on the
  hard examples. The final prediction is a weighted majority vote of all rounds.
- **Bagging:** Generic bootstrap aggregation — trains B copies of the same base estimator on
  independent bootstrap samples and aggregates by majority vote. Unlike Random Forest, no
  feature randomisation is applied at each split.
- **Voting:** Combines the predictions of multiple distinct classifiers (Decision Tree, Random
  Forest, AdaBoost, and Bagging) by majority vote, exploiting the diversity of their error
  patterns.

## Dataset

- **Source:** UCI Adult Census Income
- **Task:** Binary classification — predict whether annual income exceeds $50K
- **Samples:** 32,561 (after dropping rows with missing values)
- **Features:** 15 (continuous and categorical; categorical features are one-hot encoded)
- **Class balance:** 24,720 ≤50K / 7,841 >50K (~75% / ~25%, imbalanced)
- **Split:** 26,048 train / 6,513 test

## Results

| Model | Train Accuracy | Test Accuracy | vs. Baseline |
|---|---|---|---|
| Decision Tree (baseline) | 0.8372 | 0.8287 | — |
| **Random Forest** | **0.8408** | **0.8348** | **+0.0061** |
| Voting | 0.8506 | 0.8325 | +0.0038 |
| AdaBoost | 0.8307 | 0.8282 | -0.0005 |
| Bagging | 0.8298 | 0.8253 | -0.0034 |

### Random Forest Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| ≤50K (0) | 0.84 | 0.96 | 0.90 | 4,926 |
| >50K (1) | 0.79 | 0.44 | 0.56 | 1,587 |
| Weighted avg | 0.83 | 0.83 | 0.82 | 6,513 |

## Key Findings

**Performance and why:** Random Forest is the only ensemble that meaningfully outperforms the
single Decision Tree baseline (+0.0061 test accuracy). Its advantage comes from the
combination of bootstrap sampling and random feature subsets at each split: the two sources
of randomness together decorrelate the individual trees so that their errors do not overlap.
Averaging uncorrelated classifiers reduces prediction variance — the mechanism is the same as
reducing noise by averaging independent measurements — and the tight train-to-test gap (0.8408
vs. 0.8348) confirms that the model generalises well without memorising training-specific
patterns.

**Ensembles are not a guaranteed improvement:** AdaBoost (0.8282) and Bagging (0.8253) both
fall below the Decision Tree baseline (0.8287). Generic Bagging without feature randomisation
produces trees that remain correlated — each bootstrap sample still exposes the same dominant
features — so the variance-reduction benefit is limited. AdaBoost's sequential reweighting is
effective when misclassification errors are informative, but on this imbalanced dataset (75%
majority class) the errors concentrate on the rare >50K samples. The exponential upweighting
amplifies noise from those samples rather than correcting systematic model error, and the
ensemble's train accuracy (0.8307) falls below the baseline from the start, indicating the
weak learners in each round do not improve on the baseline's splits.

**Voting's diversity benefit:** The Voting classifier (+0.0038 over baseline) benefits from
combining four models that make different types of errors. However, it is bounded by its
weakest members — including the underperforming AdaBoost and Bagging — which is why it
outperforms the Decision Tree but not Random Forest alone.

**Class imbalance is the binding constraint:** All five models, including the best (Random
Forest), achieve recall of only 0.44–0.46 for the >50K minority class. Ensemble methods that
reduce variance do not address the fundamental imbalance: the majority class dominates the
loss landscape and the models learn conservative boundaries that sacrifice minority-class
recall to maximise overall accuracy. Addressing this would require class-weight balancing,
oversampling (SMOTE), or a threshold adjusted below 0.5.

**Strengths of ensemble architectures:** Random Forest and Voting provide modest but real
gains over a single tree with no additional hyperparameter tuning of the base learner. Random
Forest also offers feature importance scores that are more stable than those from a single
tree, since they are averaged across many bootstrap samples.

**Limitations grounded in these results:** The gains are small in absolute terms (≤0.006
accuracy points). AdaBoost and Bagging add computational cost without any benefit on this
dataset. All models share the same low minority-class recall, confirming that ensemble
diversity alone cannot compensate for structural data imbalance. The Voting classifier's
train accuracy (0.8506) is the highest of all five models, but its test accuracy (0.8325) is
third — the largest train-test gap among the ensembles — suggesting it is the most prone to
overfitting the training set by combining overfit-prone members.
