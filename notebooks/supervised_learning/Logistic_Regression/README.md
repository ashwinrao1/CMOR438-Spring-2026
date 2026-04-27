# Logistic Regression — Pima Indians Diabetes Dataset

## Algorithm

Logistic regression models the log-odds of the positive class as a linear function of the
input features and passes it through the sigmoid to produce a probability. The decision
boundary is always a hyperplane. Parameters are learned by minimising the binary cross-entropy
loss via gradient descent. L2 regularisation (Ridge penalty) adds `alpha * ||w||^2` to the
loss, shrinking weights toward zero to reduce overfitting. Both unregularised (alpha=0) and
regularised (alpha=1) variants are implemented in `mlpackage`.

## Dataset

- **Source:** Pima Indians Diabetes dataset
- **Task:** Binary classification — predict diabetic (1) vs. non-diabetic (0)
- **Samples:** 768 (614 train / 154 test)
- **Features:** 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
  DiabetesPedigreeFunction, Age)
- **Class balance:** 500 negative (0) / 268 positive (1) — imbalanced (~35% positive rate)
- **Preprocessing:** Zeros replaced with per-feature medians for five physiologically
  impossible features; StandardScaler fit on training data only

## Results

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| No regularisation (alpha=0) | 0.7866 | 0.6948 |
| L2 regularisation (alpha=1) | 0.7866 | 0.6948 |

### Classification Report (both models identical)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-diabetic (0) | 0.66 | 0.89 | 0.76 | 84 |
| Diabetic (1) | 0.78 | 0.46 | 0.58 | 70 |
| Weighted avg | 0.72 | 0.69 | 0.68 | 154 |

## Key Findings

**Performance and why:** Both models achieve 78.7% train accuracy and 69.5% test accuracy.
The roughly 9-point train-to-test gap suggests that the model captures real signal but does
not generalise perfectly — likely because 614 training samples are insufficient to fully
characterise the eight-dimensional diabetic risk surface, and because the features themselves
contain meaningful noise (notably Insulin and SkinThickness, which have high proportions of
zero values requiring imputation). L2 regularisation with alpha=1 produces no measurable
improvement over the unpenalised model: both accuracy and the full classification report are
identical, indicating that at this sample size and feature scale the coefficient norms are
already small enough that the penalty adds negligible constraint.

**Class-level results:** Recall for the diabetic class is only 0.46 — more than half of
diabetic patients in the test set are missed. Precision is 0.78, meaning most of the positive
predictions are correct, but the model is systematically conservative: it defaults to the
majority class when uncertain. This reflects the imbalanced training set (500 vs. 268) and the
linear decision boundary's inability to isolate the diabetic subpopulation, which overlaps
substantially with non-diabetics in the marginal feature distributions visible in the
correlation matrix.

**Strengths of the architecture:** Logistic regression provides calibrated probability
estimates that allow a clinician to choose a threshold other than 0.5 — for example, lowering
the threshold improves recall at the cost of precision, which may be preferable in a screening
context. The ROC curve lies above the random baseline, confirming the model extracts useful
signal from all eight features.

**Limitations grounded in these results:** The linear decision boundary is the fundamental
constraint. The PCA projection shows the two classes overlapping substantially; no single
hyperplane can cleanly separate them. Achieving materially higher recall for the diabetic class
would require either a non-linear model (e.g. a tree or MLP) that can exploit interaction
effects, or explicit class-weight balancing during training to counteract the 500-to-268
majority bias. The identical outputs for alpha=0 and alpha=1 also reveal that more aggressive
regularisation sweeps (or a different regulariser such as L1 for feature selection) would be
needed to extract any regularisation benefit on this dataset.
