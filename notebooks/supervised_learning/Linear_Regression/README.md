# Linear Regression — Combined Cycle Power Plant Dataset

## Algorithm

Ordinary Least Squares (OLS) linear regression finds the closed-form solution
`w = (X'X)^{-1} X'y` that minimises the sum of squared residuals. Ridge regression adds an
L2 penalty `alpha * ||w||^2` to the loss, shrinking coefficients toward zero in proportion
to `alpha` and trading a small amount of bias for reduced variance. Both are implemented
from scratch in `mlpackage`.

## Dataset

- **Source:** UCI Combined Cycle Power Plant
- **Task:** Regression — predict net electrical output (PE) in MW
- **Samples:** 9,568
- **Features:** 4 continuous (AT: ambient temperature, V: vacuum, AP: ambient pressure, RH: humidity)
- **Split:** 7,654 train / 1,914 test
- **Preprocessing:** StandardScaler fit on training data only

## Results

| Model | Train R² | Test R² | Test MSE | Test MAE |
|---|---|---|---|---|
| OLS | 0.9281 | 0.9311 | 20.03 | 3.62 |
| Ridge alpha=0.1 | 0.9281 | 0.9311 | 20.03 | 3.62 |
| Ridge alpha=100 | 0.9276 | 0.9298 | 20.39 | 3.65 |

## Key Findings

**Performance and why:** OLS achieves R²=0.9311 on the test set, indicating that ambient
temperature, vacuum, pressure, and humidity together explain over 93% of the variance in
power output. The near-linear relationship between these thermodynamic variables and turbine
efficiency is physically motivated: gas turbine output is governed by the Carnot cycle, in
which ambient conditions enter the efficiency formula approximately linearly over the observed
operating range. The predicted-vs-actual scatter confirms this: both train and test predictions
cluster tightly around the y=x diagonal with no systematic curvature.

**Regularisation effect:** Ridge with alpha=0.1 is numerically indistinguishable from OLS
(R² identical to four decimal places). At this scale the penalty is negligible relative to the
magnitude of X'X, so coefficient shrinkage is imperceptible. Heavy regularisation at alpha=100
compresses all four coefficients substantially — visible in the coefficient shrinkage plot —
and introduces enough bias to lower test R² from 0.9311 to 0.9298 and raise MSE from 20.03
to 20.39. This is the bias–variance trade-off in direct numerical form: the penalty that would
protect against overfitting on a noisier dataset simply adds bias here because the OLS solution
is already stable.

**Feature interpretation:** AT carries the largest negative coefficient across all three models.
Higher ambient temperature reduces net output because warmer inlet air is less dense, reducing
the mass flow rate through the turbine and lowering thermodynamic efficiency. This direction is
consistent with physics and confirms the model has learned meaningful signal rather than
spurious correlations.

**Strengths of the architecture:** Linear regression provides a highly interpretable, closed-form
solution that generalises well when the true relationship is approximately linear. On this
dataset the linearity assumption holds strongly, yielding R²>0.93 with only four features and
no tuning.

**Limitations grounded in these results:** The residual plot shows roughly constant spread
(homoscedasticity), which validates the linear assumption here, but the regression tree
notebook demonstrates that a tuned tree achieves R²=0.9433 on the same data — a measurable
gap that linear regression cannot close because it cannot represent the mild non-linearity in
the AT–PE relationship. Linear regression also provides no mechanism to identify or handle
feature interactions; any synergistic effect between, for example, temperature and humidity
is averaged out rather than modelled explicitly.
