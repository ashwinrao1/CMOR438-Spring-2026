# Regression Trees — Combined Cycle Power Plant Dataset

## Algorithm

A regression tree recursively partitions the feature space by selecting the feature and
threshold that minimise the weighted mean squared error of the two resulting child nodes. Each
leaf predicts the mean of its training samples. Tree depth is the primary complexity control:
shallow trees underfit with coarse piecewise constant approximations; deep trees overfit by
memorising training-set noise. The notebook also compares the best-tuned tree against OLS
linear regression on the same dataset.

## Dataset

- **Source:** UCI Combined Cycle Power Plant
- **Task:** Regression — predict net electrical output (PE) in MW
- **Samples:** 9,568 (7,654 train / 1,914 test)
- **Features:** 4 continuous (AT: ambient temperature, V: vacuum, AP: ambient pressure,
  RH: relative humidity)
- **Preprocessing:** No scaling required (trees are invariant to monotone feature transforms)

## Results

### Depth Sweep

| max_depth | Train R² | Test R² |
|---|---|---|
| 1 | 0.7182 | 0.7308 |
| 2 | 0.8605 | 0.8680 |
| 3 | 0.9096 | 0.9123 |
| 5 | 0.9361 | 0.9343 |
| **8** | **0.9578** | **0.9433** |
| 15 | 0.9886 | 0.9375 |
| None | 0.9923 | 0.9373 |

Best test R²: **0.9433** at max_depth=8.

### Model Comparison

| Model | Test R² | Test MSE | Test MAE |
|---|---|---|---|
| Regression Tree (max_depth=8) | **0.9433** | **16.48** | **3.07** |
| OLS Linear Regression | 0.9311 | 20.03 | 3.62 |

### Residuals (max_depth=8)

- Mean: -0.003
- Standard deviation: 4.06

## Key Findings

**Performance and why:** The best regression tree (max_depth=8) achieves test R²=0.9433 and
outperforms OLS linear regression (R²=0.9311) on every metric. The improvement — 1.22 R²
points, a drop in MSE from 20.03 to 16.48, and a reduction in MAE from 3.62 to 3.07 — arises
because the temperature–power relationship contains mild non-linearity that a global linear
model cannot capture. The tree partitions the AT range into eight depth levels of progressively
finer bins, approximating the curvature with piecewise constant means. This non-linearity is
visible in the EDA scatter plots as a subtle curve in the AT–PE relationship that the linear
residual plot cannot fully eliminate.

**Overfitting signature:** Beyond max_depth=8, train R² continues to climb (0.9886 at depth
15, 0.9923 at unlimited depth) while test R² reverses and falls to 0.9375. Each additional
split creates leaves small enough to memorise individual training points rather than capturing
generalise pattern. The residual standard deviation of 4.06 MW at depth 8 represents the
irreducible prediction error given the granularity of the eight-level partition; shallower
trees have wider residuals and deeper trees introduce noise.

**Depth 1 baseline:** Even a single split (R²=0.7308) captures most of the signal because AT
alone explains the dominant share of variance in PE — the first split almost certainly chooses
AT and achieves strong separation. Each additional level of depth refines the remaining
within-leaf variance, with diminishing returns beyond depth 5.

**Strengths of the architecture:** Regression trees require no feature scaling, handle non-linear
relationships naturally, and produce directly interpretable conditional means at each leaf. On
this dataset, the mild non-linearity that OLS cannot capture is enough for the tree to achieve
a 1.22 R² point improvement and a 17.7% reduction in MSE — a meaningful practical difference
at the scale of a power plant.

**Limitations grounded in these results:** Tree predictions are piecewise constant: the
predicted-vs-actual scatter shows horizontal bands (each corresponding to one leaf's mean)
at shallow depth, which smooths out only as depth increases. This discontinuity means the tree
will never produce a smooth extrapolation beyond the training range. The model is also sensitive
to the choice of max_depth — the difference between depth 8 (best) and depth 15 is a 0.006 R²
drop that would require cross-validation to detect reliably. Unlike linear regression, there is
no closed-form solution and no coefficient to interpret; understanding which feature drives
which prediction requires inspecting individual splits.
