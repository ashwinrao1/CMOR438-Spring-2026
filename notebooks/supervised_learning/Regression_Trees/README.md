# Regression Trees

## Model Architecture

A regression tree recursively partitions the feature space into axis-aligned rectangular regions. At each node the algorithm selects the feature and threshold that maximise variance reduction:

$$\text{Gain} = \text{Var}(\text{parent}) - \frac{n_L}{n}\,\text{Var}(L) - \frac{n_R}{n}\,\text{Var}(R)$$

Each leaf predicts the mean target value of the training samples it contains. The result is a piecewise constant approximation to the true regression function. `max_depth` is the primary regularisation parameter: shallow trees produce coarse, high-bias approximations; deep trees memorise training noise. Feature standardisation is not required — regression trees are invariant to monotone transformations of the features.

## Dataset

**UCI Combined Cycle Power Plant** (`fetch_ucirepo(id=294)`) — 9,568 hourly measurements. Four features: ambient temperature (AT), exhaust vacuum (V), ambient pressure (AP), relative humidity (RH). Target: net electrical output (PE, MW). The AT–PE relationship contains mild non-linear curvature that makes this dataset a useful comparison between tree-based and linear regression.

## What the Notebook Covers

- Training a regression tree at depth 5 as a baseline (test $R^2 = 0.9343$, MSE = 19.08, MAE = 3.39)
- Depth sweep from 1 to unconstrained: best at depth 8 (test $R^2 = 0.9433$, MSE = 16.48, MAE = 3.07); at unconstrained depth, train $R^2 = 0.9923$, test $R^2 = 0.9373$
- Side-by-side comparison table: regression tree (depth 8) vs OLS linear regression (test $R^2 = 0.9311$)
- Predicted-vs-actual scatter and residual histogram (mean $-0.003$, std $4.06$ MW) for the best tree
