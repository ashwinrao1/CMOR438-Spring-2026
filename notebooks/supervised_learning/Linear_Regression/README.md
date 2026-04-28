# Linear Regression (OLS and Ridge)

## Model Architecture

**Ordinary Least Squares (OLS)** finds the coefficient vector $\boldsymbol{\beta}$ that minimises the sum of squared residuals. The closed-form solution is the normal equation:

$$\boldsymbol{\beta} = (X^\top X)^{-1} X^\top y$$

This is an exact, non-iterative solution with no hyperparameters. Under the Gauss-Markov assumptions it is the minimum-variance unbiased estimator.

**Ridge Regression** adds an $\ell_2$ penalty $\alpha \|\boldsymbol{\beta}\|^2$ to the objective:

$$\boldsymbol{\beta}_{\text{ridge}} = (X^\top X + \alpha I)^{-1} X^\top y$$

The penalty shrinks all coefficients toward zero, trading a small amount of bias for reduced variance. As $\alpha \to 0$ the solution approaches OLS; as $\alpha \to \infty$ all coefficients approach zero. Both models assume a linear relationship between features and the target.

## Dataset

**UCI Combined Cycle Power Plant** (`fetch_ucirepo(id=294)`) — 9,568 hourly measurements from a gas turbine power plant. Four continuous features: ambient temperature (AT, °C), exhaust vacuum (V, cm Hg), ambient pressure (AP, mbar), and relative humidity (RH, %). Target: net electrical energy output (PE, MW), ranging from 420 to 496 MW. No missing values.

## What the Notebook Covers

- Fitting OLS, Ridge ($\alpha = 0.1$), and Ridge ($\alpha = 100$); comparing train/test $R^2$, MSE, and MAE (OLS: test $R^2 = 0.9311$, MSE = 20.03, MAE = 3.62)
- Coefficient shrinkage bar chart: Ridge $\alpha = 100$ visibly compresses all four weights; OLS and Ridge $\alpha = 0.1$ are numerically identical
- Predicted-vs-actual scatter plots for all three models
- Residual plot for OLS showing homoscedastic, zero-centred errors
