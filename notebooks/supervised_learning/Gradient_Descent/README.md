# Gradient Descent

## Algorithm Architecture

Gradient descent is a first-order iterative optimisation algorithm. At each step the current parameters move in the direction opposite to the gradient of the loss function, scaled by the learning rate $\eta$:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \, \nabla L(\mathbf{w})$$

The algorithm makes no assumption about the shape of the loss surface. On convex problems (such as mean squared error for linear regression) it is guaranteed to reach the global minimum. On non-convex problems the solution depends on initialisation — different starting points converge to different local minima.

The learning rate controls the speed-stability trade-off. Too small: convergence is slow. Too large: the update overshoots and the loss may oscillate or diverge.

## Datasets and Tasks

**Part 1 — 1D polynomial:** $f(x) = x^4 - 4x^3 + 4x$, a synthetic function with two local minima (near $x \approx 0.4$ and $x \approx 3$). Starting point $x_0 = 3.0$; converges to the deeper minimum at $x \approx 2.879$, $f(x^*) = -15.234$.

**Part 2 — 2D elliptical bowl:** $L(w_0, w_1) = w_0^2 + 5w_1^2$, a quadratic with known minimum at the origin. The elongated contours demonstrate the characteristic zig-zag descent path when curvatures differ across dimensions.

**Part 3 — Linear Regression on CCPP:** UCI Combined Cycle Power Plant dataset (9,568 hourly measurements, 4 features, target: net electrical output PE in MW). Gradient descent is compared against the closed-form OLS solution; both converge to slope $= -16.1688$, intercept $= 454.2477$.

## What the Notebook Covers

- Trajectory plot of GD on the 1D polynomial from $x_0 = 3.0$, showing convergence to the deeper basin
- Learning rate sweep ($\eta \in \{0.001, 0.01, 0.05, 0.1\}$) showing speed-stability trade-offs
- Contour plot with the optimisation path on the 2D bowl, illustrating zig-zag behaviour due to anisotropic curvature
- GD vs OLS regression line comparison on CCPP data, with MSE convergence curve
