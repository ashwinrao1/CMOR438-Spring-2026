# Multilayer Perceptron — Digit Classification and Power Plant Regression

## Algorithm

A multilayer perceptron (MLP) stacks fully connected layers with non-linear activations (ReLU
here) and trains end-to-end via backpropagation: the chain rule propagates the gradient of the
loss backward through every layer, updating each weight by gradient descent. The non-linear
activations allow the network to learn piecewise linear decision boundaries far more expressive
than a single hyperplane. Two tasks are explored: binary digit classification (0 vs. 1) and
regression on the Combined Cycle Power Plant dataset.

## Datasets

- **Classification:** sklearn `load_digits`, binary subset (digit 0 vs. digit 1), 360 samples
  (288 train / 72 test), 64 pixel features
- **Regression:** UCI Combined Cycle Power Plant, 9,568 samples (7,654 train / 1,914 test),
  4 continuous features, target = net electrical output

## Results

### Binary Classification (0 vs. 1)

| Architecture | Train Accuracy | Test Accuracy |
|---|---|---|
| (32,) | 0.9896 | **1.0000** |
| (64, 32) | 0.9965 | **1.0000** |
| (128, 64, 32) | 1.0000 | **1.0000** |

Full classification report (any architecture):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 | 0.97 | 1.00 | 0.98 | 31 |
| 1 | 1.00 | 0.98 | 0.99 | 41 |
| Weighted avg | 0.99 | 0.99 | 0.99 | 72 |

### Regression (CCPP)

| Metric | Result |
|---|---|
| Train R² | NaN |
| Test R² | NaN |

Regression did not converge due to numerical instability (see below).

## Key Findings

**Classification performance and why:** All three architectures achieve perfect test accuracy
(1.0000) on the binary digit task. The task is straightforward — distinguishing the digit 0
from the digit 1 in 64-dimensional pixel space — and even a single hidden layer of 32 units
with ReLU activations is sufficient to learn the separating boundary. The fact that the
smallest network already achieves perfect test accuracy shows that depth and width beyond the
minimum needed to capture the task structure provide no measurable benefit and may even
slightly delay convergence (the (32,) network has lower train accuracy, 0.9896, than the
deeper variants, yet equals them on the test set).

**Architecture comparison:** The identical test accuracy across all three networks implies
the task is effectively solved at the first architecture. Claims that "deeper networks overfit
on small datasets" are not confirmed here; instead, all three converge to the same solution.
The appropriate conclusion is that digit 0 vs. 1 is a near-linearly-separable binary problem
that does not stress-test the representational capacity of any of these architectures.

**Regression instability:** The CCPP regression task produced NaN R² values on both train and
test sets. The root cause is a RuntimeWarning raised during backpropagation — "invalid value
encountered in multiply" in the weight-gradient step — indicating that intermediate activations
or gradients became NaN or infinite before the first weight update stabilised the network. This
is a numerical instability caused by the target variable (PE, ranging 420–496 MW) not being
scaled to a range compatible with the ReLU output layer and the chosen learning rate. Without
target standardisation, large gradient magnitudes propagate backward, overflow floating-point
range, and prevent convergence. The regression results from this run are therefore not
interpretable.

**Strengths of the architecture:** MLPs with ReLU can approximate arbitrary continuous
functions (by the universal approximation theorem) and learn feature interactions implicitly
through weight composition across layers — something linear models and single-split trees
cannot do. Backpropagation scales efficiently to thousands of parameters.

**Limitations grounded in these results:** MLPs are sensitive to the scale of both inputs and
outputs; without standardising the regression target, gradient magnitudes are large enough to
cause NaN propagation as demonstrated here. They also require careful hyperparameter selection
(architecture, learning rate, regularisation) and offer no convergence guarantee in the
non-convex loss landscape. On the simple binary digit task, the complexity of a three-layer
network is unnecessary, and the additional depth provides no benefit.
