# Multilayer Perceptron

## Model Architecture

A multilayer perceptron (MLP) is a feedforward neural network of fully connected layers. Each hidden layer applies an affine transformation followed by a ReLU activation:

$$\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}, \quad \mathbf{a}^{(\ell)} = \max(0,\, \mathbf{z}^{(\ell)})$$

The output layer uses a sigmoid for binary classification or a linear activation for regression. Parameters are learned via backpropagation: the chain rule propagates loss gradients from the output backward through each layer, and weights are updated by gradient descent. The non-linear activations allow the network to learn arbitrary decision boundaries, unlike linear models.

Target standardisation is essential for regression: scaling the target to zero mean and unit variance keeps gradient magnitudes numerically stable during backpropagation.

## Datasets and Tasks

**Classification — UCI Dry Bean (binary):** 4,564 beans filtered to two morphologically similar elongated varieties, SIRA (positive class) and HOROZ (negative class). 16 morphological features. All features standardised.

**Regression — UCI CCPP:** 9,568 hourly measurements from a gas turbine power plant. 4 features (AT, V, AP, RH); target is net electrical output (PE, MW). Both features and target are standardised before training; predictions are inverse-transformed back to MW for evaluation.

## What the Notebook Covers

- Binary classification with architecture (64, 32): train accuracy 0.9996, test accuracy 1.0000 (perfect, confirmed by confusion matrix)
- Architecture sweep — (32,), (64, 32), (128, 64, 32) — all achieve test accuracy ≥ 0.9996
- Regression MLP: train $R^2 = 0.9357$, test $R^2 = 0.9382$, MSE = 17.97 MW$^2$ (outperforms OLS at $R^2 = 0.9311$)
- Predicted-vs-actual scatter for the regression task showing tight diagonal clustering
