"""
Multilayer Perceptron

A fully connected neural network trained by backpropagation and batch gradient
descent. Hidden layers use ReLU activations; the output layer uses sigmoid for
binary classification or a linear activation for regression.

Weights are initialized with He initialization to keep activations in a stable
range at the start of training. L2 regularization penalizes all weight matrices
but not bias vectors. An intercept (bias) is maintained per layer and is never
regularized.

The network architecture is specified as a list of hidden layer widths at
construction time. A single output unit is always appended. Forward and
backward passes operate on the full batch, so memory scales with dataset size.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Sequence

import numpy as np

from ._utils import _as2d_float, _as1d, _sigmoid

ArrayLike = Union[np.ndarray, Sequence]

__all__ = ["MLPClassifier", "MLPRegressor"]


# ---------------------------------------------------------------------------
# MLP-specific activations (not shared — live here only)
# ---------------------------------------------------------------------------

def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _MLPBase(ABC):
    """
    Shared forward/backward-pass logic for MLPClassifier and MLPRegressor.

    Subclasses must implement ``_output_activation`` and ``_output_grad``
    to define how the output layer is activated and how its loss gradient
    is computed.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (64, 32),
        *,
        alpha: float = 0.0,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: float | None = 1e-6,
        random_state: int | None = None,
    ) -> None:
        if any(h < 1 for h in hidden_layer_sizes):
            raise ValueError("All hidden layer sizes must be >= 1.")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}.")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}.")
        self.hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        self.alpha = float(alpha)
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.tol = tol
        self.random_state = random_state
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []

    def _init_params(self, n_features: int, layer_sizes: Sequence[int]) -> None:
        rng = np.random.default_rng(self.random_state)
        dims = [n_features, *layer_sizes]
        self._weights = []
        self._biases = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            # He initialization
            std = np.sqrt(2.0 / fan_in)
            self._weights.append(rng.normal(0.0, std, (fan_in, fan_out)))
            self._biases.append(np.zeros(fan_out))

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Run a forward pass through the network.

        Returns pre-activations (z) and activations (a) for every layer,
        including the input (a[0] = X).
        """
        zs, activations = [], [X]
        a = X
        for i, (W, b) in enumerate(zip(self._weights, self._biases)):
            z = a @ W + b
            zs.append(z)
            is_output = i == len(self._weights) - 1
            a = self._output_activation(z) if is_output else _relu(z)
            activations.append(a)
        return zs, activations

    @abstractmethod
    def _output_activation(self, z: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _output_grad(self, a_out: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient of loss w.r.t. pre-activation at the output layer."""
        ...

    def _backward(
        self,
        zs: list[np.ndarray],
        activations: list[np.ndarray],
        y: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n = y.shape[0]
        grad_W = [np.zeros_like(W) for W in self._weights]
        grad_b = [np.zeros_like(b) for b in self._biases]

        delta = self._output_grad(activations[-1], y)

        for i in reversed(range(len(self._weights))):
            a_prev = activations[i]
            grad_W[i] = (a_prev.T @ delta) / n
            grad_b[i] = delta.mean(axis=0)
            if self.alpha > 0.0:
                grad_W[i] += self.alpha * self._weights[i]
            if i > 0:
                delta = (delta @ self._weights[i].T) * _relu_grad(zs[i - 1])

        return grad_W, grad_b

    def _update(self, grad_W: list[np.ndarray], grad_b: list[np.ndarray]) -> None:
        for i in range(len(self._weights)):
            self._weights[i] -= self.learning_rate * grad_W[i]
            self._biases[i] -= self.learning_rate * grad_b[i]

    def _gradient_norm(self, grad_W: list[np.ndarray], grad_b: list[np.ndarray]) -> float:
        all_grads = [g.ravel() for g in grad_W] + [g.ravel() for g in grad_b]
        return float(np.linalg.norm(np.concatenate(all_grads)))

    def _fit(self, X: np.ndarray, y: np.ndarray, layer_sizes: Sequence[int]) -> None:
        self._init_params(X.shape[1], layer_sizes)
        for _ in range(self.n_iterations):
            zs, activations = self._forward(X)
            grad_W, grad_b = self._backward(zs, activations, y)
            self._update(grad_W, grad_b)
            if self.tol is not None and self._gradient_norm(grad_W, grad_b) < self.tol:
                break

    def _check_fitted(self) -> None:
        if not self._weights:
            raise RuntimeError("Call fit before using this method.")

    def _check_feature_count(self, X: np.ndarray) -> None:
        expected = self._weights[0].shape[0]
        if X.shape[1] != expected:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was "
                f"fitted on {expected} features."
            )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MLPClassifier(_MLPBase):
    """
    MLP binary classifier with sigmoid output and binary cross-entropy loss.

    Parameters
    ----------
    hidden_layer_sizes : sequence of int
        Width of each hidden layer, in order from input to output.
    alpha : float
        L2 regularization strength applied to weight matrices. Biases are
        never regularized.
    learning_rate : float
        Gradient descent step size.
    n_iterations : int
        Maximum number of gradient descent steps.
    tol : float or None
        Gradient norm convergence threshold. None disables early stopping.
    threshold : float
        Probability threshold for converting output to a class label.
    random_state : int or None
        Seed for weight initialization.
    """

    def __init__(
        self,
        hidden_layer_sizes: Sequence[int] = (64, 32),
        *,
        alpha: float = 0.0,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        tol: float | None = 1e-6,
        threshold: float = 0.5,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            hidden_layer_sizes,
            alpha=alpha,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            tol=tol,
            random_state=random_state,
        )
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
        self.threshold = float(threshold)

    def _output_activation(self, z: np.ndarray) -> np.ndarray:
        return _sigmoid(z)

    def _output_grad(self, a_out: np.ndarray, y: np.ndarray) -> np.ndarray:
        # gradient of BCE loss w.r.t. z_out equals (a_out - y) for sigmoid output
        return (a_out - y[:, None])

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPClassifier":
        """
        Fit the classifier to binary training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels. Must contain exactly two unique values; they are
            mapped internally to {0, 1}.

        Returns
        -------
        self : MLPClassifier

        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        >>> y = np.array([0, 1, 1, 0])   # XOR
        >>> clf = MLPClassifier(hidden_layer_sizes=(4,), n_iterations=2000,
        ...                     random_state=0)
        >>> clf.fit(X, y).score(X, y) >= 0.75
        True
        """
        X = _as2d_float(X, "X")
        y = _as1d(y, "y")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}."
            )
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError(
                f"MLPClassifier requires exactly 2 classes, "
                f"got {classes.shape[0]}: {classes}."
            )
        self.classes_ = classes
        y_bin = (y == classes[1]).astype(float)
        self._fit(X, y_bin, [*self.hidden_layer_sizes, 1])
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Return class probability estimates for each sample.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Columns are [P(negative class), P(positive class)].
        """
        self._check_fitted()
        X = _as2d_float(X, "X")
        self._check_feature_count(X)
        _, activations = self._forward(X)
        p_pos = activations[-1].ravel()
        return np.column_stack([1.0 - p_pos, p_pos])

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the predicted class label for each sample."""
        p_pos = self.predict_proba(X)[:, 1]
        return self.classes_[(p_pos >= self.threshold).astype(int)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == _as1d(y, "y")))


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------

class MLPRegressor(_MLPBase):
    """
    MLP regressor with linear output and mean squared error loss.

    Parameters
    ----------
    hidden_layer_sizes : sequence of int
        Width of each hidden layer, in order from input to output.
    alpha : float
        L2 regularization strength applied to weight matrices. Biases are
        never regularized.
    learning_rate : float
        Gradient descent step size.
    n_iterations : int
        Maximum number of gradient descent steps.
    tol : float or None
        Gradient norm convergence threshold. None disables early stopping.
    random_state : int or None
        Seed for weight initialization.
    """

    def _output_activation(self, z: np.ndarray) -> np.ndarray:
        return z

    def _output_grad(self, a_out: np.ndarray, y: np.ndarray) -> np.ndarray:
        # gradient of MSE loss w.r.t. z_out equals (a_out - y) for linear output
        return (a_out - y[:, None])

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPRegressor":
        """
        Fit the regressor to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : MLPRegressor

        Examples
        --------
        >>> import numpy as np
        >>> X = np.linspace(0, 1, 20).reshape(-1, 1)
        >>> y = (2 * X.ravel() + 1)
        >>> reg = MLPRegressor(hidden_layer_sizes=(8,), n_iterations=3000,
        ...                    learning_rate=0.05, random_state=0)
        >>> reg.fit(X, y).score(X, y) > 0.95
        True
        """
        X = _as2d_float(X, "X")
        y = _as1d(y, "y").astype(float)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}."
            )
        self._fit(X, y, [*self.hidden_layer_sizes, 1])
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return predicted target values for each sample."""
        self._check_fitted()
        X = _as2d_float(X, "X")
        self._check_feature_count(X)
        _, activations = self._forward(X)
        return activations[-1].ravel()

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the R² coefficient of determination on (X, y)."""
        y_true = _as1d(y, "y").astype(float)
        y_pred = self.predict(X)
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        if ss_tot == 0.0:
            raise ValueError("R² is undefined when y is constant.")
        return float(1.0 - ((y_true - y_pred) ** 2).sum() / ss_tot)
