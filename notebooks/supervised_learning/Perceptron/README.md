# Perceptron

## Model Architecture

The Rosenblatt perceptron is the simplest linear classifier. It predicts the sign of a weighted dot product:

$$\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b)$$

where class labels are mapped to $\{-1, +1\}$. The weight update rule fires only on incorrect predictions:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta \, y_i \, \mathbf{x}_i, \quad b \leftarrow b + \eta \, y_i$$

The Perceptron Convergence Theorem guarantees finite convergence if and only if the data are linearly separable. On non-separable data the algorithm cycles indefinitely; the solution at termination depends on the epoch budget. There is no loss function being minimised — only an error-correction rule.

A key theoretical property demonstrated here: the learning rate $\eta$ scales the weight magnitude but does not change the decision hyperplane direction. All learning rates produce the same boundary up to a scalar multiple.

## Dataset

**UCI Bank Marketing** (`fetch_ucirepo(id=222)`) — 5,000 rows subsampled for computational tractability (the perceptron processes one sample per inner-loop iteration in pure Python). 37 features after one-hot encoding; binary subscription outcome with approximately 12% positive rate.

## What the Notebook Covers

- Training a perceptron (train accuracy 0.8698, test accuracy 0.8590) and evaluating with a confusion matrix
- Learning curve across epoch counts 1–400 (log scale) showing early plateau and no further convergence
- Learning rate sweep ($\eta \in \{0.001, 0.01, 0.1, 1.0\}$): test accuracy identical at 0.8590 for all rates; weight norm scales proportionally (0.027, 0.271, 2.707, 27.066)
- 2D PCA decision boundary (test accuracy 0.7850 vs full-space 0.8590)
