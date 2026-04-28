# Logistic Regression

## Model Architecture

Logistic regression models the probability that a binary outcome is positive as the sigmoid of a linear function of the features:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

Parameters are learned by minimising binary cross-entropy loss, optionally with an $\ell_2$ penalty $\alpha \|\mathbf{w}\|^2$. The loss is convex, so gradient descent is guaranteed to find the global optimum. The decision boundary is always a single hyperplane in feature space.

Each coefficient is interpretable as the log-odds change associated with a one-unit increase in the corresponding feature, making the model well-suited to problems where class probabilities can be approximated as a linear function of the inputs.

## Dataset

**UCI Bank Marketing** (`fetch_ucirepo(id=222)`) — telephone marketing campaign data from a Portuguese bank. 10,000 rows are subsampled for tractability. Features include demographic information (age, job, marital status, education), financial indicators (balance, housing loan, personal loan), and prior contact history. The `duration` column is dropped to prevent data leakage. Target: whether the client subscribed to a term deposit. The positive rate is approximately 12%, creating a significant class imbalance.

## What the Notebook Covers

- One-hot encoding 37 features and standardising; training unregularised ($\alpha = 0$) and L2-regularised ($\alpha = 1$) models
- Both models: train accuracy 0.8900, test accuracy 0.8960; positive-class recall only 0.19 (F1 = 0.30)
- Confusion matrices and classification reports revealing the class-imbalance effect
- ROC curves and AUC for both models
- Top-15 coefficient bar chart identifying the strongest subscription predictors
- 2D PCA decision boundary visualisation (test accuracy 0.8865)
