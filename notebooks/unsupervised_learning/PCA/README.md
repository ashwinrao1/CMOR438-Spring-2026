# Principal Component Analysis

## Model Architecture

Principal Component Analysis (PCA) finds the directions of maximum variance in a dataset via eigendecomposition of the sample covariance matrix. The eigenvectors (principal components) form an orthonormal basis ordered by decreasing eigenvalue. Projecting onto the first $k$ components gives the optimal linear dimensionality reduction in the sense of minimising mean squared reconstruction error.

Given a mean-centred data matrix $X \in \mathbb{R}^{n \times d}$:

$$C = \frac{1}{n} X^\top X, \quad C \mathbf{v}_j = \lambda_j \mathbf{v}_j$$

The proportion of variance explained by component $j$ is $\lambda_j / \sum_i \lambda_i$. PCA must be fit on training data only; test data is transformed using the training-set principal axes. Feature standardisation is required when features have different units or scales — without it, high-variance features would dominate all principal components.

PCA is strictly linear. It discovers the best linear subspace, not non-linear manifolds.

## Dataset

**UCI Dry Bean** (`fetch_ucirepo(id=602)`) — 13,611 bean images, 16 morphological features, 7 variety classes. The features include highly correlated size measurements (area, perimeter, major axis length, minor axis length) and shape descriptors (compactness, roundness, eccentricity, solidity). The strong inter-feature correlations create significant redundancy, making this an ideal dataset for PCA.

## What the Notebook Covers

- Fitting PCA on the standardised training set and computing explained variance per component
- 2 components explain 80% of variance; 4 components explain 90% and 95%
- Scree plot and cumulative explained variance bar-and-line chart
- 2D PCA scatter of the training set coloured by true variety, showing visible cluster separation with BOMBAY fully isolated
- Reconstruction MSE curve: 0.4484 at $k = 1$, 0.1816 at $k = 2$, 0.0222 at $k = 5$, 0.0007 at $k = 8$, 0 at $k = 16$
- Component loadings for the first 8 principal axes, showing which morphological features each component weights most
