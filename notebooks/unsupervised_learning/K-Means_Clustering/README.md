# K-Means Clustering

## Model Architecture

K-Means (Lloyd's algorithm) partitions $n$ data points into $k$ clusters by alternating two steps until convergence:

1. **Assignment:** assign each point to the cluster of its nearest centroid (Euclidean distance)
2. **Update:** recompute each centroid as the mean of its assigned points

The objective minimised is within-cluster inertia (sum of squared distances to the nearest centroid). The algorithm converges to a local minimum; the global optimum is not guaranteed. The number of clusters $k$ must be specified in advance. The **elbow method** aids this choice by plotting inertia vs $k$ and looking for the point where additional clusters yield diminishing inertia reduction.

Feature standardisation is essential: K-Means uses Euclidean distance, so features with large raw variances dominate the distance calculation unless scaled.

## Dataset

**UCI Dry Bean** (`fetch_ucirepo(id=602)`) — 13,611 bean grain images described by 16 morphological features (area, perimeter, axis lengths, shape factors). Seven true variety labels (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA) are withheld during clustering and used only for evaluation via the Adjusted Rand Index. The known number of true classes makes this dataset ideal for validating an unsupervised method.

## What the Notebook Covers

- 2D PCA preview coloured by true variety before clustering
- Elbow plot sweeping $k$ from 1 to 12 (inertia from 217,776 at $k = 1$ to 38,911 at $k = 12$; natural elbow near $k = 7$)
- Fitting K-Means at $k = 7$ (54 iterations, inertia 53,273); cluster sizes: {2771, 2339, 2246, 2002, 1884, 1848, 521}
- 2D PCA scatter coloured by cluster assignment vs true variety
- Cluster-vs-variety contingency heatmap: BOMBAY perfectly isolated (521/521); DERMASON and SIRA partially overlapping
- Adjusted Rand Index: 0.5818
