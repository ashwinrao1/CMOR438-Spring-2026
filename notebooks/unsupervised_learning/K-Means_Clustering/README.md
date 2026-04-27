# K-Means Clustering — Mall Customer Segmentation

## Algorithm

K-Means partitions n samples into k non-overlapping clusters by alternating between two steps:
(1) assign each point to the nearest centroid by Euclidean distance, and (2) recompute each
centroid as the mean of its assigned points. The algorithm converges when assignments stop
changing. The number of clusters k must be specified in advance; the elbow method guides this
choice by plotting total inertia (within-cluster sum of squared distances) against k and
identifying where marginal inertia reduction diminishes sharply.

## Dataset

- **Source:** Mall Customer Segmentation dataset
- **Task:** Unsupervised clustering — discover natural customer segments
- **Samples:** 200
- **Features:** Age, Annual Income (k$), Spending Score (1–100)
- **Preprocessing:** StandardScaler applied so no feature dominates by scale

## Results

### Elbow Method

The elbow clearly bends at k=5; beyond this point each additional cluster yields a much
smaller reduction in inertia.

### k=5 Clustering

- **Inertia:** 168.74 (in standardised space)
- **Convergence:** 6 iterations
- **Cluster sizes:** 33, 48, 39, 22, 58

### Cluster Profile (feature means, original scale)

| Cluster | Size | Age | Annual Income (k$) | Spending Score | Segment |
|---|---|---|---|---|---|
| 0 | 33 | 41.9 | 88.9 | 17.0 | High-income / low-spending (cautious savers) |
| 1 | 48 | 27.8 | 50.5 | 43.9 | Young / moderate income and spending |
| 2 | 39 | 32.7 | 86.5 | 82.1 | High-income / high-spending (premium targets) |
| 3 | 22 | 25.3 | 25.7 | 79.4 | Low-income / high-spending (impulsive buyers) |
| 4 | 58 | 55.5 | 48.5 | 41.8 | Middle-aged / moderate income and spending |

## Key Findings

**Performance and why:** K-Means recovers five interpretable customer archetypes from three
continuous features. The algorithm succeeds here because the Annual Income vs. Spending Score
scatter shows five visually distinct, approximately convex groups — exactly the geometry that
K-Means is designed to find. Standardisation was essential: Annual Income spans 15–137 k$
while Spending Score spans 1–100; without scaling, the income axis would dominate every
distance calculation and the algorithm would effectively ignore spending score, collapsing two
or more distinct segments into one.

**Cluster interpretation:** The two high-income clusters (0 and 2) separate cleanly by spending
behaviour — cluster 2 spends aggressively while cluster 0 saves despite comparable income.
The two young clusters (1 and 3) separate by income — cluster 3 spends well above its income
level, a distinct and actionable pattern. Cluster 4 (the largest, 58 members) represents the
broad middle of the distribution: middle-aged customers with moderate income and restrained
spending.

**Strengths of the architecture:** K-Means is computationally efficient (converged in 6
iterations here), scales to large datasets, and produces interpretable centroids that directly
summarise each segment in the original feature units. The elbow method provides a principled
heuristic for selecting k without labelled data.

**Limitations grounded in these results:** K-Means assumes clusters are convex and of roughly
equal variance. Clusters 1 and 4 overlap in the income dimension and differ mainly in age —
a distinction K-Means captures by including all three features, but that would be lost if the
feature set were reduced. The method is also sensitive to initialisation; random_state=42 fixes
this, but different seeds may yield slightly different boundaries (particularly between the
overlapping middle segments). Inertia measures compactness within clusters but not separation
between them — a low inertia does not guarantee that two adjacent clusters are genuinely
distinct rather than an artefact of splitting a continuous distribution. Complementary metrics
such as silhouette score would provide a fuller picture.
