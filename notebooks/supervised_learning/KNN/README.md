# K-Nearest Neighbors — Digits Classification Dataset

## Algorithm

K-Nearest Neighbors (KNN) is a non-parametric, instance-based classifier. At prediction time
it computes the distance from the query point to every training point, selects the k closest,
and returns the majority class label. There is no explicit training phase — the entire dataset
is the model. Key hyperparameters are k, the distance metric, and the weighting scheme.

## Dataset

- **Source:** sklearn `load_digits`
- **Task:** Multi-class classification — identify handwritten digits 0–9
- **Samples:** 1,797 (8×8 grayscale images flattened to 64 pixel features)
- **Classes:** 10, approximately balanced (~180 samples per class)
- **Split:** 1,437 train / 360 test
- **Preprocessing:** StandardScaler fit on training data only

## Results

### k-Sweep (Euclidean, uniform weights)

| k | Train Accuracy | Test Accuracy |
|---|---|---|
| **1** | 1.0000 | **0.9861** |
| 3 | 0.9861 | 0.9861 |
| 5 | 0.9882 | 0.9861 |
| 7 | 0.9833 | 0.9833 |
| 10 | 0.9763 | 0.9694 |
| 15 | 0.9687 | 0.9639 |
| 20 | 0.9617 | 0.9611 |
| 30 | 0.9548 | 0.9500 |

Best test accuracy: **0.9861** at k=1.

### Distance Metric Comparison (k=1)

| Metric | Train | Test |
|---|---|---|
| Euclidean | 1.0000 | 0.9861 |
| Manhattan | 1.0000 | 0.9861 |

### Weighting Scheme Comparison (k=1)

| Weights | Train | Test |
|---|---|---|
| Uniform | 1.0000 | 0.9861 |
| Distance | 1.0000 | 0.9861 |

## Key Findings

**Performance and why:** KNN achieves 98.6% test accuracy because the 64-dimensional
standardised pixel space has strong local structure — images of the same digit form tight,
well-separated clusters. The model exploits this directly: finding the nearest stored example
is essentially a visual similarity lookup, and the lookup succeeds because intra-class variance
is low relative to inter-class distance.

**k selection:** k=1 is already optimal. Test accuracy peaks at 0.9861 for k=1, 3, and 5,
then decreases monotonically to 0.9500 at k=30. There is no "noisy k=1, improve with larger k"
pattern here — the dataset is clean enough that the single nearest neighbor is a reliable
predictor. Larger k dilutes the vote with increasingly distant (and less similar) samples,
lowering accuracy without any compensating noise-reduction benefit.

**Distance and weighting:** Manhattan and Euclidean distance perform identically because, in
64 dimensions, both metrics spread pairwise distances similarly and rank neighbors in the same
order on clean data. Distance weighting offers no advantage for the same reason: when all
k neighbors belong to the correct class, re-weighting them does not change the vote outcome.

**Strengths of the architecture:** KNN requires no closed-form assumptions about the data
distribution and adapts perfectly to any local geometry. On this structured, balanced dataset
those properties translate into near-perfect accuracy with no hyperparameter tuning beyond
k=1.

**Limitations grounded in these results:** The entire training set must be stored and searched
at prediction time — 1,437 distance computations per query at 64 dimensions. On larger,
noisier datasets the absence of a training phase becomes a liability: KNN cannot generalise
beyond the training distribution, and performance degrades when the test manifold contains
patterns not densely covered by training samples. The monotonic accuracy drop from k=7 to k=30
also illustrates that, even on ideal data, the algorithm has no mechanism to recover the
local structure it dilutes by averaging over a wider neighborhood.
