# K-Nearest Neighbors Classifier

## Model Architecture

K-Nearest Neighbors (KNN) is a non-parametric, instance-based classifier. It stores the entire training set and, for each query point, finds the $k$ training samples with the smallest distance to that point. The majority class among the $k$ neighbours is returned as the prediction.

There is no explicit training phase — computation happens entirely at prediction time. The algorithm has no learned parameters; its only decisions are the choice of $k$, the distance metric (Euclidean or Manhattan), and the weighting scheme (uniform majority vote vs inverse-distance weighting).

KNN is highly sensitive to feature scale because distance is computed in the raw feature space. Standardising features to zero mean and unit variance before fitting is therefore essential.

## Dataset

**UCI Dry Bean** (`fetch_ucirepo(id=602)`) — 13,611 bean grain images, each described by 16 morphological measurements: area, perimeter, major and minor axis lengths, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness, shape factors. The target is the bean variety: one of seven Turkish cultivars (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA). DERMASON is the most common variety (3,546 samples) and BOMBAY the rarest (522 samples).

## What the Notebook Covers

- Default KNN at $k = 5$ (Euclidean, uniform): train 0.9416, test 0.9203
- Sweep of $k$ from 1 to 30: best at $k = 10$, test accuracy 0.9291
- Distance metric comparison at $k = 10$: Euclidean (0.9291) vs Manhattan (0.9218)
- Weighting scheme comparison at $k = 10$: uniform (0.9291) vs distance-weighted (0.9266)
- Confusion matrix and per-class accuracy: BOMBAY achieves 100%; SIRA is lowest at 87.20%
- Visualisation of a query point and its 5 nearest neighbours in 2D PCA space
