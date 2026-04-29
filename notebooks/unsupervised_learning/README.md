# Unsupervised Learning Notebooks

Four notebooks covering clustering, dimensionality reduction, and graph
community detection. Evaluation uses intrinsic metrics (inertia, modularity)
and external validation (Adjusted Rand Index) against ground-truth labels
where available.

| Notebook | Dataset | Key experiment |
|---|---|---|
| `K-Means_Clustering/k_means_clustering_example.ipynb` | UCI Dry Bean (13,611 samples, 7 varieties) | Elbow at k=7; ARI = 0.5818; BOMBAY perfectly isolated, SIRA split due to DERMASON overlap |
| `DBScan/dbscan_example.ipynb` | USGS Global Earthquakes M≥5 (1,780 samples) | eps sensitivity sweep; 56 clusters, 1.7% noise; arc geometry vs K-Means circular segments |
| `PCA/pca_example.ipynb` | UCI Dry Bean (13,611 samples, 16 features) | 2 PCs explain 80% variance; reconstruction MSE drops from 0.45 (k=1) to 0.0007 (k=8) |
| `Community_Detection/community_detection_example.ipynb` | Zachary's Karate Club (34 nodes) | Label propagation finds 3 communities; modularity Q = 0.4138; ARS = 0.7420 |
