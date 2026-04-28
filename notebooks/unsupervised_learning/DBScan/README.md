# DBSCAN — Density-Based Spatial Clustering

## Model Architecture

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) classifies every point as one of three types based on the density of its neighbourhood:

- **Core point:** has at least `min_samples` neighbours within distance `eps`
- **Border point:** within `eps` of a core point but not itself a core point
- **Noise point:** neither core nor border; assigned label $-1$

A cluster is the maximal set of mutually density-connected core points together with their border points. Because connectivity is defined through chains of dense neighbourhoods rather than distance to a fixed centroid, clusters can take any shape — arcs, rings, elongated corridors — that K-Means cannot recover (K-Means partitions space into convex Voronoi cells). The algorithm also produces noise labels automatically, correctly flagging genuine outliers rather than forcing them into a cluster.

The two hyperparameters are `eps` (neighbourhood radius) and `min_samples` (minimum neighbours to be a core point). Standardising features before fitting is essential because `eps` is scale-dependent.

## Dataset

**USGS Global Earthquake Catalog** — 1,780 earthquakes with magnitude $\geq 5.0$ during calendar year 2023, fetched directly from the USGS public API. Features used: standardised latitude and longitude of the epicentre. Earthquake epicentres cluster along tectonic plate boundaries — the Pacific Ring of Fire, the Alpide Belt, and mid-ocean spreading ridges — which are continuous, arc-shaped, non-convex structures that DBSCAN is uniquely suited to recover.

## What the Notebook Covers

- Baseline DBSCAN (eps=0.2, min_samples=5): 56 clusters, 30 noise points (1.7%)
- Epsilon sensitivity sweep (eps = 0.1, 0.2, 0.4, 0.8) showing fragmentation at small eps and merging at large eps
- min_samples sensitivity sweep at fixed eps
- Side-by-side comparison of DBSCAN vs K-Means on the same geographic data, demonstrating why convex Voronoi partitioning fails for fault-line arc geometry
