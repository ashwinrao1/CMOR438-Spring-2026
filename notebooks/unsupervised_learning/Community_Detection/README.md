# Community Detection — Label Propagation

## Model Architecture

Label propagation is a graph-based, unsupervised community detection algorithm. Each node starts with a unique label. At every iteration, each node adopts the most common label among its immediate neighbours:

$$c_i^{(t+1)} = \underset{c}{\operatorname{argmax}} \sum_{j \in \mathcal{N}(i)} \mathbf{1}[c_j^{(t)} = c]$$

Ties are broken randomly. The algorithm converges when no node changes its label. Because labels propagate along edges, densely connected subgraphs synchronise quickly into a single consensus label, forming communities. The number of communities is not specified in advance — it emerges from the graph topology.

Community quality is measured by modularity $Q$: the fraction of edges within communities minus the expected fraction under a null random graph with the same degree sequence. Values above $\approx 0.3$ indicate meaningful community structure. Agreement with a known partition is measured by the Adjusted Rand Score (ARS), which corrects for chance agreement.

## Dataset

**Zachary's Karate Club** (built into NetworkX) — 34 nodes and 78 edges representing social interactions among members of a university karate club. The network has a verified ground-truth partition into two factions (Mr. Hi's group vs the Officer's group) formed after a real-world club split. This is the standard benchmark for community detection because the ground truth is empirically documented.

## What the Notebook Covers

- Fitting label propagation: 3 communities found (sizes 13, 3, 18), converged in 3 iterations
- Modularity $Q = 0.4138$ (well above the 0.3 threshold for meaningful structure)
- Adjusted Rand Score of 0.7420 against the two-faction ground truth
- Side-by-side network visualisation: predicted communities vs ground-truth faction colours
- Node degree distribution histogram
