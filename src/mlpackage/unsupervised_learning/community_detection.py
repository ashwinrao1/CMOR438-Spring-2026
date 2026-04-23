"""
Community Detection

Label propagation algorithm for detecting communities in undirected graphs.
Each node is initialized with a unique label. At every iteration, each node
adopts the label held by the majority of its neighbors, with ties broken
uniformly at random. Nodes with no neighbors retain their label throughout.

The algorithm converges when no node changes its label in a full pass. Update
order is shuffled each iteration to prevent systematic bias and reduce the
chance of getting stuck in non-community-aligned fixed points. Convergence is
not guaranteed for every graph, so a maximum iteration count acts as a
fallback stopping condition.

The graph is supplied as an adjacency matrix. Weighted edges are supported:
neighbor votes are summed by edge weight rather than counted uniformly. The
output partitions nodes into communities identified by integer labels, which
are relabelled 0, 1, 2, ... in order of first appearance after fitting.
"""

from __future__ import annotations

from typing import Union, Sequence

import numpy as np

ArrayLike = Union[np.ndarray, Sequence]

__all__ = ["LabelPropagation"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as2d_float(A: ArrayLike, name: str = "A") -> np.ndarray:
    arr = np.asarray(A, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}.")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be square, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _majority_label(
    neighbors: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    Return the label with the highest total edge weight among neighbors.
    Ties are broken uniformly at random.
    """
    label_scores: dict[int, float] = {}
    for neighbor, weight in zip(neighbors, weights):
        label = labels[neighbor]
        label_scores[label] = label_scores.get(label, 0.0) + weight

    max_score = max(label_scores.values())
    candidates = [l for l, s in label_scores.items() if s == max_score]
    return int(rng.choice(candidates))


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Remap arbitrary integer labels to a contiguous 0-based sequence."""
    mapping: dict[int, int] = {}
    out = np.empty_like(labels)
    for i, label in enumerate(labels):
        if label not in mapping:
            mapping[label] = len(mapping)
        out[i] = mapping[label]
    return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LabelPropagation:
    """
    Community detection via label propagation on an undirected weighted graph.

    Parameters
    ----------
    n_iterations : int
        Maximum number of full passes over all nodes.
    random_state : int or None
        Seed for the random number generator used to shuffle update order
        and break label ties.
    """

    def __init__(
        self,
        *,
        n_iterations: int = 100,
        random_state: int | None = None,
    ) -> None:
        if n_iterations < 1:
            raise ValueError(f"n_iterations must be >= 1, got {n_iterations}.")
        self.n_iterations = int(n_iterations)
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.n_communities_: int | None = None
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, A: ArrayLike) -> "LabelPropagation":
        """
        Detect communities in the graph described by adjacency matrix A.

        Parameters
        ----------
        A : array-like of shape (n_nodes, n_nodes)
            Adjacency matrix. Entry A[i, j] is the edge weight between nodes
            i and j. Use 0 for absent edges. The matrix should be symmetric;
            only the upper triangle is used to determine neighbors, but weights
            are read from whichever entry is non-zero.

        Returns
        -------
        self
        """
        A = _as2d_float(A, "A")
        n_nodes = A.shape[0]
        rng = np.random.default_rng(self.random_state)

        labels = np.arange(n_nodes, dtype=int)

        # precompute neighbor lists and weights once
        neighbor_data: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_nodes):
            idx = np.flatnonzero(A[i])
            neighbor_data.append((idx, A[i, idx]))

        for iteration in range(self.n_iterations):
            order = rng.permutation(n_nodes)
            changed = False
            for i in order:
                neighbors, weights = neighbor_data[i]
                if neighbors.size == 0:
                    continue
                new_label = _majority_label(neighbors, weights, labels, rng)
                if new_label != labels[i]:
                    labels[i] = new_label
                    changed = True
            self.n_iter_ = iteration + 1
            if not changed:
                break

        self.labels_ = _relabel(labels)
        self.n_communities_ = int(np.unique(self.labels_).size)
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def modularity(self, A: ArrayLike) -> float:
        """
        Compute the modularity Q of the detected partition.

        Modularity measures how much the density of edges within communities
        exceeds the density expected under a null model that preserves node
        degrees. Values near 1 indicate strong community structure; values
        near 0 suggest the partition is no better than random.

        Parameters
        ----------
        A : array-like of shape (n_nodes, n_nodes)
            The same adjacency matrix passed to fit.
        """
        self._check_fitted()
        A = _as2d_float(A, "A")
        if A.shape[0] != self.labels_.shape[0]:
            raise ValueError(
                f"A has {A.shape[0]} nodes but the model was fitted on "
                f"{self.labels_.shape[0]} nodes."
            )
        m = A.sum() / 2.0
        if m == 0.0:
            raise ValueError("Modularity is undefined for a graph with no edges.")
        degrees = A.sum(axis=1)
        Q = 0.0
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                if self.labels_[i] == self.labels_[j]:
                    Q += A[i, j] - (degrees[i] * degrees[j]) / (2.0 * m)
        return float(Q / (2.0 * m))

    def community_sizes(self) -> dict[int, int]:
        """Return a mapping from community label to the number of nodes it contains."""
        self._check_fitted()
        labels, counts = np.unique(self.labels_, return_counts=True)
        return {int(label): int(count) for label, count in zip(labels, counts)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.labels_ is None:
            raise RuntimeError("Call fit before using this method.")
