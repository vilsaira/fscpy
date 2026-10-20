"""Weighted, undirected graphs shared by the FSC workflows."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class WeightedGraph:
    n_nodes: int
    edge_i: np.ndarray
    edge_j: np.ndarray
    conductance: np.ndarray

    def __post_init__(self):
        """Convert edge inputs to arrays and check graph validity."""
        self.edge_i = np.asarray(self.edge_i, dtype=int)
        self.edge_j = np.asarray(self.edge_j, dtype=int)
        self.conductance = np.asarray(self.conductance, dtype=float)

        if not (self.edge_i.shape == self.edge_j.shape == self.conductance.shape):
            raise ValueError("Edge arrays must have matching shapes.")

        if np.any(~np.isfinite(self.conductance)) or np.any(self.conductance < 0):
            raise ValueError("Conductances must be finite and non-negative.")

        for nodes in (self.edge_i, self.edge_j):
            if np.any(nodes < 0) or np.any(nodes >= self.n_nodes):
                raise ValueError("Edge node indices must be within the graph.")

    def laplacian(self):
        """Construct the graph Laplacian L = D - W.

        The weighted graph W is undirected. Each edge (i, j) contributes its conductance to W[i, j] and W[j, i].

        D is the diagonal matrix of weighted node degrees, and parallel edges contribute additively.
        """
        rows = np.concatenate((self.edge_i, self.edge_j))
        cols = np.concatenate((self.edge_j, self.edge_i))
        weights = np.concatenate((self.conductance, self.conductance))

        adjacency = sparse.coo_matrix(
            (weights, (rows, cols)),
            shape=(self.n_nodes, self.n_nodes)
        ).tocsr()

        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        return (sparse.diags(degree) - adjacency).tocsr()