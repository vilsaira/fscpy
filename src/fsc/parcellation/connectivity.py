"""Parcellation FSC from structural and functional connectivity matrices."""

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized
from warnings import warn
import fsc


def graph_from_sc(sc):
    """Use the upper triangle of a symmetric SC matrix as graph conductances."""
    sc = fsc.validate_symmetric_matrix(sc, "SC")
    source, target = np.nonzero(np.triu(sc, k=1))
    return fsc.WeightedGraph(len(sc), source, target, sc[source, target])

from warnings import warn

def solve_from_parcellations(sc, fc):
    """Calculate FSC support from parcellation SC and FC matrices."""
    graph = graph_from_sc(sc)

    if np.shape(fc) != (graph.n_nodes, graph.n_nodes):
        raise ValueError("SC and FC dimensions differ.")

    pairs, values = fsc.fc_pairs(fc)
    edge_support = np.zeros(len(graph.edge_i))

    if len(pairs) == 0:
        return edge_currents_to_matrix(graph, edge_support, absolute=True)

    laplacian = graph.laplacian()
    _, component_labels = connected_components(laplacian, directed=False)

    # Only FC pairs with a structural path can carry current.
    connected = (
        component_labels[pairs[:, 0]] == component_labels[pairs[:, 1]]
    )

    pairs = pairs[connected]
    values = values[connected]

    if len(pairs) == 0:
        warn("No structural support for any FC pair.", stacklevel=2)
        return edge_currents_to_matrix(graph, edge_support, absolute=True)

    # Ground one node in each connected component.
    _, ground_nodes = np.unique(component_labels, return_index=True)
    mask = np.ones(graph.n_nodes, dtype=bool)
    mask[ground_nodes] = False

    solve = factorized(laplacian[mask][:, mask].tocsc())

    rhs = np.zeros(graph.n_nodes)
    potentials = np.zeros(graph.n_nodes)

    for (source, target), value in zip(pairs, values):
        rhs.fill(0)
        rhs[source] = 1.0
        rhs[target] = -1.0

        potentials.fill(0)
        potentials[mask] = solve(rhs[mask])

        resistance = potentials[source] - potentials[target]
        potentials *= abs(value) / resistance

        currents = graph.conductance * (
            potentials[graph.edge_i] - potentials[graph.edge_j]
        )
        edge_support += np.abs(currents)

    return edge_currents_to_matrix(graph, edge_support, absolute=True)


def edge_currents_to_matrix(graph, edge_currents, absolute=False):
    """Convert graph edge currents to an N-by-N connectivity matrix."""
    matrix = np.zeros((graph.n_nodes, graph.n_nodes))
    for node_i, node_j, current in zip(
        graph.edge_i, graph.edge_j, edge_currents
    ):
        if absolute:
            current = abs(current)
        matrix[node_i, node_j] += current
        matrix[node_j, node_i] += current if absolute else -current
    return matrix
