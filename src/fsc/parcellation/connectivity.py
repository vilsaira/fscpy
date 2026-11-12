"""Parcellation FSC from structural and functional connectivity matrices."""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import factorized
from warnings import warn
import fsc


def graph_from_sc(sc):
    """Use the upper triangle of a symmetric SC matrix as graph conductances."""
    sc = fsc.validate_symmetric_matrix(sc, "SC")
    source, target = np.nonzero(np.triu(sc, k=1))
    return fsc.WeightedGraph(len(sc), source, target, sc[source, target])

def solve_from_parcellations(sc, fc, *, method="pairwise", solver="auto"):
    """Calculate FSC support from parcellation SC and FC matrices.

    method="pairwise" solves each positive FC pair separately and sums the
    absolute edge currents. It is defined for any FC matrix.

    method="joint" solves all FC constraints in one modified-nodal-analysis
    system (Network Neuroscience paper, Eq. 11) and returns absolute edge
    currents. It requires the FC pairs to contain no cycles. Therefore, a dense FC likely has no exact joint solution.

    solver is passed to fsc.solve_fsc() and is used only by method="joint".
    """
    if method == "pairwise":
        return _solve_pairwise(sc, fc)
    if method == "joint":
        return _solve_joint(sc, fc, solver=solver)
    raise ValueError(f"Unknown method: {method}")

def _solve_pairwise(sc, fc):
    """Solve each FC pair separately and sum absolute edge currents."""
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
        warn("No structural support for any FC pair.", stacklevel=3)
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
        potentials *= value / resistance

        currents = graph.conductance * (
            potentials[graph.edge_i] - potentials[graph.edge_j]
        )
        edge_support += np.abs(currents)

    return edge_currents_to_matrix(graph, edge_support, absolute=True)

def _independent_cycles(pairs, n_nodes):
    """Number of independent cycles in the graph whose edges are FC pairs."""
    if len(pairs) == 0:
        return 0
    adjacency = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(n_nodes, n_nodes),
    )
    n_components, _ = connected_components(adjacency, directed=False)
    return len(pairs) - (n_nodes - n_components)

def _joint_solution(sc, fc, solver="auto"):
    """Solve all FC constraints jointly (NetN Eq. 11) for cycle-free FC pairs.

    Returns the structural graph and the FSCResult. Raises ValueError if the
    FC pairs contain any cycle.
    """
    graph = graph_from_sc(sc)

    if np.shape(fc) != (graph.n_nodes, graph.n_nodes):
        raise ValueError("SC and FC dimensions differ.")

    pairs, values = fsc.fc_pairs(fc)
    cycles = _independent_cycles(pairs, graph.n_nodes)
    if cycles > 0:
        raise ValueError(
            f"The FC pairs form {cycles} independent cycle(s). The joint "
            "system has an exact solution only if FC values sum to zero "
            "around every cycle, which correlation values do not. Use "
            "method='pairwise' for dense FC."
        )

    constraints = fsc.functional_connectivity_from_fc_pairs(
        pairs, values, graph.n_nodes
    )
    return graph, fsc.solve_fsc(graph, constraints, solver=solver)


def _solve_joint(sc, fc, *, solver="auto"):
    """Joint FSC support matrix (absolute edge currents)."""
    graph, result = _joint_solution(sc, fc, solver)
    return edge_currents_to_matrix(graph, result.edge_currents, absolute=True)


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
