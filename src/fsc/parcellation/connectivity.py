"""Parcellation-level FSC from precalculated SC and FC matrices.

REFACTORED"""

import numpy as np

import fsc

def graph_from_sc(sc):
    """
    Construct a weighted graph from a structural connectivity matrix.

    Nonzero entries are treated as edge conductances. The matrix
    should be symmetric, so only the upper triangle is used.
    """
    sc = fsc.validate_symmetric_matrix(sc, "SC")

    source, target = np.nonzero(np.triu(sc, k=1))
    weights = sc[source, target]

    return fsc.WeightedGraph(
        len(sc), source, target, weights
    )


def solve_parcellation(sc, fc, **options):
    """ Calculate FSC using parcellation based SC and FC matrices. """
    if len(sc) != len(fc):
            raise ValueError("SC and FC dimensions differ.")

    structural_graph = graph_from_sc(sc)
    parcel_pairs, functional_constraints = fsc.constraints_from_fc(fc)

    return fsc.solve_fsc(structural_graph, parcel_pairs, functional_constraints, **options)


def edge_currents_to_matrix(graph, edge_currents, absolute=False):
    """Convert structural edge currents into an N x N connectivity matrix.

    Each current is assigned to the two nodes connected by its graph edge.
    Signed currents produce an antisymmetric matrix, whereas absolute
    currents produce a symmetric matrix of structural support magnitudes.

    Parallel edges contribute additively.
    """
    matrix = np.zeros((graph.n_nodes, graph.n_nodes))

    for node_i, node_j, current in zip(
        graph.edge_i, graph.edge_j, edge_currents
    ):
        if absolute:
            current = abs(current)

        matrix[node_i, node_j] += current
        matrix[node_j, node_i] += current if absolute else -current

    return matrix