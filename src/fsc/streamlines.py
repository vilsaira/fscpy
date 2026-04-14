#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamline-level utilities for FSC.

These functions operate on tractography-derived streamline information such as
MRtrix endpoint assignments and streamline lengths. They are intentionally kept
separate from the FSC core model, which solves the graph-level constrained
Laplacian / MNA system.

Main ideas
----------
- Streamline resistance can be defined from streamline length
- Streamline conductance is then 1 / resistance
- Graph-level SC can be built by summing streamline conductances between nodes
- Streamline-wise currents can be computed from solved nodal potentials via
  Ohm's law
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from networkx.algorithms.approximation import steiner_tree


def build_sc_from_streamlines(
    streamline_assignments: np.ndarray,
    streamline_resistances: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """
    Build a graph-level structural conductance matrix from streamlines.

    Each streamline is treated as a parallel conductive path between two nodes.
    The conductance of one streamline is defined as:

        g_s = 1 / R_s

    and the node-pair conductance is the sum of streamline conductances:

        SC_ij = sum_s g_s

    Parameters
    ----------
    streamline_assignments : np.ndarray
        Array of shape (n_streamlines, 2), where each row contains the 1-based
        node labels assigned to a streamline.
    streamline_resistances : np.ndarray
        Array of shape (n_streamlines,), giving one resistance per streamline.
        A common choice is streamline length.
    n_nodes : int
        Number of nodes in the connectome/parcellation.

    Returns
    -------
    np.ndarray
        Symmetric structural conductance matrix of shape (n_nodes, n_nodes).
    """
    streamline_assignments_arr = np.asarray(streamline_assignments)
    streamline_resistances_arr = np.asarray(streamline_resistances, dtype=float)

    if streamline_assignments_arr.ndim != 2 or streamline_assignments_arr.shape[1] != 2:
        raise ValueError("streamline_assignments must have shape (n_streamlines, 2).")

    if streamline_resistances_arr.ndim != 1:
        raise ValueError("streamline_resistances must have shape (n_streamlines,).")

    if streamline_assignments_arr.shape[0] != streamline_resistances_arr.shape[0]:
        raise ValueError(
            "streamline_assignments and streamline_resistances must have the same length."
        )

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")

    sc = np.zeros((n_nodes, n_nodes), dtype=float)

    for idx, assignment in enumerate(streamline_assignments_arr):
        if np.any(assignment == 0):
            continue

        i = int(assignment[0]) - 1
        j = int(assignment[1]) - 1

        if i < 0 or j < 0 or i >= n_nodes or j >= n_nodes:
            raise ValueError("streamline assignment contains node index out of bounds.")

        resistance = float(streamline_resistances_arr[idx])

        if resistance <= 0 or not np.isfinite(resistance):
            continue

        conductance = 1.0 / resistance
        sc[i, j] += conductance
        sc[j, i] += conductance

    np.fill_diagonal(sc, 0.0)
    return sc


def get_streamline_currents(
    nodal_potentials: np.ndarray,
    streamline_assignments: np.ndarray,
    streamline_resistances: np.ndarray,
) -> np.ndarray:
    """
    Compute streamline-wise currents using Ohm's law.

    For a streamline s connecting nodes i and j:

        I_s = (phi_i - phi_j) / R_s

    Parameters
    ----------
    nodal_potentials : np.ndarray
        Array of shape (n_nodes,), containing nodal potentials phi.
    streamline_assignments : np.ndarray
        Array of shape (n_streamlines, 2), where each row contains the 1-based
        node labels assigned to a streamline.
    streamline_resistances : np.ndarray
        Array of shape (n_streamlines,), giving one resistance per streamline.
        A common choice is streamline length.

    Returns
    -------
    np.ndarray
        Signed streamline-wise currents of shape (n_streamlines,).
    """
    phi = np.asarray(nodal_potentials, dtype=float)
    streamline_assignments_arr = np.asarray(streamline_assignments)
    streamline_resistances_arr = np.asarray(streamline_resistances, dtype=float)

    if phi.ndim != 1:
        raise ValueError("nodal_potentials must have shape (n_nodes,).")

    if streamline_assignments_arr.ndim != 2 or streamline_assignments_arr.shape[1] != 2:
        raise ValueError("streamline_assignments must have shape (n_streamlines, 2).")

    if streamline_resistances_arr.ndim != 1:
        raise ValueError("streamline_resistances must have shape (n_streamlines,).")

    if streamline_assignments_arr.shape[0] != streamline_resistances_arr.shape[0]:
        raise ValueError(
            "streamline_assignments and streamline_resistances must have the same length."
        )

    streamline_currents = np.zeros(streamline_resistances_arr.shape[0], dtype=float)

    for idx, assignment in enumerate(streamline_assignments_arr):
        if np.any(assignment == 0):
            continue

        i = int(assignment[0]) - 1
        j = int(assignment[1]) - 1

        if i < 0 or j < 0 or i >= phi.shape[0] or j >= phi.shape[0]:
            raise ValueError("streamline assignment contains node index out of bounds.")

        resistance = float(streamline_resistances_arr[idx])

        if resistance <= 0 or not np.isfinite(resistance):
            continue

        streamline_currents[idx] = (phi[i] - phi[j]) / resistance

    return streamline_currents


def get_streamline_current_magnitudes(
    nodal_potentials: np.ndarray,
    streamline_assignments: np.ndarray,
    streamline_resistances: np.ndarray,
) -> np.ndarray:
    """
    Compute absolute streamline-wise current magnitudes.

    This is useful for tractography filtering and color mapping when current
    direction is not needed.

    Returns
    -------
    np.ndarray
        Absolute streamline current magnitudes.
    """
    return np.abs(
        get_streamline_currents(
            nodal_potentials=nodal_potentials,
            streamline_assignments=streamline_assignments,
            streamline_resistances=streamline_resistances,
        )
    )


def build_support_graph_from_edge_currents(
    edge_currents: np.ndarray,
    sc_matrix: np.ndarray,
    use_absolute_currents: bool = True,
    epsilon: float = 1e-15,
) -> nx.Graph:
    """
    Build a NetworkX graph for pathway selection from FSC edge currents.

    Higher current corresponds to lower graph cost.

    Parameters
    ----------
    edge_currents : np.ndarray
        FSC edge-current matrix of shape (n_nodes, n_nodes).
    sc_matrix : np.ndarray
        Structural conductance matrix of shape (n_nodes, n_nodes).
    use_absolute_currents : bool, optional
        If True, use absolute edge currents for support strength.
    epsilon : float, optional
        Small constant added to finite weights for numerical safety.

    Returns
    -------
    nx.Graph
        Weighted graph suitable for shortest-path or Steiner-tree selection.
    """
    edge_currents_arr = np.asarray(edge_currents, dtype=float)
    sc_arr = np.asarray(sc_matrix, dtype=float)

    if edge_currents_arr.shape != sc_arr.shape:
        raise ValueError("edge_currents and sc_matrix must have the same shape.")

    strength = np.abs(edge_currents_arr) if use_absolute_currents else edge_currents_arr.copy()

    max_strength = float(np.max(strength))
    if max_strength <= 0 or not np.isfinite(max_strength):
        raise ValueError("All edge-current strengths are zero or invalid.")

    weights = 1.0 - (strength / max_strength)
    weights[sc_arr <= 0] = np.inf
    weights += epsilon

    return nx.from_numpy_array(weights)


def get_pairwise_steiner_streamline_currents(
    fc_matrix: np.ndarray,
    nodal_potentials: np.ndarray,
    edge_currents: np.ndarray,
    sc_matrix: np.ndarray,
    streamline_assignments: np.ndarray,
    streamline_resistances: np.ndarray,
    use_absolute_currents: bool = True,
    use_absolute_streamline_currents: bool = True,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Compute pair-specific streamline-current maps using FSC-guided Steiner trees.

    This reproduces the earlier pathway-selection logic:
    1. Use FSC edge currents to build a support graph
    2. For each nonzero FC pair in the upper triangle, compute a Steiner tree
    3. Assign streamline currents only to streamlines whose node pair lies on
       the selected tree

    Parameters
    ----------
    fc_matrix : np.ndarray
        Functional connectivity matrix. Nonzero upper-triangular entries define
        the node pairs to process.
    nodal_potentials : np.ndarray
        Solved FSC nodal potentials.
    edge_currents : np.ndarray
        Solved FSC edge-current matrix.
    sc_matrix : np.ndarray
        Structural conductance matrix used by FSC.
    streamline_assignments : np.ndarray
        MRtrix-like streamline endpoint assignments, shape (n_streamlines, 2).
    streamline_resistances : np.ndarray
        Per-streamline resistances, e.g. lengths.
    use_absolute_currents : bool, optional
        If True, use absolute edge currents when building the support graph.
    use_absolute_streamline_currents : bool, optional
        If True, return absolute streamline currents for each selected pair.

    Returns
    -------
    dict[tuple[int, int], np.ndarray]
        Dictionary keyed by zero-based node-pair tuples (i, j). Each value is a
        streamline-current vector of shape (n_streamlines,) for that FC pair.
    """
    fc_arr = np.asarray(fc_matrix, dtype=float)
    phi = np.asarray(nodal_potentials, dtype=float)
    streamline_assignments_arr = np.asarray(streamline_assignments)
    streamline_resistances_arr = np.asarray(streamline_resistances, dtype=float)

    if fc_arr.ndim != 2 or fc_arr.shape[0] != fc_arr.shape[1]:
        raise ValueError("fc_matrix must be a square matrix.")

    support_graph = build_support_graph_from_edge_currents(
        edge_currents=edge_currents,
        sc_matrix=sc_matrix,
        use_absolute_currents=use_absolute_currents,
    )

    fc_edges = np.argwhere(np.triu(fc_arr, k=1) != 0)
    pairwise_streamline_currents: dict[tuple[int, int], np.ndarray] = {}

    for i, j in fc_edges:
        tree = steiner_tree(support_graph, [int(i), int(j)], weight="weight", method="kou")

        tree_edges = set()
        for u, v in tree.edges:
            tree_edges.add((u, v))
            tree_edges.add((v, u))

        currents = np.zeros(streamline_resistances_arr.shape[0], dtype=float)

        for idx, assignment in enumerate(streamline_assignments_arr):
            if np.any(assignment == 0):
                continue

            u = int(assignment[0]) - 1
            v = int(assignment[1]) - 1

            if (u, v) not in tree_edges:
                continue

            resistance = float(streamline_resistances_arr[idx])
            if resistance <= 0 or not np.isfinite(resistance):
                continue

            currents[idx] = (phi[u] - phi[v]) / resistance

        if use_absolute_streamline_currents:
            currents = np.abs(currents)

        pairwise_streamline_currents[(int(i), int(j))] = currents

    return pairwise_streamline_currents