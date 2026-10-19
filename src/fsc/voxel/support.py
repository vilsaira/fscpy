"""Convert FSC edge currents into voxel-wise support maps."""

import numpy as np

from fsc.solver import fc_pairs, solve_fsc
from .roi import roi_constraints


def support_on_nodes(n_nodes, source_nodes, target_nodes, edge_currents):
    """Sum absolute edge currents at their two endpoint nodes."""
    support = np.zeros(n_nodes)
    magnitudes = np.abs(edge_currents)
    np.add.at(support, source_nodes, magnitudes)
    np.add.at(support, target_nodes, magnitudes)
    return support


def voxel_support(graph, voxel, edge_currents):
    """Accumulate currents on white-matter voxel edges only."""
    edges = voxel.edge_indices
    values = np.asarray(edge_currents)[edges]
    support = support_on_nodes(
        voxel.n_voxels, graph.edge_i[edges], graph.edge_j[edges], values
    )
    return voxel.to_volume(support)


def sum_pairwise_support(graph, voxel, parcellation, fc, pairs=None, **solver_options):
    """Sum voxel support from independently solved functional ROI pairs."""
    fc = np.asarray(fc)
    n_rois = len(parcellation.roi_labels)
    if fc.shape != (n_rois, n_rois):
        raise ValueError("FC and parcellation dimensions differ.")

    if pairs is None:
        pairs, values = fc_pairs(fc)
    else:
        pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        values = fc[pairs[:, 0], pairs[:, 1]]

    total_support = np.zeros(voxel.node_id.shape)
    for pair, value in zip(pairs, values):
        functional_constraints = roi_constraints(
            parcellation, [pair], [value], graph.n_nodes
        )
        result = solve_fsc(graph, functional_constraints, **solver_options)
        total_support += voxel_support(graph, voxel, result.edge_currents)

    return total_support
