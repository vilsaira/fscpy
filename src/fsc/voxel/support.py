"""Convert FSC edge currents into voxel-wise support maps."""

import numpy as np

from fsc.solver import solve_fsc_voxel


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



def sum_pairwise_support(graph, voxel, parcellation, fc, pairs=None,
                         **solver_options):
    """Calculate voxel support from independently solved FC pairs."""
    edge_currents = solve_fsc_voxel(
        graph, parcellation, fc, pairs=pairs, **solver_options
    )

    return voxel_support(graph, voxel, edge_currents)

