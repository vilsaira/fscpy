"""Function–structure coupling on weighted structural graphs."""

from .weighted_graph import WeightedGraph
from .solver import (
    FSCResult, FunctionalConnectivity, fc_pairs,
    functional_connectivity_from_fc_pairs, functional_connectivity_from_fc_matrix, solve_fsc, solve_fsc_voxel
)
from .parcellation import graph_from_sc, solve_from_parcellations, edge_currents_to_matrix
from .spectral import SpectralResult, decompose, laplacian_modes, spectral_from_parcellations
from .utils import validate_symmetric_matrix

__all__ = [
    "WeightedGraph", "FSCResult", "FunctionalConnectivity", "fc_pairs",
    "functional_connectivity_from_fc_pairs", "functional_connectivity_from_fc_matrix", "solve_fsc", "solve_fsc_voxel",
    "graph_from_sc", "solve_from_parcellations", "edge_currents_to_matrix",
    "validate_symmetric_matrix", "SpectralResult", "decompose", "laplacian_modes", "spectral_from_parcellations",
]
