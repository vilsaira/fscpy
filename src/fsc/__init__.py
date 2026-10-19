"""Function–structure coupling on weighted structural graphs."""

from .weighted_graph import WeightedGraph
from .solver import (
    FSCResult, FunctionalConnectivity, fc_pairs,
    functional_connectivity_from_fc_pairs, functional_connectivity_from_fc_matrix, solve_fsc,
)
from .parcellation import graph_from_sc, solve_from_parcellations, edge_currents_to_matrix
from .utils import validate_symmetric_matrix

__all__ = [
    "WeightedGraph", "FSCResult", "FunctionalConnectivity", "fc_pairs",
    "functional_connectivity_from_fc_pairs", "functional_connectivity_from_fc_matrix", "solve_fsc",
    "graph_from_sc", "solve_from_parcellations", "edge_currents_to_matrix",
    "validate_symmetric_matrix",
]
