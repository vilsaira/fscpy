"""Minimal regional FSC with one indirect functional relationship."""

import numpy as np

from fsc.parcellation import graph_from_sc, solve_parcellation, edge_currents_to_matrix

sc = np.array([[0, 1, 0], [1, 0, 0.5], [0, 0.5, 0]])
fc = np.array([[0, 0, 3], [0, 0, 0], [3, 0, 0]])
result = solve_parcellation(sc, fc)
print("Node potentials:", result.potentials)
print("Absolute current matrix:\n", edge_currents_to_matrix(
    graph_from_sc(sc), result.edge_currents, absolute=True))
print("Relative MNA residual:", result.relative_residual)
