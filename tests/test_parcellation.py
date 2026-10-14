import numpy as np
from numpy.testing import assert_allclose

from fsc import WeightedGraph
from fsc.parcellation import graph_from_sc, solve_parcellation, edge_currents_to_matrix


def test_parcellation_workflow():
    sc = np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]])
    fc = np.array([[0, 0, 3], [0, 0, 0], [3, 0, 0]])
    result = solve_parcellation(sc, fc)
    assert_allclose(result.edge_currents, [2, 2], atol=1e-10)
    graph = graph_from_sc(sc)
    assert isinstance(graph, WeightedGraph)
    signed = edge_currents_to_matrix(graph, result.edge_currents)
    absolute = edge_currents_to_matrix(graph, result.edge_currents, absolute=True)
    assert_allclose(signed, -signed.T)
    assert_allclose(absolute, np.abs(signed))


def test_empty_fc_produces_zero_field():
    sc = np.array([[0, 2], [2, 0]])
    result = solve_parcellation(sc, np.zeros_like(sc))
    assert_allclose(result.edge_currents, 0)
