from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fsc import graph_from_edges, solve_fsc, constraints_from_fc
from fsc.parcellation import graph_from_sc


def test_one_edge_ohms_law():
    result = solve_fsc(graph_from_edges(2, [0], [1], [2]), [[0, 1]], [3])
    assert_allclose(result.potentials, [0, -3])
    assert_allclose(result.edge_currents, [6])
    assert result.relative_residual < 1e-11


def test_two_series_edges_and_ground_invariance():
    graph = graph_from_edges(3, [0, 1], [1, 2], [1, 0.5])
    a = solve_fsc(graph, [[0, 2]], [3], rtol=1e-12)
    b = solve_fsc(graph, [[0, 2]], [3], ground=2, rtol=1e-12)
    assert_allclose(a.potentials, [0, -1, -3], atol=1e-10)
    assert_allclose(a.edge_currents, [1, 1], atol=1e-10)
    assert_allclose(a.edge_currents, b.edge_currents, atol=1e-10)


def test_parallel_currents_split_by_conductance():
    graph = graph_from_edges(2, [0, 0], [1, 1], [2, 3])
    result = solve_fsc(graph, [[0, 1]], [4])
    assert_allclose(result.edge_currents, [8, 12], atol=1e-10)


def test_edge_orientation_only_reverses_signed_current():
    graph = graph_from_edges(2, [0], [1], [2])
    reversed_graph = graph_from_edges(2, [1], [0], [2])
    a = solve_fsc(graph, [[0, 1]], [3])
    b = solve_fsc(reversed_graph, [[0, 1]], [3])
    assert_allclose(b.edge_currents, -a.edge_currents)


def test_sign_conventions_are_explicit():
    graph = graph_from_edges(2, [0], [1], [1])
    legacy = solve_fsc(graph, [[0, 1]], [-3])
    signed = solve_fsc(graph, [[0, 1]], [-3], sign="signed")
    assert_allclose(legacy.edge_currents, [3])
    assert_allclose(signed.edge_currents, [-3])


def test_joint_inconsistent_constraints_report_residual():
    graph = graph_from_edges(3, [0, 1], [1, 2], [1, 1])
    result = solve_fsc(graph, [[0, 1], [1, 2], [0, 2]], [1, 1, 1])
    assert np.linalg.norm(result.constraint_residual) > 0.5
    assert result.relative_residual > 0.1


def test_explicit_zero_constraint_is_not_omitted():
    graph = graph_from_edges(2, [0], [1], [1])
    result = solve_fsc(graph, [[0, 1]], [0])
    assert_allclose(result.potentials[0] - result.potentials[1], 0)
    assert_allclose(result.edge_currents, 0)


def test_fc_matrix_omits_zero_nan_and_maps_nodes():
    fc = np.array([[0, np.nan, -.5], [np.nan, 0, 0], [-.5, 0, 0]])
    pairs, values = constraints_from_fc(fc, [4, 5, 6])
    assert_allclose(pairs, [[4, 6]])
    assert_allclose(values, [-.5])


def test_regional_reference_from_unmodified_branch():
    with np.load(Path(__file__).parent / "data/legacy_reference.npz") as ref:
        pairs, values = constraints_from_fc(ref["fc"])
        result = solve_fsc(graph_from_sc(ref["sc"]), pairs, values, rtol=1e-5)
        assert_allclose(result.potentials, ref["phi"], atol=1e-5)
        assert_allclose(result.source_currents, ref["sources"], atol=1e-5)
