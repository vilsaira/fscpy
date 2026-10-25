"""Small deterministic tests for parcellation FSC."""

import numpy as np

import fsc


def five_node_matrices():
    sc = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=float)
    fc = np.zeros((5, 5))
    fc[1, 4] = fc[4, 1] = 1.0
    return sc, fc


def test_five_node_path():
    sc, fc = five_node_matrices()
    support = fsc.solve_from_parcellations(sc, fc)
    expected = np.zeros((5, 5))
    for i, j in [(1, 0), (0, 2), (2, 4)]:
        expected[i, j] = expected[j, i] = 1 / 3
    np.testing.assert_allclose(support, expected, rtol=1e-12, atol=1e-12)


def test_negative_and_empty_fc():
    sc, fc = five_node_matrices()
    np.testing.assert_array_equal(
        fsc.solve_from_parcellations(sc, -fc),
        np.zeros_like(sc),
    )
    np.testing.assert_array_equal(
        fsc.solve_from_parcellations(sc, np.zeros_like(fc)),
        np.zeros_like(sc),
    )


def test_general_mna_single_constraint():
    sc, fc = five_node_matrices()
    graph = fsc.graph_from_sc(sc)
    constraints = fsc.functional_connectivity_from_fc_matrix(fc)
    result = fsc.solve_fsc(graph, constraints, solver="direct")

    assert result.relative_residual < 1e-10
    np.testing.assert_allclose(
        result.potentials[1] - result.potentials[4], 1.0, atol=1e-12
    )
    np.testing.assert_allclose(np.abs(result.edge_currents),
                               np.array([1 / 3, 1 / 3, 0, 1 / 3]), atol=1e-12)
