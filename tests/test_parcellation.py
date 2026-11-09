"""Small deterministic tests for parcellation FSC."""

import numpy as np
import pytest

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

def test_joint_single_pair_matches_pairwise():
    """With one constraint, Eq. 11 and the pairwise method coincide."""
    sc, fc = five_node_matrices()
    np.testing.assert_allclose(
        fsc.solve_from_parcellations(sc, fc, method="joint"),
        fsc.solve_from_parcellations(sc, fc),
        atol=1e-12,
    )


def test_joint_two_constraints_analytic():
    """Two constraints sharing node 4 form no cycle; Eq. 11 is solvable."""
    sc, _ = five_node_matrices()
    fc = np.zeros((5, 5))
    fc[1, 4] = fc[4, 1] = 1.0
    fc[3, 4] = fc[4, 3] = 1.0

    support = fsc.solve_from_parcellations(sc, fc, method="joint")

    # Joint solution: potentials 4:0, 2:0.6, 0:0.8, 1:1.0, 3:1.0.
    expected = np.zeros((5, 5))
    for (i, j), c in {(1, 0): 0.2, (0, 2): 0.2, (2, 3): 0.4, (2, 4): 0.6}.items():
        expected[i, j] = expected[j, i] = c
    np.testing.assert_allclose(support, expected, atol=1e-12)

    # The pairwise method is a different quantity for several pairs
    # (1/3, 1/3, 1/2, 5/6 on the same edges).
    assert not np.allclose(support, fsc.solve_from_parcellations(sc, fc))


def test_joint_rejects_cyclic_fc():
    """Three mutually correlated regions form a cycle no exact solution for eq. 11."""
    sc = np.ones((3, 3)) - np.eye(3)
    fc = 0.5 * (np.ones((3, 3)) - np.eye(3))
    with pytest.raises(ValueError, match="cycle"):
        fsc.solve_from_parcellations(sc, fc, method="joint")


def test_unknown_method():
    sc, fc = five_node_matrices()
    with pytest.raises(ValueError, match="Unknown method"):
        fsc.solve_from_parcellations(sc, fc, method="bogus")
