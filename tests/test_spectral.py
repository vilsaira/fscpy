"""Tests for the spectral decomposition of parcellation FSC solutions."""

import numpy as np
import pytest

import fsc
from fsc.spectral import decompose, laplacian_modes


def five_node():
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


def random_network(n=12, seed=0):
    rng = np.random.default_rng(seed)
    sc = rng.random((n, n)) * (rng.random((n, n)) < 0.4)
    sc = np.triu(sc, 1)
    sc = sc + sc.T
    for i in range(n):                                   # ring keeps it connected
        j = (i + 1) % n
        sc[i, j] = sc[j, i] = max(sc[i, j], 0.3)
    fc = np.triu(rng.uniform(0.1, 0.8, (n, n)), 1)
    return sc, fc + fc.T


def effective_resistance(sc, i, j):
    lap = np.diag(sc.sum(axis=1)) - sc
    a = np.zeros(len(sc))
    a[i], a[j] = 1.0, -1.0
    return a @ np.linalg.pinv(lap) @ a


def test_full_basis_reconstructs_pairwise_support():
    sc, fc = random_network()
    spec = fsc.spectral_from_parcellations(sc, fc)
    graph = fsc.graph_from_sc(sc)

    support = np.zeros(len(graph.edge_i))
    for coefficients in spec.coefficients:
        phi = spec.eigenvectors @ coefficients
        support += np.abs(graph.conductance * (phi[graph.edge_i] - phi[graph.edge_j]))

    np.testing.assert_allclose(
        fsc.edge_currents_to_matrix(graph, support, absolute=True),
        fsc.solve_from_parcellations(sc, fc),
        atol=1e-10,
    )


def test_modal_energy_sums_to_dissipated_power():
    sc, fc = random_network()
    spec = fsc.spectral_from_parcellations(sc, fc)
    for (i, j), energy in zip(spec.pairs, spec.energy):
        expected = fc[i, j] ** 2 / effective_resistance(sc, i, j)    # d^2 / R
        assert energy.sum() == pytest.approx(expected, rel=1e-10)
    np.testing.assert_allclose(spec.captured, 1.0, atol=1e-12)


def test_fc_scaling():
    sc, fc = random_network()
    base = fsc.spectral_from_parcellations(sc, fc)
    scaled = fsc.spectral_from_parcellations(sc, 3 * fc)
    np.testing.assert_allclose(scaled.coefficients, 3 * base.coefficients, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(scaled.energy, 9 * base.energy, rtol=1e-10, atol=1e-14)


def test_sign_reversal_keeps_energy():
    sc, _ = random_network()
    lap = fsc.graph_from_sc(sc).laplacian().toarray()
    eigenvalues, eigenvectors = laplacian_modes(lap)
    phi = np.random.default_rng(1).normal(size=len(sc))

    c_plus, e_plus = decompose(phi, eigenvalues, eigenvectors)
    c_minus, e_minus = decompose(-phi, eigenvalues, eigenvectors)
    np.testing.assert_allclose(c_minus, -c_plus)
    np.testing.assert_allclose(e_minus, e_plus)


def test_disconnected_graph():
    # Components {0, 1, 2} and {3, 4}, and the isolated node 5.
    sc = np.zeros((6, 6))
    for i, j in [(0, 1), (1, 2), (3, 4)]:
        sc[i, j] = sc[j, i] = 1.0
    lap = fsc.graph_from_sc(sc).laplacian().toarray()
    eigenvalues, _ = laplacian_modes(lap)
    assert len(eigenvalues) == 6 - 3                     # one zero mode per component
    assert np.all(eigenvalues > 0)

    fc = np.zeros((6, 6))
    fc[0, 2] = fc[2, 0] = 1.0                            # within a component
    fc[0, 3] = fc[3, 0] = 1.0                            # across components: no path
    spec = fsc.spectral_from_parcellations(sc, fc)
    assert spec.pairs.tolist() == [[0, 2]]
    np.testing.assert_allclose(spec.captured, 1.0)


def test_truncated_basis_reports_captured_fraction():
    sc, fc = random_network()
    full = fsc.spectral_from_parcellations(sc, fc)
    part = fsc.spectral_from_parcellations(sc, fc, n_modes=3)

    assert part.eigenvalues.shape == (3,)
    np.testing.assert_allclose(part.eigenvalues, full.eigenvalues[:3])
    np.testing.assert_allclose(part.energy, full.energy[:, :3], rtol=1e-10)
    np.testing.assert_allclose(
        part.captured, full.energy[:, :3].sum(axis=1) / full.energy.sum(axis=1), rtol=1e-10
    )
    assert np.all(part.captured < 1)


def test_joint_spectrum():
    sc, fc = five_node()
    fc[3, 4] = fc[4, 3] = 1.0                            # two pairs sharing node 4: no cycle
    spec = fsc.spectral_from_parcellations(sc, fc, method="joint")

    # Joint potentials 0.8, 1, 0.6, 1, 0 give edge currents 0.2, 0.2, 0.4, 0.6:
    # dissipated energy 0.04 + 0.04 + 0.16 + 0.36 = 0.6.
    assert spec.energy.shape[0] == 1
    assert spec.energy.sum() == pytest.approx(0.6, rel=1e-10)
    np.testing.assert_allclose(spec.captured, 1.0)


def test_joint_spectrum_rejects_cycles():
    sc, fc = random_network()
    with pytest.raises(ValueError, match="cycle"):
        fsc.spectral_from_parcellations(sc, fc, method="joint")
