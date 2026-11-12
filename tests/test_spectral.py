"""Tests for the spectral decomposition of parcellation FSC solutions."""

import numpy as np
import pytest

import fsc
from fsc.spectral import decompose, laplacian_modes
from fsc.voxel import ParcellationMapping


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

def lattice_with_supernodes(nx=6, ny=5, extra=False, seed=0):
    """WM lattice with random conductances. Supernode A touches the left
    column, B the right column; with extra=True, C touches the bottom row and
    D the top row. A and B are identical with and without C and D."""
    rng = np.random.default_rng(seed)
    idx = np.arange(nx * ny).reshape(nx, ny)
    wm_i = np.r_[idx[:-1, :].ravel(), idx[:, :-1].ravel()]
    wm_j = np.r_[idx[1:, :].ravel(), idx[:, 1:].ravel()]
    wm_g = rng.uniform(0.5, 2.0, len(wm_i))

    n_wm = nx * ny
    contacts = [idx[0, :], idx[-1, :]]
    if extra:
        contacts += [idx[:, 0], idx[:, -1]]
    roi_nodes = n_wm + np.arange(len(contacts))
    c_i = np.concatenate([np.full(len(c), r) for c, r in zip(contacts, roi_nodes)])
    c_j = np.concatenate(contacts)
    c_g = 1.0 + 0.5 * (c_j % 3)                          # deterministic contact strengths

    graph = fsc.WeightedGraph(n_wm + len(contacts), np.r_[wm_i, c_i],
                              np.r_[wm_j, c_j], np.r_[wm_g, c_g])
    parcellation = ParcellationMapping(
        roi_labels=np.arange(1, len(contacts) + 1),
        roi_nodes=roi_nodes,
        interface_edge_indices=np.arange(len(wm_i), len(wm_i) + len(c_i)),
        contact_voxel_indices=np.arange(len(c_i)),
    )
    return graph, parcellation


def pair_resistance(graph, a_node, b_node, excluded_nodes=()):
    """Effective resistance between two supernodes, other supernodes removed."""
    keep = ~(np.isin(graph.edge_i, excluded_nodes) | np.isin(graph.edge_j, excluded_nodes))
    lap = fsc.WeightedGraph(graph.n_nodes, graph.edge_i[keep], graph.edge_j[keep],
                            graph.conductance[keep]).laplacian().toarray()
    a = np.zeros(graph.n_nodes)
    a[a_node], a[b_node] = 1.0, -1.0
    return a @ np.linalg.pinv(lap) @ a


def test_voxel_modes_full_basis():
    from fsc.spectral import voxel_modes
    graph, parcellation = lattice_with_supernodes()
    modes = voxel_modes(graph, parcellation)

    n_wm = 6 * 5
    wm = (graph.edge_i < n_wm) & (graph.edge_j < n_wm)
    lap = fsc.WeightedGraph(n_wm, graph.edge_i[wm], graph.edge_j[wm],
                            graph.conductance[wm]).laplacian().toarray()
    expected = np.linalg.eigvalsh(lap)[1:]               # connected lattice: one zero mode
    np.testing.assert_allclose(modes.eigenvalues, expected, rtol=1e-10)
    assert modes.eigenvectors.shape == (n_wm, n_wm - 1)


def test_voxel_energy_balance():
    from fsc.spectral import spectral_fsc_voxel, voxel_modes
    graph, parcellation = lattice_with_supernodes()
    fc = np.array([[0.0, 0.7], [0.7, 0.0]])
    spec = spectral_fsc_voxel(graph, parcellation, fc, voxel_modes(graph, parcellation),
                              solver="direct")

    a, b = parcellation.roi_nodes
    resistance = pair_resistance(graph, a, b)
    np.testing.assert_allclose(spec.resistance, [resistance], rtol=1e-10)

    total = 0.7 ** 2 / resistance
    wm_main = total * (1 - spec.contact_share - spec.other_share)
    np.testing.assert_allclose(spec.energy.sum(axis=1), wm_main, rtol=1e-8)
    np.testing.assert_allclose(spec.captured, 1.0, rtol=1e-8)   # full basis
    assert 0 < spec.contact_share[0] < 1
    assert spec.other_share[0] == pytest.approx(0.0, abs=1e-12)


def test_voxel_fc_scaling():
    from fsc.spectral import spectral_fsc_voxel, voxel_modes
    graph, parcellation = lattice_with_supernodes()
    modes = voxel_modes(graph, parcellation)
    fc = np.array([[0.0, 0.7], [0.7, 0.0]])
    base = spectral_fsc_voxel(graph, parcellation, fc, modes, solver="direct")
    scaled = spectral_fsc_voxel(graph, parcellation, 3 * fc, modes, solver="direct")
    np.testing.assert_allclose(scaled.energy, 9 * base.energy, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(scaled.captured, base.captured, rtol=1e-8)
    np.testing.assert_allclose(scaled.contact_share, base.contact_share, rtol=1e-8)


def test_voxel_spectra_ignore_unrelated_supernodes():
    from fsc.spectral import spectral_fsc_voxel, voxel_modes
    g2, p2 = lattice_with_supernodes()
    g4, p4 = lattice_with_supernodes(extra=True)
    fc2 = np.array([[0.0, 0.7], [0.7, 0.0]])
    fc4 = np.zeros((4, 4))
    fc4[0, 1] = fc4[1, 0] = 0.7

    s2 = spectral_fsc_voxel(g2, p2, fc2, voxel_modes(g2, p2), solver="direct")
    s4 = spectral_fsc_voxel(g4, p4, fc4, voxel_modes(g4, p4), solver="direct")
    np.testing.assert_allclose(s4.resistance, s2.resistance, rtol=1e-10)
    np.testing.assert_allclose(s4.energy, s2.energy, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(s4.contact_share, s2.contact_share, rtol=1e-8)


def test_voxel_partial_modes_with_lobpcg():
    pytest.importorskip("pyamg")
    from fsc.spectral import spectral_fsc_voxel, voxel_modes
    graph, parcellation = lattice_with_supernodes(nx=30, ny=30)
    full = voxel_modes(graph, parcellation)
    part = voxel_modes(graph, parcellation, n_modes=10)

    np.testing.assert_allclose(part.eigenvalues, full.eigenvalues[:10], rtol=1e-4)
    assert part.residuals.max() < 1e-2

    fc = np.array([[0.0, 0.7], [0.7, 0.0]])
    s_full = spectral_fsc_voxel(graph, parcellation, fc, full, solver="direct")
    s_part = spectral_fsc_voxel(graph, parcellation, fc, part, solver="direct")
    expected = s_full.energy[:, :10].sum(axis=1) / s_full.energy.sum(axis=1)
    np.testing.assert_allclose(s_part.captured, expected, rtol=1e-3)
    assert np.all(s_part.captured < 1)
