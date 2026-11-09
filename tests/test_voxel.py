"""Synthetic voxel graph tests; no MRI datasets or DIPY installation needed."""

import numpy as np
import pytest
import fsc
from fsc.voxel import ParcellationMapping, VoxelMapping, voxel_support
from pathlib import Path
import inspect

def synthetic_graph():
    # WM graph: 0 -- 1 -- 2; ROI nodes 3, 4, 5 attach to WM 0, 2, 1.
    graph = fsc.WeightedGraph(
        n_nodes=6,
        edge_i=np.array([0, 1, 3, 4, 5]),
        edge_j=np.array([1, 2, 0, 2, 1]),
        conductance=np.ones(5),
    )
    node_id = np.full((5, 3, 3), -1)
    node_id[1:4, 1, 1] = np.arange(3)
    voxel = VoxelMapping(node_id, np.eye(4), np.array([0, 1]))
    parcels = ParcellationMapping(
        roi_labels=np.array([10, 20, 30]),
        roi_nodes=np.array([3, 4, 5]),
        interface_edge_indices=np.array([2, 3, 4]),
        contact_voxel_indices=np.array([0, 1, 2]),
    )
    return graph, voxel, parcels


def test_single_pair_and_voxel_support():
    graph, voxel, parcels = synthetic_graph()
    fc = np.zeros((3, 3))
    fc[0, 1] = fc[1, 0] = 1.0
    currents = fsc.solve_fsc_voxel(graph, parcels, fc, solver="cg", rtol=1e-12)
    np.testing.assert_allclose(currents, [0.25, 0.25, 0.25, 0.25, 0.0], atol=1e-12)
    support = voxel_support(graph, voxel, currents)
    np.testing.assert_allclose(support[1:4, 1, 1], [0.25, 0.5, 0.25], atol=1e-12)
    assert np.count_nonzero(support) == 3


def test_roi_basis_matches_individual_pairs():
    graph, _, parcels = synthetic_graph()

    fc = np.array([
        [0.0, 0.8, -0.4],
        [0.8, 0.0, 0.6],
        [-0.4, 0.6, 0.0],
    ])

    combined = fsc.solve_fsc_voxel(graph, parcels, fc, rtol=1e-12)

    summed = np.zeros(len(graph.edge_i))
    for pair in [(0, 1), (0, 2), (1, 2)]:
        summed += fsc.solve_fsc_voxel(
            graph, parcels, fc, pairs=[pair], rtol=1e-12
        )

    np.testing.assert_allclose(combined, summed, atol=1e-12)

    # Negative correlations must not contribute.
    positive_fc = np.maximum(fc, 0)
    np.testing.assert_allclose(
        combined,
        fsc.solve_fsc_voxel(graph, parcels, positive_fc, rtol=1e-12),
        atol=1e-12,
    )

    np.testing.assert_array_equal(
        fsc.solve_fsc_voxel(graph, parcels, np.zeros_like(fc)),
        np.zeros(len(graph.edge_i)),
    )


def test_disconnected_rois_have_no_support():
    graph = fsc.WeightedGraph(
        n_nodes=4,
        edge_i=np.array([2, 3]),
        edge_j=np.array([0, 1]),
        conductance=np.ones(2),
    )
    parcels = ParcellationMapping(
        roi_labels=np.array([1, 2]),
        roi_nodes=np.array([2, 3]),
        interface_edge_indices=np.array([0, 1]),
        contact_voxel_indices=np.array([0, 1]),
    )
    fc = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(
        fsc.solve_fsc_voxel(graph, parcels, fc), np.zeros(2)
    )


def test_synthetic_fod_example():
    """Run the complete image-to-support example when optional deps exist."""
    import runpy
    from pathlib import Path
    import pytest

    pytest.importorskip("nibabel")
    pytest.importorskip("dipy")
    script = Path(__file__).resolve().parents[1] / "examples" / "voxel.py"
    output = runpy.run_path(str(script))
    support = output["support"]
    assert np.all(support[2:7, 2, 2] > 0)


def lattice_graph(nx, ny, conductance):
    """nx-by-ny WM lattice; ROI A touches column x=0, ROI B touches x=nx-1."""
    idx = np.arange(nx * ny).reshape(nx, ny)
    h_i, h_j = idx[:-1, :].ravel(), idx[1:, :].ravel()
    v_i, v_j = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    n_wm = nx * ny
    roi_a, roi_b = n_wm, n_wm + 1
    left, right = idx[0, :], idx[-1, :]

    edge_i = np.concatenate((h_i, v_i, np.full(ny, roi_a), np.full(ny, roi_b)))
    edge_j = np.concatenate((h_j, v_j, left, right))
    graph = fsc.WeightedGraph(
        n_nodes=n_wm + 2,
        edge_i=edge_i,
        edge_j=edge_j,
        conductance=np.full(len(edge_i), conductance),
    )
    n_roi_edges = 2 * ny
    parcels = ParcellationMapping(
        roi_labels=np.array([1, 2]),
        roi_nodes=np.array([roi_a, roi_b]),
        interface_edge_indices=np.arange(len(edge_i) - n_roi_edges, len(edge_i)),
        contact_voxel_indices=np.arange(n_roi_edges),
    )
    return graph, parcels


@pytest.mark.parametrize("conductance", [1e-2, 1.0, 1e3, 1e6])
def test_convergence_check_is_scale_invariant(conductance):
    """A correct CG solution must not be rejected because conductances are large."""
    graph, parcels = lattice_graph(30, 30, conductance)
    fc = np.array([[0.0, 0.7], [0.7, 0.0]])

    # Default rtol on purpose: it lets CG stop near the tolerance,
    # which is what exposes the faulty residual check.
    cg_currents = fsc.solve_fsc_voxel(graph, parcels, fc, solver="cg")
    reference = fsc.solve_fsc_voxel(graph, parcels, fc, solver="direct")

    np.testing.assert_allclose(
        cg_currents, reference, rtol=1e-2, atol=1e-3 * reference.max()
    )

def test_default_sh_basis_matches_mrtrix3():
    """Default SH basis must decode MRtrix3 FOD coefficients like sh2amp."""
    pytest.importorskip("dipy")
    from fsc.voxel.fod import fod_kernels

    ref = np.load(Path(__file__).parent / "data" / "mrtrix3_sh_reference.npz")
    coefficients = ref["coefficients"]
    _, sh_basis = fod_kernels(coefficients.shape[1], np.eye(3))  # defaults, any directions

    predicted = coefficients @ sh_basis[ref["vertex_indices"]].T
    np.testing.assert_allclose(predicted, ref["amplitudes"], atol=1e-5)


def test_sh_basis_defaults_agree():
    """graph_from_fod and add_roi_nodes must default to the same basis."""
    from fsc.voxel.fod import fod_kernels, graph_from_fod
    from fsc.voxel.roi import add_roi_nodes

    def defaults(fn):
        p = inspect.signature(fn).parameters
        return (p["sphere"].default, p["basis"].default, p["legacy"].default)

    assert defaults(fod_kernels) == defaults(graph_from_fod) == defaults(add_roi_nodes)

    assert (inspect.signature(graph_from_fod).parameters["length_power"].default == inspect.signature(add_roi_nodes).parameters["length_power"].default)


def _tube_phantom(direction, n=24, length=10, radius=2.0, k=16.0):
    """Straight fibre tube through the grid centre, capped by ROIs 1 and 2."""
    import nibabel as nib
    from dipy.data import get_sphere
    from dipy.reconst.shm import sf_to_sh

    sphere = get_sphere(name="repulsion724")
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    sh = sf_to_sh(np.exp(k * ((sphere.vertices @ d) ** 2 - 1)), sphere,
                  sh_order_max=8, basis_type="tournier07", legacy=False)

    grid = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), axis=-1) - n // 2
    t = grid @ d
    r = np.linalg.norm(grid - t[..., None] * d, axis=-1)
    inside = r <= radius

    wm = inside & (np.abs(t) <= length / 2)
    labels = np.zeros((n, n, n), int)
    labels[inside & (t > length / 2) & (t <= length / 2 + 2)] = 1
    labels[inside & (t < -length / 2) & (t >= -length / 2 - 2)] = 2

    fod = np.zeros((n, n, n, len(sh)), np.float32)
    fod[inside] = sh
    return nib.Nifti1Image(fod, np.eye(4)), wm, labels


@pytest.mark.parametrize("direction", [(1, 0, 0), (1, 1, 0), (1, 1, 1)])
def test_roi_contacts_for_oblique_tracts(direction):
    """An ROI must connect to a tract whatever its angle to the voxel grid."""
    pytest.importorskip("nibabel")
    pytest.importorskip("dipy")
    from fsc.voxel import graph_from_fod, add_roi_nodes

    image, wm, labels = _tube_phantom(direction)
    wm_graph, voxel = graph_from_fod(image, wm_mask=wm, fod_norm_threshold=0.0)
    graph, parc = add_roi_nodes(wm_graph, voxel, image, labels, roi_labels=[1, 2])

    fc = np.array([[0.0, 1.0], [1.0, 0.0]])
    currents = fsc.solve_fsc_voxel(graph, parc, fc, solver="direct")

    idx = parc.interface_edge_indices
    out_of_roi1 = idx[graph.edge_i[idx] == parc.roi_nodes[0]]
    assert np.abs(currents[out_of_roi1]).sum() > 0

def _tube_resistance(direction, length):
    """Effective resistance between the ROIs of a k=64 tube phantom."""
    from fsc.voxel import graph_from_fod, add_roi_nodes

    image, wm, labels = _tube_phantom(direction, n=32, length=length, radius=3.0, k=64.0)
    wm_graph, voxel = graph_from_fod(image, wm_mask=wm, fod_norm_threshold=0.0)
    graph, parc = add_roi_nodes(wm_graph, voxel, image, labels, roi_labels=[1, 2])

    fc = np.array([[0.0, 1.0], [1.0, 0.0]])
    currents = fsc.solve_fsc_voxel(graph, parc, fc, solver="direct")
    idx = parc.interface_edge_indices
    out_of_roi1 = idx[graph.edge_i[idx] == parc.roi_nodes[0]]
    return 1.0 / np.abs(currents[out_of_roi1]).sum()


@pytest.mark.parametrize("direction", [(1, 1, 0), (1, 1, 1)],
                         ids=["face-diagonal", "body-diagonal"])
def test_aligned_tracts_conduct_equally(direction):
    """A tract along a lattice diagonal conducts like one along an axis."""
    pytest.importorskip("nibabel")
    pytest.importorskip("dipy")

    def per_mm(d):
        # Two lengths: the difference cancels the ROI end effects.
        return (_tube_resistance(d, 20) - _tube_resistance(d, 10)) / 10

    ratio = per_mm(direction) / per_mm((1, 0, 0))
    assert ratio == pytest.approx(1.0, abs=0.15)


def test_length_power_one_reproduces_paper_weights():
    """length_power=1 gives Eq. 4 weights: each edge is l times the default."""
    pytest.importorskip("nibabel")
    pytest.importorskip("dipy")
    from fsc.voxel import graph_from_fod

    image, wm, _ = _tube_phantom((1, 1, 0))
    g1, _ = graph_from_fod(image, wm_mask=wm, fod_norm_threshold=0.0, length_power=1)
    g2, voxel = graph_from_fod(image, wm_mask=wm, fod_norm_threshold=0.0)

    coords = np.argwhere(voxel.node_id >= 0)
    ell = np.linalg.norm(coords[g2.edge_i] - coords[g2.edge_j], axis=1)
    np.testing.assert_array_equal(g1.edge_i, g2.edge_i)
    np.testing.assert_allclose(g1.conductance, g2.conductance * ell, rtol=1e-5)
