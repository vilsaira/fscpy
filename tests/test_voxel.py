"""Synthetic voxel graph tests; no MRI datasets or DIPY installation needed."""

import numpy as np

import fsc
from fsc.voxel import ParcellationMapping, VoxelMapping, voxel_support


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
    np.testing.assert_allclose(
        fsc.solve_fsc_voxel(graph, parcels, -fc, rtol=1e-12), combined
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
