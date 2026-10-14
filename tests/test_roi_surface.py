"""Synthetic tests for structural ROI boundaries and functional constraints."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import fsc
from fsc.voxel import roi
from fsc.voxel.support import sum_pairwise_support, voxel_support


class FakeImage:
    def __init__(self, data):
        self.data = data
        self.affine = np.eye(4)

    def get_fdata(self, dtype=np.float32):
        return self.data.astype(dtype)


def fake_fod_evaluation(monkeypatch):
    """Use FOD amplitude as support, independently of direction for topology tests."""
    monkeypatch.setattr(
        roi, "fod_kernels",
        lambda count, directions, **options: (
            np.ones((len(directions), count)), np.ones((1, count))
        ),
    )
    monkeypatch.setattr(
        roi, "positive_fod_mean",
        lambda coefficients, basis: np.zeros(len(coefficients)),
    )
    monkeypatch.setattr(
        roi, "directional_support",
        lambda coefficients, kernel, baseline, orientation_floor:
            np.maximum(coefficients[:, 0], 0),
    )


def create_wm_graph(shape, wm_coords, edges=()):
    node_id = np.full(shape, -1, dtype=int)
    for index, coord in enumerate(wm_coords):
        node_id[coord] = index
    voxel = SimpleNamespace(
        node_id=node_id, n_voxels=len(wm_coords), affine=np.eye(4),
        edge_indices=np.arange(len(edges)),
        to_volume=lambda values: np.asarray(values)[node_id.clip(0)] * (node_id >= 0),
    )
    graph = fsc.WeightedGraph(
        len(wm_coords),
        np.array([edge[0] for edge in edges], dtype=int),
        np.array([edge[1] for edge in edges], dtype=int),
        np.ones(len(edges)),
    )
    return graph, voxel


def test_only_active_roi_surface_is_included(monkeypatch):
    fake_fod_evaluation(monkeypatch)
    # A occupies x=1,2,3. Its interior x=2 must not become a graph node.
    labels = np.zeros((7, 1, 1), dtype=int)
    labels[1:4] = 17
    labels[6] = 18
    graph, voxel = create_wm_graph(
        labels.shape, [(0, 0, 0), (4, 0, 0), (5, 0, 0)], [(1, 2)]
    )
    image = FakeImage(np.ones(labels.shape + (1,)))
    augmented, mapping = roi.add_roi_nodes(graph, voxel, image, labels, [17, 18])

    assert_array_equal(mapping.roi_labels, [17, 18])
    assert_array_equal(mapping.surface_labels, [17, 17, 18])
    assert_array_equal(mapping.surface_nodes, [3, 4, 5])
    assert augmented.n_nodes == 6
    assert len(mapping.interface_edge_indices) == 3
    assert len(mapping.surface_edge_indices) == 0
    assert not np.any((augmented.edge_i == 3) & (augmented.edge_j == 4))
    assert_array_equal(np.sort(augmented.edge_i[mapping.interface_edge_indices]), [3, 4, 5])


def test_surface_edges_require_support_at_both_endpoints(monkeypatch):
    fake_fod_evaluation(monkeypatch)
    labels = np.zeros((4, 2, 1), dtype=int)
    labels[1] = 17
    labels[3, 0] = 18
    wm_coords = [(0, 0, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)]
    graph, voxel = create_wm_graph(labels.shape, wm_coords, [(0, 1), (1, 3), (3, 2)])
    data = np.ones(labels.shape + (1,))
    image = FakeImage(data)

    supported_graph, mapping = roi.add_roi_nodes(graph, voxel, image, labels, [17, 18])
    assert len(mapping.surface_edge_indices) == 1
    e = mapping.surface_edge_indices[0]
    assert {supported_graph.edge_i[e], supported_graph.edge_j[e]} == set(mapping.surface_nodes[:2])

    data[1, 1, 0] = 0   # Surface A1 still contacts WM, but lacks FOD support itself.
    unsupported_graph, mapping = roi.add_roi_nodes(graph, voxel, image, labels, [17, 18])
    assert len(mapping.surface_edge_indices) == 0
    assert len(mapping.interface_edge_indices) >= 3


def test_equal_boundary_potentials_and_no_surface_current():
    graph = fsc.WeightedGraph(
        6,
        np.array([0, 1, 3, 4, 5, 3]),
        np.array([1, 2, 0, 1, 2, 4]),
        np.array([1.0, 1.0, 2.0, 2.0, 2.0, 7.0]),
    )
    mapping = roi.ParcellationMapping(
        roi_labels=np.array([17, 18]),
        surface_labels=np.array([17, 17, 18]),
        surface_nodes=np.array([3, 4, 5]),
        surface_edge_indices=np.array([5]),
        interface_edge_indices=np.array([2, 3, 4]),
    )
    incidence, targets = roi.roi_constraints(mapping, [[0, 1]], [0.8], graph.n_nodes)
    assert incidence.shape == (6, 2)
    assert_allclose(targets, [0.0, 0.8])

    result = fsc.solve_fsc(graph, incidence, targets, rtol=1e-12)
    assert_allclose(result.potentials[3], result.potentials[4], atol=1e-9)
    assert_allclose(result.potentials[3] - result.potentials[5], 0.8, atol=1e-9)
    assert_allclose(result.edge_currents[5], 0, atol=1e-9)
    assert result.relative_residual < 1e-9

    voxel = SimpleNamespace(
        node_id=np.arange(3).reshape(3, 1, 1), n_voxels=3,
        edge_indices=np.array([0, 1]),
        to_volume=lambda values: np.asarray(values).reshape(3, 1, 1),
    )
    fc = np.array([[0, 0.8], [0.8, 0]])
    assert_allclose(
        sum_pairwise_support(graph, voxel, mapping, fc, rtol=1e-12),
        voxel_support(graph, voxel, result.edge_currents),
    )


def test_constraints_keep_original_fc_order():
    mapping = roi.ParcellationMapping(
        roi_labels=np.array([18, 17]),
        surface_labels=np.array([17, 17, 18]),
        surface_nodes=np.array([3, 4, 5]),
        surface_edge_indices=np.array([], dtype=int),
        interface_edge_indices=np.array([], dtype=int),
    )
    incidence, values = roi.roi_constraints(mapping, [[0, 1]], [-0.7], 6)
    assert_allclose(incidence.toarray()[:, 0], [0, 0, 0, -1, 1, 0])
    assert_allclose(incidence.toarray()[:, 1], [0, 0, 0, -1, 0, 1])
    assert_allclose(values, [0, -0.7])


def test_diagonal_contact_does_not_create_roi_to_wm_edge(monkeypatch):
    fake_fod_evaluation(monkeypatch)
    labels = np.zeros((2, 2, 1), dtype=int)
    labels[0, 0, 0] = 17
    graph, voxel = create_wm_graph(labels.shape, [(1, 1, 0)])
    with pytest.raises(ValueError, match="No FOD-supported ROI-to-white-matter contacts"):
        roi.add_roi_nodes(graph, voxel, FakeImage(np.ones(labels.shape + (1,))), labels)


def test_interface_conductances_not_normalized_by_number_of_contacts(monkeypatch):
    fake_fod_evaluation(monkeypatch)
    labels = np.zeros((4, 2, 1), dtype=int)
    labels[1] = 17
    labels[3, 0] = 18
    wm_coords = [(0, 0, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)]
    graph, voxel = create_wm_graph(labels.shape, wm_coords, [(0, 1), (1, 3), (3, 2)])
    augmented, mapping = roi.add_roi_nodes(
        graph, voxel, FakeImage(np.ones(labels.shape + (1,))), labels,
        [17, 18], interface_scale=3.0,
    )
    assert len(mapping.interface_edge_indices) == 5
    assert_allclose(augmented.conductance[mapping.interface_edge_indices], 3)


def test_tissue_mask_can_exclude_csf_label_voxels(monkeypatch):
    fake_fod_evaluation(monkeypatch)
    labels = np.zeros((4, 1, 1), dtype=int)
    labels[0] = 17
    labels[3] = 18
    graph, voxel = create_wm_graph(labels.shape, [(1, 0, 0), (2, 0, 0)], [(0, 1)])
    mask = np.ones(labels.shape, dtype=bool)
    mask[0] = False
    with pytest.raises(ValueError, match=r"ROIs \[17\]"):
        roi.add_roi_nodes(
            graph, voxel, FakeImage(np.ones(labels.shape + (1,))),
            labels, [17, 18], tissue_mask=mask
        )
