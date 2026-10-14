import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fsc import graph_from_edges
from fsc.voxel.fod import VoxelMapping
from fsc.voxel.roi import ParcellationMapping, add_roi_nodes


def small_voxel():
    graph = graph_from_edges(3, [0, 1], [1, 2], [2, 2])
    mapping = VoxelMapping(np.arange(3).reshape(3, 1, 1), np.eye(4),
                           np.arange(2), np.array([0, 0]), np.array([1.]))
    return graph, mapping


def test_roi_interface_order_and_total_weight():
    graph, voxel = small_voxel()
    labels = np.array([2, 0, 1]).reshape(3, 1, 1)
    graph, parcellation = add_roi_nodes(graph, voxel, labels, [1, 2], dilation=0)
    assert isinstance(parcellation, ParcellationMapping)
    assert_array_equal(parcellation.labels, [1, 2])
    assert_array_equal(parcellation.supernodes, [3, 4])
    assert_array_equal(parcellation.interface_edge_indices, [2, 3])
    assert_array_equal(graph.edge_i, [0, 1, 3, 4])
    assert_array_equal(graph.edge_j, [1, 2, 2, 0])
    assert_allclose(graph.conductance[parcellation.interface_edge_indices], [2, 2])


def test_missing_roi_does_not_silently_disappear():
    graph, voxel = small_voxel()
    with pytest.raises(ValueError, match="no graph interfaces"):
        add_roi_nodes(graph, voxel, np.zeros((3, 1, 1)), [1], dilation=0)


def test_interface_weight_is_split_across_boundary_nodes():
    graph, voxel = small_voxel()
    labels = np.array([1, 1, 2]).reshape(3, 1, 1)
    graph, parcellation = add_roi_nodes(graph, voxel, labels, [1, 2],
                                       dilation=0, conductance=6)
    assert_allclose(graph.conductance[parcellation.interface_edge_indices], [3, 3, 6])
