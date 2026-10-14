import numpy as np
from numpy.testing import assert_allclose

from fsc import graph_from_edges
from fsc.voxel.fod import VoxelMapping
from fsc.voxel.support import support_on_nodes, voxel_support, sum_pairwise_support
from fsc.voxel.roi import add_roi_nodes


def make_voxel():
    graph = graph_from_edges(3, [0, 1], [1, 2], [1, 1])
    voxel = VoxelMapping(np.arange(3).reshape(3, 1, 1), np.eye(4),
                         np.arange(2), np.array([0, 0]), np.array([1.]))
    return graph, voxel


def test_incident_current_accumulation_and_double_counting():
    support = support_on_nodes(3, [0, 1], [1, 2], [2, -3])
    assert_allclose(support, [2, 5, 3])
    assert_allclose(support.sum(), 2 * (2 + 3))


def test_joint_and_pairwise_workflow_on_roi_graph():
    graph, voxel = make_voxel()
    labels = np.array([1, 0, 2]).reshape(3, 1, 1)
    graph, parcellation = add_roi_nodes(graph, voxel, labels, [1, 2],
                                        dilation=0, conductance=1)
    fc = np.array([[0, 3], [3, 0]])
    support = sum_pairwise_support(graph, voxel, parcellation, fc)
    assert_allclose(support[:, 0, 0], [.75, 1.5, .75], atol=1e-9)
    extra = np.r_[[1, 1], [50, 50]]
    assert_allclose(voxel_support(graph, voxel, extra)[:, 0, 0], [1, 2, 1])


def test_explicit_voxel_edge_indices_survive_graph_edge_reordering():
    graph, voxel = make_voxel()
    graph, parcellation = add_roi_nodes(graph, voxel,
                                        np.array([1, 0, 2]).reshape(3, 1, 1),
                                        [1, 2], dilation=0, conductance=1)
    original_currents = np.array([2., -3., 20., 40.])
    original_support = voxel_support(graph, voxel, original_currents)
    permutation = np.array([2, 0, 3, 1])
    reordered_graph = graph_from_edges(graph.n_nodes, graph.edge_i[permutation],
                                      graph.edge_j[permutation],
                                      graph.conductance[permutation])
    # Edge metadata follows the voxel-edge order, not the sorted graph edge indices.
    voxel.edge_indices = np.array([np.flatnonzero(permutation == i)[0] for i in [0, 1]])
    assert_allclose(voxel_support(reordered_graph, voxel, original_currents[permutation]),
                    original_support)
