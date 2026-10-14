import numpy as np
from numpy.testing import assert_allclose

from fsc import graph_from_edges, solve_fsc
from fsc.voxel.fod import VoxelMapping
from fsc.voxel.roi import ParcellationMapping
from fsc.voxel.paths import extract_roi_paths, paths_to_streamlines


def test_downhill_paths_follow_current_field():
    voxel = VoxelMapping(np.arange(3).reshape(3, 1, 1), np.eye(4),
                         np.arange(2), np.array([0, 0]), np.array([1.]))
    graph = graph_from_edges(5, [0, 1, 3, 4], [1, 2, 0, 2], [1, 1, 1, 1])
    parcellation = ParcellationMapping(np.array([1, 2]),
                                       np.array([3, 4]), np.array([2, 3]))
    result = solve_fsc(graph, [[3, 4]], [4])
    extracted = extract_roi_paths(graph, voxel, parcellation, result, 0, 1)
    assert extracted["path_mask"].tolist() == [True, True, True]
    streamlines = paths_to_streamlines(extracted["forward"].paths, voxel)
    assert_allclose(streamlines[0], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def test_flux_selection_prefers_major_endpoints():
    from fsc.voxel.paths import select_flux_nodes
    selected = select_flux_nodes([2, 3, 4], [.7, .2, .1], cumulative_fraction=.8)
    assert selected.tolist() == [2, 3]


def test_disconnected_path_has_no_results():
    import scipy.sparse as sparse
    from fsc.voxel.paths import fast_endpoint_paths_to_roi
    graph = sparse.csr_matrix((3, 3))
    result = fast_endpoint_paths_to_roi(graph, [0], [2])
    assert not result.reached[0]
    assert not result.paths
