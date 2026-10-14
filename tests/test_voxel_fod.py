from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from fsc import WeightedGraph
from fsc.voxel.fod import (VoxelMapping, directional_support, fod_edges, graph_from_fod,
                             voxelhood_geometrys)


class Image:
    def __init__(self, data, affine=None):
        self.data = np.asarray(data, dtype=np.float32)
        self.affine = np.eye(4) if affine is None else affine

    def get_fdata(self, dtype):
        return self.data.astype(dtype)


def test_directional_support_and_two_endpoint_rule():
    coeff = np.array([[2., 1.], [.5, 1.]])
    kernel, isotropic = [1., 0.], [0., 1.]
    assert_allclose(directional_support(coeff, kernel, isotropic), [1, 0])
    fod = coeff.reshape(2, 1, 1, 2)
    args = (np.ones((2, 1, 1), bool), np.arange(2).reshape(2, 1, 1),
            np.array([[1, 0, 0]]), [2.], np.array([kernel]), np.array(isotropic))
    with pytest.raises(ValueError, match="No FOD-supported"):
        fod_edges(fod, *args)
    fod[1, 0, 0, 0] = 5
    first, second, weights, offset_ids = fod_edges(fod, *args)
    assert_array_equal(first, [0])
    assert_array_equal(second, [1])
    assert_allclose(weights, [1])
    assert_array_equal(offset_ids, [0])


def test_offsets_use_physical_affine():
    affine = np.diag([2, 3, 4, 1])
    offsets, lengths, directions = voxelhood_geometrys(affine)
    assert len(offsets) == 13
    assert_allclose(lengths, np.linalg.norm(offsets * [2, 3, 4], axis=1))
    assert_allclose(np.linalg.norm(directions, axis=1), 1)


def test_complete_graph_with_controlled_fod_kernel():
    image = Image(np.full((3, 1, 1, 2), [2., 1.]))
    with patch("fsc.voxel.fod.fod_kernels", return_value=(
            np.tile([1., 0.], (13, 1)), np.array([0., 1.]))):
        graph, voxel = graph_from_fod(image, wm_mask=np.ones((3, 1, 1), bool))
    assert isinstance(graph, WeightedGraph)
    assert isinstance(voxel, VoxelMapping)
    assert graph.n_nodes == 3
    assert_allclose(graph.conductance, [1, 1])
    assert_array_equal(voxel.to_volume([5, 6, 7])[:, 0, 0], [5, 6, 7])


def test_legacy_real_fod_reference_if_dipy_available():
    nib = pytest.importorskip("nibabel")
    pytest.importorskip("dipy")
    from pathlib import Path
    with np.load(Path(__file__).parent / "data/legacy_reference.npz") as ref:
        graph, voxel = graph_from_fod(nib.Nifti1Image(ref["fod"], ref["affine"]))
        assert_array_equal(graph.edge_i, ref["edge_i"])
        assert_array_equal(graph.edge_j, ref["edge_j"])
        assert_allclose(graph.conductance, ref["conductance"], atol=1e-7)


def test_mask_and_exclusion_remove_nodes():
    image = Image(np.full((3, 1, 1, 2), [2., 1.]))
    mask = np.ones((3, 1, 1), bool)
    exclusion = np.zeros_like(mask)
    exclusion[2] = True
    with patch("fsc.voxel.fod.fod_kernels", return_value=(
            np.tile([1., 0.], (13, 1)), np.array([0., 1.]))):
        graph, voxel = graph_from_fod(image, wm_mask=mask, exclusion_mask=exclusion)
    assert graph.n_nodes == 2
    assert voxel.node_id[2, 0, 0] == -1
    assert_allclose(voxel.to_volume([7, 8])[:, 0, 0], [7, 8, 0])


def test_chunking_does_not_change_voxel_edges():
    fod = np.tile([3., 1.], (4, 1, 1, 1))
    mask = np.ones((4, 1, 1), bool)
    node_id = np.arange(4).reshape(4, 1, 1)
    args = (fod, mask, node_id, np.array([[1, 0, 0]]),
            np.array([2.]), np.array([[1., 0.]]), np.array([0., 1.]))
    one = fod_edges(*args, chunk_size=1)
    many = fod_edges(*args, chunk_size=100)
    for a, b in zip(one, many):
        assert_array_equal(a, b)
