"""Regression against numerical results saved from the unmodified older branch."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fsc import graph_from_edges, solve_fsc, constraints_from_fc
from fsc.voxel.fod import VoxelMapping
from fsc.voxel.roi import add_roi_nodes
from fsc.voxel.support import voxel_support, sum_pairwise_support


def test_voxel_interface_and_solver_against_previous_branch():
    # Precomputed voxel edges let this regression run without DIPY/NiBabel.
    with np.load(Path(__file__).parent / "data/legacy_reference.npz") as ref:
        graph = graph_from_edges(45, ref["edge_i"], ref["edge_j"], ref["conductance"])
        voxel = VoxelMapping(np.arange(45).reshape(5, 3, 3), ref["affine"],
                             np.arange(len(graph.edge_i)), ref["edge_k"], np.ones(13))
        graph, parcellation = add_roi_nodes(graph, voxel, ref["labels"],
                                            [1, 2], dilation=0)
        assert_array_equal(graph.conductance[parcellation.interface_edge_indices],
                           ref["interface_conductance"])
        fc = np.array([[0, .8], [.8, 0]])
        pairs, values = constraints_from_fc(fc, parcellation.supernodes)
        result = solve_fsc(graph, pairs, values, rtol=1e-12)
        assert_allclose(result.potentials, ref["voxel_phi"], atol=1e-8)
        assert_allclose(result.edge_currents[voxel.edge_indices],
                        ref["voxel_currents"], atol=1e-8)
        assert_allclose(voxel_support(graph, voxel, result.edge_currents),
                        ref["voxel_support"], atol=1e-7)
        assert_allclose(sum_pairwise_support(graph, voxel, parcellation, fc, rtol=1e-12),
                        ref["voxel_support"], atol=1e-7)
