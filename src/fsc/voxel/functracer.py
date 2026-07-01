#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""FunCTracer: functionally constrained tract tracing via voxelwise FSC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import nibabel as nib

from fsc.sparse import SparseFSC, SparseFSCResult, ConstraintMode, constraints_from_fc_matrix
from .fod_graph import FODGraph, build_fod_graph
from .roi_interface import VoxelROIInterface, build_roi_supernode_interface, resample_labels_to_reference
from .metrics import edge_current_to_node_scalar, effective_conductance_matrix
from .paths import extract_pair_paths


@dataclass(frozen=True)
class FunCTracerResult:
    """Solved FunCTracer result."""

    sparse_result: SparseFSCResult
    graph: FODGraph
    interface: VoxelROIInterface
    fc_matrix: np.ndarray
    voxel_edge_slice: slice
    interface_edge_slice: slice

    @property
    def phi(self) -> np.ndarray:
        return self.sparse_result.phi

    @property
    def voxel_phi(self) -> np.ndarray:
        return self.phi[:self.graph.n_nodes]

    @property
    def roi_phi(self) -> np.ndarray:
        return self.phi[self.interface.roi_node_ids]

    @property
    def voxel_edge_current(self) -> np.ndarray:
        return self.sparse_result.edge_currents[self.voxel_edge_slice]

    @property
    def interface_edge_current(self) -> np.ndarray:
        return self.sparse_result.edge_currents[self.interface_edge_slice]

    def voxel_potential_volume(self, fill_value=np.nan) -> np.ndarray:
        return self.graph.values_to_volume(self.voxel_phi, fill_value=fill_value, dtype=np.float32)

    def voxel_current_magnitude_volume(self) -> np.ndarray:
        node_current = edge_current_to_node_scalar(
            self.graph.n_nodes, self.graph.edge_i, self.graph.edge_j, self.voxel_edge_current
        )
        return self.graph.values_to_volume(node_current, fill_value=0.0, dtype=np.float32)

    def effective_conductance_matrix(self) -> np.ndarray:
        return effective_conductance_matrix(
            n_rois=self.fc_matrix.shape[0],
            constraint_i=self.sparse_result.constraint_i,
            constraint_j=self.sparse_result.constraint_j,
            constraint_values=self.sparse_result.constraint_values,
            source_currents=self.sparse_result.source_currents,
            roi_node_ids=self.interface.roi_node_ids,
        )

    def extract_paths_for_pair(self, roi_index_a: int, roi_index_b: int,
                               length_weight=0.05, cumulative_fraction=1.0,
                               min_relative_flux=0.0, min_nodes=1):
        """Extract current-guided voxel paths for one ROI pair."""
        roi_node_a = int(self.interface.roi_node_ids[int(roi_index_a)])
        roi_node_b = int(self.interface.roi_node_ids[int(roi_index_b)])
        all_edge_i = np.concatenate([self.graph.edge_i, self.interface.edge_i])
        all_edge_j = np.concatenate([self.graph.edge_j, self.interface.edge_j])
        return extract_pair_paths(
            n_voxel_nodes=self.graph.n_nodes,
            edge_i=self.graph.edge_i,
            edge_j=self.graph.edge_j,
            edge_k=self.graph.edge_k,
            edge_current=self.voxel_edge_current,
            offset_dist=self.graph.offset_dist,
            phi=self.phi,
            roi_node_a=roi_node_a,
            roi_node_b=roi_node_b,
            all_edge_i=all_edge_i,
            all_edge_j=all_edge_j,
            all_edge_current=self.sparse_result.edge_currents,
            length_weight=length_weight,
            cumulative_fraction=cumulative_fraction,
            min_relative_flux=min_relative_flux,
            min_nodes=min_nodes,
        )


class FunCTracer:
    """High-level voxelwise FSC workflow.

    FC values remain MNA constraints. The structural graph is the FOD-derived
    voxel conductance graph plus ROI-supernode interface edges.
    """

    def __init__(self, graph: FODGraph, interface: VoxelROIInterface,
                 fc_matrix: np.ndarray, constraint_mode: ConstraintMode = "fsc",
                 ground_node=0, rtol=1e-8, maxiter: int | None = None):
        self.graph = graph
        self.interface = interface
        self.fc_matrix = np.asarray(fc_matrix, dtype=float)
        self.constraint_mode = constraint_mode
        self.ground_node = int(ground_node)
        self.rtol = float(rtol)
        self.maxiter = maxiter
        if self.fc_matrix.ndim != 2 or self.fc_matrix.shape[0] != self.fc_matrix.shape[1]:
            raise ValueError("fc_matrix must be square.")
        if self.fc_matrix.shape[0] != self.interface.roi_labels.size:
            raise ValueError("fc_matrix size must match number of ROI labels.")
        self._result: FunCTracerResult | None = None

    @classmethod
    def from_images(cls, fod_img: nib.Nifti1Image, roi_label_img: nib.Nifti1Image,
                    fc_matrix: np.ndarray, roi_labels: Iterable[int] | None = None,
                    wm_mask=None, exclusion_mask=None, fod_norm_threshold=0.25,
                    interface_conductance: float | str = "median", interface_scale=1.0,
                    normalize_total_interface_conductance=True,
                    dilation_iterations=1, constraint_mode: ConstraintMode = "fsc",
                    ground_node=0, rtol=1e-8, maxiter: int | None = None,
                    verbose=False) -> "FunCTracer":
        """Construct FunCTracer directly from FOD image, ROI labels, and FC matrix."""
        label_data = resample_labels_to_reference(roi_label_img, fod_img)
        graph = build_fod_graph(
            fod_img=fod_img, wm_mask=wm_mask, exclusion_mask=exclusion_mask,
            fod_norm_threshold=fod_norm_threshold, verbose=verbose,
        )
        interface = build_roi_supernode_interface(
            graph=graph, label_data=label_data, roi_labels=roi_labels,
            interface_conductance=interface_conductance, interface_scale=interface_scale,
            normalize_total_conductance=normalize_total_interface_conductance,
            dilation_iterations=dilation_iterations,
        )
        return cls(graph, interface, fc_matrix, constraint_mode, ground_node, rtol, maxiter)

    def solve(self) -> FunCTracerResult:
        """Solve the augmented voxel+ROI FSC system."""
        edge_i, edge_j, conductance = self.interface.augmented_edges(self.graph)
        ci, cj, values = constraints_from_fc_matrix(
            self.fc_matrix, node_ids=self.interface.node_ids_for_fc_order()
        )
        model = SparseFSC(
            n_nodes=self.interface.n_total_nodes,
            edge_i=edge_i, edge_j=edge_j, conductance=conductance,
            constraint_i=ci, constraint_j=cj, constraint_values=values,
            constraint_mode=self.constraint_mode, ground_node=self.ground_node,
            rtol=self.rtol, maxiter=self.maxiter,
        )
        sparse_result = model.solve()
        n_voxel_edges = self.graph.edge_i.size
        n_interface_edges = self.interface.edge_i.size
        self._result = FunCTracerResult(
            sparse_result=sparse_result, graph=self.graph, interface=self.interface,
            fc_matrix=self.fc_matrix, voxel_edge_slice=slice(0, n_voxel_edges),
            interface_edge_slice=slice(n_voxel_edges, n_voxel_edges + n_interface_edges),
        )
        return self._result

    def solve_pair(self, roi_index_a: int, roi_index_b: int, fc_value: float | None = None) -> FunCTracerResult:
        """Solve a pair-specific FunCTracer model using that pair's FC value."""
        a, b = int(roi_index_a), int(roi_index_b)
        if a == b:
            raise ValueError("ROI indices must differ.")
        if fc_value is None:
            fc_value = float(self.fc_matrix[a, b])
        pair_fc = np.zeros_like(self.fc_matrix, dtype=float)
        pair_fc[a, b] = pair_fc[b, a] = float(fc_value)
        return FunCTracer(self.graph, self.interface, pair_fc, self.constraint_mode,
                          self.ground_node, self.rtol, self.maxiter).solve()

    def get_result(self) -> FunCTracerResult:
        return self.solve() if self._result is None else self._result
