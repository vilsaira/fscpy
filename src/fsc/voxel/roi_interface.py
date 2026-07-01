#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""ROI-supernode interface for voxelwise FSC / FunCTracer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage

from .fod_graph import FODGraph


@dataclass(frozen=True)
class VoxelROIInterface:
    """One artificial ROI supernode per functional region plus ROI-WM edges."""

    roi_labels: np.ndarray
    roi_node_ids: np.ndarray
    roi_to_boundary_nodes: dict[int, np.ndarray]
    edge_i: np.ndarray
    edge_j: np.ndarray
    conductance: np.ndarray
    n_voxel_nodes: int
    n_total_nodes: int

    def augmented_edges(self, graph: FODGraph):
        """Return voxel graph edges plus ROI-supernode interface edges."""
        return (
            np.concatenate([graph.edge_i, self.edge_i]).astype(np.int64, copy=False),
            np.concatenate([graph.edge_j, self.edge_j]).astype(np.int64, copy=False),
            np.concatenate([graph.conductance, self.conductance]).astype(float, copy=False),
        )

    def node_ids_for_fc_order(self) -> np.ndarray:
        """Return ROI supernode IDs in the same order as ``roi_labels``."""
        return self.roi_node_ids.copy()


def resample_labels_to_reference(label_img: nib.Nifti1Image, reference_img: nib.Nifti1Image) -> np.ndarray:
    """Nearest-neighbour resample label image to reference image grid."""
    out = resample_from_to(label_img, (reference_img.shape[:3], reference_img.affine), order=0)
    return np.rint(out.get_fdata(dtype=np.float32)).astype(np.int32)


def get_roi_boundary_nodes(label_data: np.ndarray, label: int, graph: FODGraph,
                           dilation_iterations: int = 1,
                           include_roi_overlap: bool = True) -> np.ndarray:
    """Return voxel graph nodes that overlap/touch a given ROI label."""
    labels = np.asarray(label_data)
    if labels.shape != graph.spatial_shape:
        raise ValueError("label_data shape does not match FOD graph shape.")
    roi = labels == int(label)
    if not np.any(roi):
        return np.array([], dtype=np.int64)
    if dilation_iterations > 0:
        near = ndimage.binary_dilation(roi, structure=np.ones((3, 3, 3), bool),
                                       iterations=int(dilation_iterations))
    else:
        near = roi
    boundary = graph.work_mask & near
    if not include_roi_overlap:
        boundary &= ~roi
    nodes = graph.node_id[boundary]
    return np.unique(nodes[nodes >= 0]).astype(np.int64)


def _interface_g(graph: FODGraph, interface_conductance: float | str, scale: float) -> float:
    if isinstance(interface_conductance, str):
        if interface_conductance == "median":
            base = float(np.median(graph.conductance))
        elif interface_conductance == "mean":
            base = float(np.mean(graph.conductance))
        elif interface_conductance == "max":
            base = float(np.max(graph.conductance))
        else:
            raise ValueError("interface_conductance must be float, 'median', 'mean', or 'max'.")
    else:
        base = float(interface_conductance)
    out = base * float(scale)
    if not np.isfinite(out) or out <= 0:
        raise ValueError("Interface conductance must be positive and finite.")
    return out


def build_roi_supernode_interface(graph: FODGraph, label_data: np.ndarray,
                                  roi_labels: Iterable[int] | None = None,
                                  interface_conductance: float | str = "median",
                                  interface_scale: float = 1.0,
                                  normalize_total_conductance: bool = True,
                                  dilation_iterations: int = 1,
                                  include_roi_overlap: bool = True) -> VoxelROIInterface:
    """Build ROI-supernodes and connect them to adjacent voxel graph nodes."""
    labels = np.asarray(label_data)
    if labels.shape != graph.spatial_shape:
        raise ValueError("label_data shape does not match graph shape.")

    if roi_labels is None:
        roi_labels_arr = np.unique(labels)
        roi_labels_arr = roi_labels_arr[roi_labels_arr > 0]
    else:
        roi_labels_arr = np.asarray(list(roi_labels), dtype=np.int64)
    if roi_labels_arr.size == 0:
        raise ValueError("No ROI labels provided/found.")

    total_g = _interface_g(graph, interface_conductance, interface_scale)
    roi_node_ids = graph.n_nodes + np.arange(roi_labels_arr.size, dtype=np.int64)

    edge_i_list, edge_j_list, g_list = [], [], []
    roi_to_boundary_nodes = {}
    for label, roi_node in zip(roi_labels_arr, roi_node_ids):
        nodes = get_roi_boundary_nodes(labels, int(label), graph, dilation_iterations,
                                       include_roi_overlap=include_roi_overlap)
        if nodes.size == 0:
            raise ValueError(f"ROI label {int(label)} has no boundary nodes in voxel graph.")
        roi_to_boundary_nodes[int(label)] = nodes
        edge_g = total_g / float(nodes.size) if normalize_total_conductance else total_g
        edge_i_list.append(np.full(nodes.size, int(roi_node), dtype=np.int64))
        edge_j_list.append(nodes.astype(np.int64, copy=False))
        g_list.append(np.full(nodes.size, edge_g, dtype=float))

    return VoxelROIInterface(
        roi_labels=roi_labels_arr.astype(np.int64),
        roi_node_ids=roi_node_ids.astype(np.int64),
        roi_to_boundary_nodes=roi_to_boundary_nodes,
        edge_i=np.concatenate(edge_i_list),
        edge_j=np.concatenate(edge_j_list),
        conductance=np.concatenate(g_list),
        n_voxel_nodes=graph.n_nodes,
        n_total_nodes=graph.n_nodes + roi_labels_arr.size,
    )
