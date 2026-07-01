#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Metrics for voxelwise FSC / FunCTracer."""

from __future__ import annotations
import numpy as np


def edge_current_to_node_scalar(n_nodes: int, edge_i, edge_j, edge_current, use_absolute=True):
    """Aggregate incident edge currents to node-wise values."""
    edge_i = np.asarray(edge_i, dtype=np.int64)
    edge_j = np.asarray(edge_j, dtype=np.int64)
    values = np.asarray(edge_current, dtype=float)
    if use_absolute:
        values = np.abs(values)
    out = np.zeros(int(n_nodes), dtype=float)
    valid = np.isfinite(values)
    np.add.at(out, edge_i[valid], values[valid])
    np.add.at(out, edge_j[valid], values[valid])
    return out


def effective_conductance_matrix(n_rois: int, constraint_i, constraint_j, constraint_values,
                                 source_currents, roi_node_ids, eps=1e-15):
    """Return ROI x ROI matrix of ``|i_s| / |FC|`` for solved constraints."""
    roi_node_ids = np.asarray(roi_node_ids, dtype=np.int64)
    node_to_idx = {int(node): idx for idx, node in enumerate(roi_node_ids)}
    mat = np.full((int(n_rois), int(n_rois)), np.nan, dtype=float)
    for a, b, s, cur in zip(constraint_i, constraint_j, constraint_values, source_currents):
        if abs(float(s)) <= eps:
            continue
        ia = node_to_idx.get(int(a)); ib = node_to_idx.get(int(b))
        if ia is None or ib is None:
            continue
        mat[ia, ib] = mat[ib, ia] = abs(float(cur)) / abs(float(s))
    np.fill_diagonal(mat, 0.0)
    return mat


def current_dispersion(node_current, eps=1e-15) -> float:
    """Participation-like dispersion of a nonnegative current-density vector."""
    x = np.asarray(node_current, dtype=float)
    x = x[np.isfinite(x) & (x > eps)]
    if x.size == 0:
        return float("nan")
    p = x / np.sum(x)
    participation = 1.0 / np.sum(p ** 2)
    return float((participation - 1.0) / max(x.size - 1, 1))
