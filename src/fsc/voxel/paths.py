#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Current-guided path extraction for FunCTracer."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class PathExtractionResult:
    paths: list[np.ndarray]
    path_mask: np.ndarray
    path_count: np.ndarray
    distance_to_roi: np.ndarray
    reached: np.ndarray
    from_nodes: np.ndarray
    to_nodes: np.ndarray


def supernode_flux_endpoints(edge_i, edge_j, edge_current, roi_node: int,
                             n_voxel_nodes: int, outgoing: bool):
    """Return voxel boundary nodes carrying current out of/into a ROI supernode."""
    ei = np.asarray(edge_i, dtype=np.int64)
    ej = np.asarray(edge_j, dtype=np.int64)
    cur = np.asarray(edge_current, dtype=float)
    roi_node = int(roi_node)
    flux = np.zeros(int(n_voxel_nodes), dtype=float)
    finite = np.isfinite(cur)

    case_i = finite & (ei == roi_node) & (ej < n_voxel_nodes)
    case_j = finite & (ej == roi_node) & (ei < n_voxel_nodes)
    if outgoing:
        vals = cur[case_i]; nodes = ej[case_i]; keep = vals > 0
        np.add.at(flux, nodes[keep], vals[keep])
        vals = -cur[case_j]; nodes = ei[case_j]; keep = vals > 0
        np.add.at(flux, nodes[keep], vals[keep])
    else:
        vals = -cur[case_i]; nodes = ej[case_i]; keep = vals > 0
        np.add.at(flux, nodes[keep], vals[keep])
        vals = cur[case_j]; nodes = ei[case_j]; keep = vals > 0
        np.add.at(flux, nodes[keep], vals[keep])

    nodes = np.flatnonzero(flux > 0).astype(np.int64)
    return nodes, flux[nodes]


def select_flux_nodes(nodes, flux, cumulative_fraction=1.0, min_relative_flux=0.0, min_nodes=1):
    """Select boundary nodes by flux contribution."""
    nodes = np.asarray(nodes, dtype=np.int64)
    flux = np.asarray(flux, dtype=float)
    valid = np.isfinite(flux) & (flux > 0)
    nodes, flux = nodes[valid], flux[valid]
    if nodes.size == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(flux)[::-1]
    nodes, flux = nodes[order], flux[order]
    total = float(np.sum(flux))
    cum = np.cumsum(flux) / total
    keep = (cum <= float(cumulative_fraction)) & (flux >= float(min_relative_flux) * float(flux[0]))
    if cumulative_fraction >= 1.0:
        keep[:] = True
    else:
        keep[: int(np.argmax(cum >= cumulative_fraction)) + 1] = True
    if np.count_nonzero(keep) < int(min_nodes):
        keep[: int(min_nodes)] = True
    return nodes[keep].astype(np.int64)


def build_local_current_fraction_route_graph(n_nodes: int, edge_i, edge_j, edge_k,
                                             edge_current, offset_dist, phi,
                                             length_weight=0.05, eps=1e-12):
    """Build directed route graph with costs ``-log(local current fraction)``."""
    ei = np.asarray(edge_i, dtype=np.int64)
    ej = np.asarray(edge_j, dtype=np.int64)
    ek = np.asarray(edge_k, dtype=np.int64)
    cur_abs = np.abs(np.asarray(edge_current, dtype=float))
    phi = np.asarray(phi, dtype=float)
    valid = np.isfinite(cur_abs) & (cur_abs > 0) & np.isfinite(phi[ei]) & np.isfinite(phi[ej]) & (phi[ei] != phi[ej])
    if not np.any(valid):
        raise RuntimeError("No valid current-carrying voxel edges for routing.")
    ei, ej, ek, cur = ei[valid], ej[valid], ek[valid], cur_abs[valid]
    i_to_j = phi[ei] > phi[ej]
    u = np.where(i_to_j, ei, ej)
    v = np.where(i_to_j, ej, ei)
    out_sum = np.zeros(int(n_nodes), dtype=float)
    np.add.at(out_sum, u, cur)
    good = out_sum[u] > 0
    u, v, ek, cur = u[good], v[good], ek[good], cur[good]
    p = np.clip(cur / out_sum[u], float(eps), 1.0)
    cost = -np.log(p) + float(length_weight) * np.asarray(offset_dist, dtype=float)[ek]
    G = sparse.coo_matrix((cost, (u, v)), shape=(int(n_nodes), int(n_nodes))).tocsr()
    G.sum_duplicates(); G.eliminate_zeros()
    return G


def fast_endpoint_paths_to_roi(G_forward, from_nodes, to_nodes) -> PathExtractionResult:
    """One Dijkstra solve for shortest paths from many nodes to any ROI node."""
    G_forward = G_forward.tocsr()
    n_nodes = G_forward.shape[0]
    from_nodes = np.unique(np.asarray(from_nodes, dtype=np.int64))
    to_nodes = np.unique(np.asarray(to_nodes, dtype=np.int64))
    from_nodes = from_nodes[(from_nodes >= 0) & (from_nodes < n_nodes)]
    to_nodes = to_nodes[(to_nodes >= 0) & (to_nodes < n_nodes)]
    if from_nodes.size == 0 or to_nodes.size == 0:
        raise ValueError("from_nodes and to_nodes must be non-empty.")

    G_rev = G_forward.T.tocoo()
    virtual = n_nodes
    rows = np.concatenate([G_rev.row, np.full(to_nodes.size, virtual, dtype=np.int64)])
    cols = np.concatenate([G_rev.col, to_nodes])
    data = np.concatenate([G_rev.data, np.zeros(to_nodes.size, dtype=float)])
    G_aug = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes + 1, n_nodes + 1)).tocsr()
    dist, pred = dijkstra(G_aug, directed=True, indices=virtual, return_predecessors=True)

    path_mask = np.zeros(n_nodes, dtype=bool)
    path_count = np.zeros(n_nodes, dtype=np.float32)
    paths = []
    reached = np.zeros(from_nodes.size, dtype=bool)
    for idx, seed in enumerate(from_nodes):
        if not np.isfinite(dist[seed]):
            continue
        nodes = []
        node = int(seed)
        while node != virtual and node != -9999:
            if node < n_nodes:
                nodes.append(node)
            node = int(pred[node])
        if node != virtual:
            continue
        path = np.asarray(nodes, dtype=np.int64)
        if path.size < 2:
            continue
        paths.append(path)
        path_mask[path] = True
        path_count[path] += 1.0
        reached[idx] = True
    return PathExtractionResult(paths, path_mask, path_count, dist[:n_nodes], reached, from_nodes, to_nodes)


def extract_pair_paths(n_voxel_nodes: int, edge_i, edge_j, edge_k, edge_current,
                       offset_dist, phi, roi_node_a: int, roi_node_b: int,
                       all_edge_i, all_edge_j, all_edge_current,
                       length_weight=0.05, cumulative_fraction=1.0,
                       min_relative_flux=0.0, min_nodes=1):
    """Extract current-guided voxel paths for one ROI-supernode pair."""
    phi = np.asarray(phi, dtype=float)
    start_roi, end_roi = (int(roi_node_a), int(roi_node_b)) if phi[int(roi_node_a)] >= phi[int(roi_node_b)] else (int(roi_node_b), int(roi_node_a))
    start_nodes, start_flux = supernode_flux_endpoints(all_edge_i, all_edge_j, all_edge_current, start_roi, n_voxel_nodes, outgoing=True)
    end_nodes, end_flux = supernode_flux_endpoints(all_edge_i, all_edge_j, all_edge_current, end_roi, n_voxel_nodes, outgoing=False)
    start_nodes = select_flux_nodes(start_nodes, start_flux, cumulative_fraction, min_relative_flux, min_nodes)
    end_nodes = select_flux_nodes(end_nodes, end_flux, cumulative_fraction, min_relative_flux, min_nodes)
    if start_nodes.size == 0 or end_nodes.size == 0:
        raise RuntimeError("No supernode boundary flux endpoints found for pair.")
    G = build_local_current_fraction_route_graph(n_voxel_nodes, edge_i, edge_j, edge_k,
                                                 edge_current, offset_dist, phi[:n_voxel_nodes],
                                                 length_weight=length_weight)
    forward = fast_endpoint_paths_to_roi(G, start_nodes, end_nodes)
    backward = fast_endpoint_paths_to_roi(G.T.tocsr(), end_nodes, start_nodes)
    return {
        "start_roi": start_roi,
        "end_roi": end_roi,
        "start_nodes": start_nodes,
        "end_nodes": end_nodes,
        "forward": forward,
        "backward": backward,
        "path_mask": forward.path_mask | backward.path_mask,
        "path_count": forward.path_count + backward.path_count,
    }
