#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""FOD-derived voxel conductance graph for FunCTracer."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import nibabel as nib


@dataclass(frozen=True)
class FODGraph:
    """Sparse voxel graph whose conductances are derived from FOD amplitudes."""

    n_nodes: int
    node_id: np.ndarray
    work_mask: np.ndarray
    edge_i: np.ndarray
    edge_j: np.ndarray
    conductance: np.ndarray
    edge_k: np.ndarray
    offsets_vox: np.ndarray
    offset_dist: np.ndarray
    affine: np.ndarray
    fod_norm: np.ndarray
    fod_norm_threshold: float

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return tuple(self.node_id.shape)

    def voxel_edges(self):
        return self.edge_i, self.edge_j, self.conductance

    def values_to_volume(self, node_values, fill_value=np.nan, dtype=np.float32):
        values = np.asarray(node_values)
        vol = np.full(self.node_id.shape, fill_value, dtype=dtype)
        valid = self.node_id >= 0
        vol[valid] = values[self.node_id[valid]].astype(dtype)
        return vol

    def to_laplacian(self):
        from fsc.sparse import build_laplacian_from_edges
        return build_laplacian_from_edges(self.n_nodes, self.edge_i, self.edge_j, self.conductance)


def infer_lmax_from_ncoeff(ncoeff: int) -> int:
    """Infer symmetric even-order SH lmax from coefficient count."""
    for lmax in range(0, 64, 2):
        if (lmax + 1) * (lmax + 2) // 2 == int(ncoeff):
            return lmax
    raise ValueError(f"Could not infer symmetric SH lmax from {ncoeff} coefficients.")


def make_13_undirected_offsets() -> np.ndarray:
    """Return 13 unique offsets of a 26-neighbour undirected voxel graph."""
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0):
                    offsets.append((dx, dy, dz))
    return np.asarray(offsets, dtype=np.int32)


def voxel_offsets_to_world_directions(offsets_vox, affine):
    """Convert voxel offsets to physical unit directions and distances."""
    A = np.asarray(affine, dtype=float)[:3, :3]
    offsets_world = (A @ np.asarray(offsets_vox, dtype=float).T).T
    offset_dist = np.linalg.norm(offsets_world, axis=1)
    if np.any(offset_dist <= 0):
        raise ValueError("At least one voxel offset has zero physical distance.")
    return offsets_world, offset_dist, offsets_world / offset_dist[:, None]


def build_fod_kernel_matrix(ncoeff: int, offset_dirs, sphere_name="repulsion724",
                            sh_basis="tournier07", legacy=True,
                            kappa=8.0, power=2.0):
    """Build angular-kernel matrix M mapping SH coefficients to edge supports."""
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    lmax = infer_lmax_from_ncoeff(ncoeff)
    sphere = get_sphere(name=sphere_name)
    sphere_dirs = np.asarray(sphere.vertices, dtype=np.float64)
    n_sphere_dirs = sphere_dirs.shape[0]

    out = sh_to_sf_matrix(
        sphere=sphere,
        sh_order_max=lmax,
        basis_type=sh_basis,
        full_basis=False,
        legacy=legacy,
        return_inv=False,
    )
    B_raw = out[0] if isinstance(out, tuple) else out
    B_raw = np.asarray(B_raw, dtype=np.float64)

    # DIPY versions differ in matrix orientation. Standardize to sphere x coeff.
    if B_raw.shape == (n_sphere_dirs, ncoeff):
        B = B_raw
    elif B_raw.shape == (ncoeff, n_sphere_dirs):
        B = B_raw.T
    else:
        raise ValueError(f"Unexpected SH matrix shape {B_raw.shape}.")

    dots = np.abs(np.asarray(offset_dirs, dtype=float) @ sphere_dirs.T)
    W = np.exp(float(kappa) * dots ** float(power))
    W /= W.sum(axis=1, keepdims=True)
    M = W @ B
    return M, B, W, sphere_dirs


def _offset_slices(shape, offset):
    src, dst = [], []
    for n, d in zip(shape, offset):
        d = int(d)
        if d >= 0:
            src.append(slice(0, n - d)); dst.append(slice(d, n))
        else:
            src.append(slice(-d, n)); dst.append(slice(0, n + d))
    return tuple(src), tuple(dst)


def build_fod_conductance_edges(fod, work_mask, node_id, offsets_vox, offset_dist, M,
                                support_power=1.0, min_conductance=1e-12,
                                chunk_size=250_000, dtype=np.float32,
                                verbose=False):
    """Build FOD-derived conductance edges for a 26-neighbour voxel graph."""
    fod = np.asarray(fod)
    work_mask = np.asarray(work_mask, dtype=bool)
    node_id = np.asarray(node_id, dtype=np.int64)
    M = np.asarray(M, dtype=np.float64)

    if fod.ndim != 4:
        raise ValueError("fod must be a 4D SH image array.")
    if fod.shape[:3] != work_mask.shape or node_id.shape != work_mask.shape:
        raise ValueError("FOD, work_mask and node_id shapes do not match.")
    if M.shape[1] != fod.shape[3]:
        raise ValueError("M coefficient count does not match FOD coefficient count.")

    edge_i_list, edge_j_list, g_list, edge_k_list = [], [], [], []
    spatial_shape = fod.shape[:3]

    for k, offset in enumerate(np.asarray(offsets_vox, dtype=np.int32)):
        src_sl, dst_sl = _offset_slices(spatial_shape, offset)
        pair_mask = work_mask[src_sl] & work_mask[dst_sl]
        if not np.any(pair_mask):
            continue
        if verbose:
            print(f"Offset {k:02d} {tuple(offset)}: {np.count_nonzero(pair_mask):,} candidates")

        src_nodes = node_id[src_sl]
        dst_nodes = node_id[dst_sl]
        src_fod = fod[src_sl]
        dst_fod = fod[dst_sl]
        flat_idx = np.flatnonzero(pair_mask.ravel())

        for start in range(0, flat_idx.size, int(chunk_size)):
            chunk = flat_idx[start:start + int(chunk_size)]
            local_idx = np.unravel_index(chunk, pair_mask.shape)
            ids_i = src_nodes[local_idx].astype(np.int64, copy=False)
            ids_j = dst_nodes[local_idx].astype(np.int64, copy=False)
            coeff_i = src_fod[local_idx]
            coeff_j = dst_fod[local_idx]

            supp_i = np.maximum(coeff_i @ M[k, :], 0.0)
            supp_j = np.maximum(coeff_j @ M[k, :], 0.0)
            support = 0.5 * (supp_i + supp_j)
            g = (support ** float(support_power)) / float(offset_dist[k])
            keep = np.isfinite(g) & (g > min_conductance) & (ids_i >= 0) & (ids_j >= 0) & (ids_i != ids_j)
            if not np.any(keep):
                continue
            edge_i_list.append(ids_i[keep])
            edge_j_list.append(ids_j[keep])
            g_list.append(g[keep].astype(dtype, copy=False))
            edge_k_list.append(np.full(np.count_nonzero(keep), k, dtype=np.int16))

    if not edge_i_list:
        raise RuntimeError("No FOD conductance edges were created.")

    return (
        np.concatenate(edge_i_list).astype(np.int64, copy=False),
        np.concatenate(edge_j_list).astype(np.int64, copy=False),
        np.concatenate(g_list).astype(dtype, copy=False),
        np.concatenate(edge_k_list).astype(np.int16, copy=False),
    )


def build_fod_graph(fod_img: nib.Nifti1Image, wm_mask=None, exclusion_mask=None,
                    fod_norm_threshold=0.25, sphere_name="repulsion724",
                    sh_basis="tournier07", legacy=True, kappa=8.0,
                    kernel_power=2.0, support_power=1.0,
                    min_conductance=1e-12, chunk_size=250_000,
                    verbose=False) -> FODGraph:
    """Build a voxelwise FOD conductance graph from a 4D FOD SH image."""
    fod = fod_img.get_fdata(dtype=np.float32)
    if fod.ndim != 4:
        raise ValueError(f"Expected 4D FOD image, got {fod.shape}.")

    spatial_shape = fod.shape[:3]
    fod_norm = np.linalg.norm(fod, axis=3)
    if wm_mask is None:
        work_mask = fod_norm > float(fod_norm_threshold)
    else:
        work_mask = np.asarray(wm_mask, dtype=bool).copy()
        if work_mask.shape != spatial_shape:
            raise ValueError("wm_mask shape does not match FOD image.")

    if exclusion_mask is not None:
        exclusion_mask = np.asarray(exclusion_mask, dtype=bool)
        if exclusion_mask.shape != spatial_shape:
            raise ValueError("exclusion_mask shape does not match FOD image.")
        work_mask &= ~exclusion_mask

    node_id = -np.ones(spatial_shape, dtype=np.int64)
    n_nodes = int(np.count_nonzero(work_mask))
    if n_nodes == 0:
        raise ValueError("FOD graph domain is empty.")
    node_id[work_mask] = np.arange(n_nodes, dtype=np.int64)

    offsets_vox = make_13_undirected_offsets()
    _, offset_dist, offset_dirs = voxel_offsets_to_world_directions(offsets_vox, fod_img.affine)
    M, _, _, _ = build_fod_kernel_matrix(
        ncoeff=fod.shape[3], offset_dirs=offset_dirs, sphere_name=sphere_name,
        sh_basis=sh_basis, legacy=legacy, kappa=kappa, power=kernel_power,
    )
    edge_i, edge_j, conductance, edge_k = build_fod_conductance_edges(
        fod=fod, work_mask=work_mask, node_id=node_id,
        offsets_vox=offsets_vox, offset_dist=offset_dist, M=M,
        support_power=support_power, min_conductance=min_conductance,
        chunk_size=chunk_size, verbose=verbose,
    )

    return FODGraph(
        n_nodes=n_nodes, node_id=node_id, work_mask=work_mask,
        edge_i=edge_i, edge_j=edge_j, conductance=conductance,
        edge_k=edge_k, offsets_vox=offsets_vox, offset_dist=offset_dist,
        affine=np.asarray(fod_img.affine, dtype=float), fod_norm=fod_norm,
        fod_norm_threshold=float(fod_norm_threshold),
    )
