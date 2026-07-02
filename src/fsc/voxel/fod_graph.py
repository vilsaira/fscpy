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
    endpoint_combination: str = "geometric"
    length_power: float = 1.0
    orientation_gate: str = "excess"
    orientation_floor: float = 1.0

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
    """Build an antipodal angular-kernel matrix mapping SH coefficients to support.

    The kernel uses ``abs(dot(u, v))`` because diffusion FODs are axial: support
    along ``u`` and ``-u`` should be equivalent for standard even-order SH FODs.
    Bidirectionality is enforced later by evaluating both endpoint voxels and
    combining those endpoint supports with a bottleneck-like rule.
    """
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


def _combine_endpoint_supports(supp_i, supp_j, rule="geometric", eps=1e-15):
    """Combine two endpoint supports into one interface support value.

    The default is the bidirectional interface rule

        sqrt(s_i * s_j)

    so an edge receives high support only when both adjacent voxels support the
    edge direction. Arithmetic mean is retained only for sensitivity/backwards
    comparisons and should not be used as the default scientific model.
    """
    s_i = np.maximum(np.asarray(supp_i, dtype=float), 0.0)
    s_j = np.maximum(np.asarray(supp_j, dtype=float), 0.0)
    rule = str(rule).lower()

    if rule in {"geometric", "geom", "sqrt_product", "bidirectional"}:
        return np.sqrt(s_i * s_j)
    if rule in {"harmonic", "harm"}:
        return (2.0 * s_i * s_j) / (s_i + s_j + float(eps))
    if rule in {"minimum", "min"}:
        return np.minimum(s_i, s_j)
    if rule == "product":
        return s_i * s_j
    if rule in {"arithmetic", "mean", "legacy"}:
        return 0.5 * (s_i + s_j)

    raise ValueError(
        "endpoint_combination must be one of 'geometric', 'harmonic', "
        "'minimum', 'product', or 'arithmetic'."
    )


def _apply_orientation_gate(raw_support, coeff, isotropic_kernel=None,
                            orientation_gate="excess", orientation_floor=1.0,
                            eps=1e-15):
    """Suppress support that is merely isotropic rather than directionally oriented.

    ``raw_support`` is the non-negative FOD amplitude sampled along the candidate
    edge direction. With the default ``orientation_gate='excess'``, the support
    used for graph conductance is the directional excess over the voxel's
    isotropic FOD component:

        s_oriented(u) = max(F^+(u) - orientation_floor * mean(F), 0).

    Therefore a perfectly isotropic FOD gives zero support in every direction,
    even if its absolute FOD amplitude is positive. This prevents current from
    crossing voxel interfaces solely because both endpoints have flat isotropic
    FODs.

    Set ``orientation_gate='none'`` to recover the raw directional-amplitude
    behaviour, or use ``orientation_gate='contrast'`` for a normalized gate that
    preserves the raw amplitude but suppresses directions near the isotropic
    baseline.
    """
    raw = np.maximum(np.asarray(raw_support, dtype=float), 0.0)
    gate = str(orientation_gate).lower()

    if gate in {"none", "raw", "amplitude"}:
        return raw

    if isotropic_kernel is None:
        raise ValueError("isotropic_kernel is required when orientation_gate is not 'none'.")

    coeff = np.asarray(coeff)
    iso = np.maximum(coeff @ np.asarray(isotropic_kernel, dtype=float), 0.0)
    floor = float(orientation_floor) * iso

    if gate in {"excess", "directional_excess", "anisotropic_excess"}:
        return np.maximum(raw - floor, 0.0)

    if gate in {"contrast", "relative", "normalized"}:
        contrast = np.maximum(raw - floor, 0.0) / (raw + float(eps))
        return raw * contrast

    raise ValueError("orientation_gate must be 'excess', 'contrast', or 'none'.")


def build_fod_conductance_edges(fod, work_mask, node_id, offsets_vox, offset_dist, M,
                                M_reverse=None, isotropic_kernel=None,
                                endpoint_combination="geometric",
                                orientation_gate="excess", orientation_floor=1.0,
                                support_power=1.0, length_power=1.0,
                                min_conductance=1e-12, chunk_size=250_000,
                                dtype=np.float32, verbose=False):
    """Build bidirectional FOD-interface conductance edges.

    For each candidate edge i--j with unit direction u_ij, support is evaluated
    at both endpoints:

        s_i = F_i^+( u_ij)
        s_j = F_j^+(-u_ij)

    Raw directional support is converted to oriented support by subtracting the
    isotropic FOD component by default:

        s_i = max(F_i^+(u_ij) - mean(F_i), 0)
        s_j = max(F_j^+(-u_ij) - mean(F_j), 0).

    Thus flat/isotropic FODs do not conduct current across voxel interfaces. The
    default conductance is then

        c_ij = sqrt(s_i * s_j) / ell_ij.

    ``M`` maps SH coefficients to support along ``u_ij`` for each offset.
    ``M_reverse`` maps SH coefficients to support along ``-u_ij``. For standard
    antipodally symmetric diffusion FODs these matrices are usually identical,
    but keeping both names makes the interface logic explicit and allows future
    directional orientation fields. ``isotropic_kernel`` maps SH coefficients to
    the voxelwise spherical mean FOD amplitude used by the orientation gate.
    """
    fod = np.asarray(fod)
    work_mask = np.asarray(work_mask, dtype=bool)
    node_id = np.asarray(node_id, dtype=np.int64)
    M = np.asarray(M, dtype=np.float64)
    M_reverse = M if M_reverse is None else np.asarray(M_reverse, dtype=np.float64)
    isotropic_kernel = None if isotropic_kernel is None else np.asarray(isotropic_kernel, dtype=np.float64)

    if fod.ndim != 4:
        raise ValueError("fod must be a 4D SH image array.")
    if fod.shape[:3] != work_mask.shape or node_id.shape != work_mask.shape:
        raise ValueError("FOD, work_mask and node_id shapes do not match.")
    if M.shape[1] != fod.shape[3] or M_reverse.shape[1] != fod.shape[3]:
        raise ValueError("M/M_reverse coefficient count does not match FOD coefficient count.")
    if isotropic_kernel is not None and isotropic_kernel.shape != (fod.shape[3],):
        raise ValueError("isotropic_kernel must have one entry per FOD coefficient.")
    if M.shape[0] != len(offsets_vox) or M_reverse.shape[0] != len(offsets_vox):
        raise ValueError("M/M_reverse must have one row per voxel offset.")

    edge_i_list, edge_j_list, g_list, edge_k_list = [], [], [], []
    spatial_shape = fod.shape[:3]
    support_power = float(support_power)
    length_power = float(length_power)
    if length_power < 0:
        raise ValueError("length_power must be non-negative.")

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

            # Bidirectional interface support:
            # source voxel supports the edge direction +u_ij;
            # destination voxel supports the opposite interface direction -u_ij.
            raw_i = np.maximum(coeff_i @ M[k, :], 0.0)
            raw_j = np.maximum(coeff_j @ M_reverse[k, :], 0.0)
            supp_i = _apply_orientation_gate(
                raw_i, coeff_i, isotropic_kernel=isotropic_kernel,
                orientation_gate=orientation_gate, orientation_floor=orientation_floor,
            )
            supp_j = _apply_orientation_gate(
                raw_j, coeff_j, isotropic_kernel=isotropic_kernel,
                orientation_gate=orientation_gate, orientation_floor=orientation_floor,
            )
            support = _combine_endpoint_supports(
                supp_i, supp_j, rule=endpoint_combination
            )
            g = (support ** support_power) / (float(offset_dist[k]) ** length_power)
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
                    kernel_power=2.0, endpoint_combination="geometric",
                    orientation_gate="excess", orientation_floor=1.0,
                    support_power=1.0, length_power=1.0,
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
    M, B, _, _ = build_fod_kernel_matrix(
        ncoeff=fod.shape[3], offset_dirs=offset_dirs, sphere_name=sphere_name,
        sh_basis=sh_basis, legacy=legacy, kappa=kappa, power=kernel_power,
    )
    # Approximate spherical mean operator. Subtracting this from directional
    # support prevents flat/isotropic FODs from becoming conductive in every
    # neighbour direction.
    isotropic_kernel = np.asarray(B, dtype=np.float64).mean(axis=0)
    edge_i, edge_j, conductance, edge_k = build_fod_conductance_edges(
        fod=fod, work_mask=work_mask, node_id=node_id,
        offsets_vox=offsets_vox, offset_dist=offset_dist, M=M,
        M_reverse=M, isotropic_kernel=isotropic_kernel,
        endpoint_combination=endpoint_combination,
        orientation_gate=orientation_gate, orientation_floor=orientation_floor,
        support_power=support_power, length_power=length_power,
        min_conductance=min_conductance,
        chunk_size=chunk_size, verbose=verbose,
    )

    return FODGraph(
        n_nodes=n_nodes, node_id=node_id, work_mask=work_mask,
        edge_i=edge_i, edge_j=edge_j, conductance=conductance,
        edge_k=edge_k, offsets_vox=offsets_vox, offset_dist=offset_dist,
        affine=np.asarray(fod_img.affine, dtype=float), fod_norm=fod_norm,
        fod_norm_threshold=float(fod_norm_threshold),
        endpoint_combination=str(endpoint_combination), length_power=float(length_power),
        orientation_gate=str(orientation_gate), orientation_floor=float(orientation_floor),
    )
