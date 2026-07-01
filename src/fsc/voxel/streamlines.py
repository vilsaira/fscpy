#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Streamline-like export utilities for FunCTracer paths.

This module is for visualization/export of FunCTracer graph paths. Quantitative
path maps should always be computed from the raw graph paths, not from the
smoothed or dense-display streamlines.

Main display modes
------------------
1. Raw voxel-center polylines
2. Safe B-spline display smoothing
3. Chaikin fallback smoothing if B-spline is unsafe
4. Dense display replicas that preserve all centerlines first and add only
   visualization-only subvoxel-offset copies
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from nibabel.streamlines import Tractogram, TckFile
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# Basic node/path conversion
# -----------------------------------------------------------------------------

def build_node_to_ijk(node_id: np.ndarray, n_nodes: int) -> np.ndarray:
    """Build inverse lookup: voxel graph node index -> voxel ijk."""
    node_id = np.asarray(node_id)
    node_to_ijk = np.full((int(n_nodes), 3), -1, dtype=np.int32)

    coords = np.argwhere(node_id >= 0)
    values = node_id[node_id >= 0].astype(np.int64)

    node_to_ijk[values] = coords

    if np.any(node_to_ijk < 0):
        missing = int(np.count_nonzero(np.any(node_to_ijk < 0, axis=1)))
        raise RuntimeError(f"Some graph nodes were not mapped to voxel coordinates: {missing}")

    return node_to_ijk


def save_tck(streamlines: list[np.ndarray], path: str) -> None:
    """Save streamlines to MRtrix .tck format."""
    clean = [np.asarray(sl, dtype=np.float32) for sl in streamlines if np.asarray(sl).shape[0] >= 2]
    tractogram = Tractogram(clean, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(TckFile(tractogram), path)


# -----------------------------------------------------------------------------
# Generic geometry helpers
# -----------------------------------------------------------------------------

def _remove_consecutive_duplicates(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Remove consecutive duplicate or near-duplicate points."""
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] <= 1:
        return points

    keep = [0]
    for k in range(1, points.shape[0]):
        if np.linalg.norm(points[k] - points[keep[-1]]) > eps:
            keep.append(k)

    return points[keep]


def _streamline_length(points: np.ndarray) -> float:
    """Total arclength of a streamline."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _sample_polyline(points: np.ndarray, step_mm: float = 0.25) -> np.ndarray:
    """Densely sample a polyline at approximately step_mm intervals."""
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] < 2:
        return points

    step_mm = max(float(step_mm), 1e-6)
    sampled = [points[0]]

    for p0, p1 in zip(points[:-1], points[1:]):
        d = float(np.linalg.norm(p1 - p0))
        if d <= 0:
            continue
        n = max(1, int(np.ceil(d / step_mm)))
        for k in range(1, n + 1):
            sampled.append(p0 + (k / n) * (p1 - p0))

    return np.asarray(sampled, dtype=np.float64)


def _max_nearest_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Maximum nearest-neighbour distance from points_a to points_b."""
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)

    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return np.inf

    tree = cKDTree(points_b)
    d, _ = tree.query(points_a, k=1)
    return float(np.max(d))


def _unit_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    """Return unit vector or None if norm is too small."""
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return None
    return v / n


# -----------------------------------------------------------------------------
# Mask helpers for display validation
# -----------------------------------------------------------------------------

def make_display_mask(valid_mask: np.ndarray, dilation_iterations: int = 1) -> np.ndarray:
    """Create a slightly dilated display-only validation mask.

    The exact voxel graph mask is often too strict for subvoxel-offset display
    replicas: a harmless 0.3-0.5 mm offset can round into a neighbouring
    non-graph voxel. Dilation here is only for display QA; it is not used for
    graph construction, current solving, or quantitative path maps.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if int(dilation_iterations) <= 0:
        return valid_mask

    return ndimage.binary_dilation(
        valid_mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=int(dilation_iterations),
    )


def _fraction_points_inside_mask(
    points_xyz: np.ndarray,
    affine: np.ndarray,
    valid_mask: np.ndarray,
    sample_step_mm: float = 0.25,
) -> float:
    """Fraction of densely sampled streamline points inside valid_mask."""
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if points_xyz.shape[0] < 2:
        return 0.0

    sampled = _sample_polyline(points_xyz, step_mm=sample_step_mm)

    inv_affine = np.linalg.inv(affine)
    ijk = apply_affine(inv_affine, sampled)
    ijk = np.rint(ijk).astype(np.int64)

    inside_bounds = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < valid_mask.shape[0]) &
        (ijk[:, 1] >= 0) & (ijk[:, 1] < valid_mask.shape[1]) &
        (ijk[:, 2] >= 0) & (ijk[:, 2] < valid_mask.shape[2])
    )

    ok = np.zeros(ijk.shape[0], dtype=bool)
    if np.any(inside_bounds):
        ii = ijk[inside_bounds]
        ok[inside_bounds] = valid_mask[ii[:, 0], ii[:, 1], ii[:, 2]]

    return float(np.mean(ok))


def _points_inside_mask_all(
    points_xyz: np.ndarray,
    affine: np.ndarray,
    valid_mask: np.ndarray,
    sample_step_mm: float = 0.25,
) -> bool:
    """Strict check: all densely sampled points must be inside valid_mask."""
    return _fraction_points_inside_mask(
        points_xyz,
        affine=affine,
        valid_mask=valid_mask,
        sample_step_mm=sample_step_mm,
    ) >= 1.0


# -----------------------------------------------------------------------------
# Display smoothing
# -----------------------------------------------------------------------------

def _resample_control_points(points: np.ndarray, control_spacing: float = 3.0) -> np.ndarray:
    """Reduce a dense voxel-step path to sparse arclength control points."""
    points = _remove_consecutive_duplicates(points)

    if points.shape[0] <= 2:
        return points

    control_spacing = max(float(control_spacing), 1e-6)

    seg_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1])

    if total <= 0:
        return points

    n_ctrl = max(3, int(np.ceil(total / control_spacing)) + 1)
    s_new = np.linspace(0.0, total, n_ctrl)

    ctrl = np.zeros((n_ctrl, 3), dtype=np.float64)
    for dim in range(3):
        ctrl[:, dim] = np.interp(s_new, s, points[:, dim])

    ctrl[0] = points[0]
    ctrl[-1] = points[-1]
    return ctrl


def smooth_streamline_control_spline(
    points: np.ndarray,
    control_spacing: float = 3.0,
    output_spacing: float = 0.4,
    smoothing: float = 0.5,
    spline_order: int = 3,
    keep_endpoints: bool = True,
) -> np.ndarray:
    """Smooth one streamline for visualization using an arclength B-spline.

    This is display-only. It should not be used for quantitative path counts,
    conductance, resistance, or current maps.
    """
    points = _remove_consecutive_duplicates(points)

    if points.shape[0] < 3:
        return points.astype(np.float32)

    total_len = _streamline_length(points)
    if total_len <= 0:
        return points.astype(np.float32)

    ctrl = _resample_control_points(points, control_spacing=control_spacing)
    if ctrl.shape[0] < 3:
        return points.astype(np.float32)

    ctrl_seg = np.linalg.norm(np.diff(ctrl, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(ctrl_seg)])
    if u[-1] <= 0:
        return points.astype(np.float32)

    u = u / u[-1]

    keep = np.concatenate([[True], np.diff(u) > 1e-8])
    ctrl = ctrl[keep]
    u = u[keep]
    if ctrl.shape[0] < 3:
        return points.astype(np.float32)

    k = min(int(spline_order), ctrl.shape[0] - 1)
    output_spacing = max(float(output_spacing), 1e-6)

    try:
        tck, _ = splprep(
            [ctrl[:, 0], ctrl[:, 1], ctrl[:, 2]],
            u=u,
            k=k,
            s=float(smoothing),
        )

        n_out = max(2, int(np.ceil(total_len / output_spacing)) + 1)
        u_new = np.linspace(0.0, 1.0, n_out)

        x, y, z = splev(u_new, tck)
        out = np.vstack([x, y, z]).T.astype(np.float32)

        if keep_endpoints:
            out[0] = points[0].astype(np.float32)
            out[-1] = points[-1].astype(np.float32)

        return out

    except Exception:
        return points.astype(np.float32)


def smooth_streamline_chaikin(
    points: np.ndarray,
    n_iter: int = 2,
    keep_endpoints: bool = True,
) -> np.ndarray:
    """Conservative corner-cutting smoother for display fallback.

    Chaikin smoothing is much less likely to overshoot than a B-spline. It is
    useful when safe_control_spline fails but raw voxel paths look too jagged.
    """
    points = _remove_consecutive_duplicates(points)

    if points.shape[0] < 3:
        return points.astype(np.float32)

    out = points.astype(np.float64)

    for _ in range(max(0, int(n_iter))):
        new_points = []

        if keep_endpoints:
            new_points.append(out[0])

        for p0, p1 in zip(out[:-1], out[1:]):
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_points.extend([q, r])

        if keep_endpoints:
            new_points.append(out[-1])

        out = np.asarray(new_points, dtype=np.float64)

    return out.astype(np.float32)


def smooth_streamlines_control_spline(
    streamlines: list[np.ndarray],
    control_spacing: float = 3.0,
    output_spacing: float = 0.4,
    smoothing: float = 0.5,
    spline_order: int = 3,
) -> list[np.ndarray]:
    """Smooth multiple streamlines for display/export."""
    out = []
    for sl in streamlines:
        sl = np.asarray(sl, dtype=np.float32)
        if sl.shape[0] < 2:
            continue
        smoothed = smooth_streamline_control_spline(
            sl,
            control_spacing=control_spacing,
            output_spacing=output_spacing,
            smoothing=smoothing,
            spline_order=spline_order,
            keep_endpoints=True,
        )
        if smoothed.shape[0] >= 2:
            out.append(smoothed.astype(np.float32))
    return out


def _validate_display_streamline(
    raw_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    affine: np.ndarray,
    display_mask: np.ndarray | None,
    max_deviation_mm: float = 2.0,
    mask_sample_step_mm: float = 0.50,
    min_inside_fraction: float = 0.98,
    min_points: int = 2,
) -> bool:
    """Validate a display-smoothed candidate against raw path and mask."""
    raw_xyz = np.asarray(raw_xyz, dtype=np.float64)
    candidate_xyz = np.asarray(candidate_xyz, dtype=np.float64)

    if candidate_xyz.shape[0] < int(min_points):
        return False

    if not np.all(np.isfinite(candidate_xyz)):
        return False

    if _max_nearest_distance(candidate_xyz, raw_xyz) > float(max_deviation_mm):
        return False

    if display_mask is not None:
        inside_fraction = _fraction_points_inside_mask(
            candidate_xyz,
            affine=affine,
            valid_mask=display_mask,
            sample_step_mm=mask_sample_step_mm,
        )
        if inside_fraction < float(min_inside_fraction):
            return False

    return True


def paths_to_streamlines(
    paths: list[np.ndarray],
    node_to_ijk: np.ndarray,
    affine: np.ndarray,
    reverse: bool = False,
    min_points: int = 2,
    smoothing: str | None = None,
    control_spacing: float = 3.0,
    output_spacing: float = 0.4,
    spline_smoothing: float = 0.5,
    spline_order: int = 3,
    valid_mask: np.ndarray | None = None,
    max_smooth_deviation_mm: float = 2.0,
    mask_sample_step_mm: float = 0.50,
    mask_dilation_iterations: int = 1,
    min_inside_fraction: float = 0.98,
    fallback: str = "raw",
    chaikin_iterations: int = 2,
    verbose: bool = True,
) -> list[np.ndarray]:
    """Convert node-index paths to world-coordinate streamline-like arrays.

    Integer voxel indices are interpreted as voxel centers under the NIfTI
    affine. Do not add +0.5 here.

    Smoothing modes
    ---------------
    None / 'none':
        Raw voxel-center polylines.

    'control_spline' / 'spline' / 'display':
        B-spline display smoothing without safety validation. Mostly retained
        for backwards compatibility.

    'safe_control_spline':
        B-spline display smoothing with validation against a display mask and
        raw graph path. If the B-spline is unsafe, fallback can be:
            'raw'          -> raw voxel polyline
            'chaikin'      -> conservative corner-cutting smoother, then raw if unsafe
            'chaikin_drop' -> conservative corner-cutting smoother, then drop if unsafe
            'drop'         -> omit unsafe streamline
    """
    out: list[np.ndarray] = []
    n_voxel_nodes = int(node_to_ijk.shape[0])

    display_mask = None
    if valid_mask is not None:
        display_mask = make_display_mask(
            valid_mask,
            dilation_iterations=mask_dilation_iterations,
        )

    mode = None if smoothing is None else str(smoothing).lower()

    n_attempted = 0
    n_spline_kept = 0
    n_chaikin_kept = 0
    n_raw_fallback = 0
    n_dropped = 0

    for p in paths:
        p = np.asarray(p, dtype=np.int64)

        if reverse:
            p = p[::-1]

        # Keep only voxel graph nodes, not possible ROI supernodes.
        p = p[(p >= 0) & (p < n_voxel_nodes)]

        if p.size < int(min_points):
            continue

        ijk = node_to_ijk[p].astype(np.float64)

        # Correct NIfTI convention: integer ijk -> voxel centers.
        raw_xyz = apply_affine(affine, ijk).astype(np.float32)

        if mode is None or mode in {"none", "false"}:
            out.append(raw_xyz)
            continue

        if mode not in {"control_spline", "spline", "display", "safe_control_spline"}:
            raise ValueError(
                f"Unknown smoothing mode: {smoothing!r}. "
                "Use None, 'none', 'control_spline', or 'safe_control_spline'."
            )

        n_attempted += 1

        smooth_xyz = smooth_streamline_control_spline(
            raw_xyz,
            control_spacing=control_spacing,
            output_spacing=output_spacing,
            smoothing=spline_smoothing,
            spline_order=spline_order,
            keep_endpoints=True,
        ).astype(np.float32)

        # Non-safe mode: keep spline if it has enough points.
        if mode != "safe_control_spline":
            if smooth_xyz.shape[0] >= int(min_points):
                out.append(smooth_xyz)
                n_spline_kept += 1
            else:
                out.append(raw_xyz)
                n_raw_fallback += 1
            continue

        spline_safe = _validate_display_streamline(
            raw_xyz=raw_xyz,
            candidate_xyz=smooth_xyz,
            affine=affine,
            display_mask=display_mask,
            max_deviation_mm=max_smooth_deviation_mm,
            mask_sample_step_mm=mask_sample_step_mm,
            min_inside_fraction=min_inside_fraction,
            min_points=min_points,
        )

        if spline_safe:
            out.append(smooth_xyz.astype(np.float32))
            n_spline_kept += 1
            continue

        # Safe spline failed. Choose fallback.
        fallback_mode = str(fallback).lower()

        if fallback_mode in {"chaikin", "chaikin_drop"}:
            chaikin_xyz = smooth_streamline_chaikin(
                raw_xyz,
                n_iter=chaikin_iterations,
                keep_endpoints=True,
            ).astype(np.float32)

            chaikin_safe = _validate_display_streamline(
                raw_xyz=raw_xyz,
                candidate_xyz=chaikin_xyz,
                affine=affine,
                display_mask=display_mask,
                max_deviation_mm=max_smooth_deviation_mm,
                mask_sample_step_mm=mask_sample_step_mm,
                min_inside_fraction=min_inside_fraction,
                min_points=min_points,
            )

            if chaikin_safe:
                out.append(chaikin_xyz)
                n_chaikin_kept += 1
            elif fallback_mode == "chaikin":
                # Backwards-compatible behaviour: preserve coverage by using raw.
                out.append(raw_xyz)
                n_raw_fallback += 1
            else:
                # Pure display behaviour: never write raw jagged paths to DISPLAY.tck.
                n_dropped += 1
            continue

        if fallback_mode == "raw":
            out.append(raw_xyz)
            n_raw_fallback += 1
            continue

        if fallback_mode == "drop":
            n_dropped += 1
            continue

        raise ValueError("fallback must be 'raw', 'chaikin', 'chaikin_drop', or 'drop'.")

    if verbose and mode is not None:
        print(
            "Display smoothing:",
            f"attempted={n_attempted},",
            f"spline_kept={n_spline_kept},",
            f"chaikin_kept={n_chaikin_kept},",
            f"raw_fallback={n_raw_fallback},",
            f"dropped={n_dropped}",
        )

    return out


# -----------------------------------------------------------------------------
# Dense display tractogram helpers
# -----------------------------------------------------------------------------

def _local_tangent(points: np.ndarray, idx: int) -> np.ndarray:
    """Estimate local streamline tangent at point idx."""
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if idx == 0:
        t = points[1] - points[0]
    elif idx == points.shape[0] - 1:
        t = points[-1] - points[-2]
    else:
        t = points[idx + 1] - points[idx - 1]

    u = _unit_vector(t)
    if u is None:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    return u


def _normal_frame_from_tangent(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an arbitrary but stable normal-plane basis around a tangent."""
    t = _unit_vector(tangent)
    if t is None:
        t = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    refs = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ]
    ref = min(refs, key=lambda r: abs(float(np.dot(t, r))))

    n1 = _unit_vector(np.cross(t, ref))
    if n1 is None:
        n1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    n2 = _unit_vector(np.cross(t, n1))
    if n2 is None:
        n2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return n1, n2


def make_subvoxel_display_replicas(
    streamlines: list[np.ndarray],
    ref_img: nib.Nifti1Image,
    valid_mask: np.ndarray,
    replicas_per_streamline: int = 12,
    radius_mm: float = 0.45,
    max_streamlines: int | None = None,
    seed: int = 12345,
    sample_step_mm: float = 0.50,
    max_deviation_mm: float = 1.5,
    mask_dilation_iterations: int = 1,
    min_inside_fraction: float = 0.95,
    include_originals: bool = False,
    verbose: bool = True,
) -> list[np.ndarray]:
    """Create subvoxel-offset display replicas around centerlines.

    Visualization only. Replicas are accepted only if they remain inside a
    display mask and near the parent centerline.

    Parameters
    ----------
    include_originals:
        If True, output includes the input centerlines. For high-level dense
        display export, keep this False because make_display_dense_streamlines()
        already preserves all centerlines before adding replicas.
    """
    rng = np.random.default_rng(seed)

    streamlines = [
        np.asarray(sl, dtype=np.float32)
        for sl in streamlines
        if np.asarray(sl).shape[0] >= 2
    ]

    if valid_mask is None:
        display_mask = None
    else:
        display_mask = make_display_mask(
            valid_mask,
            dilation_iterations=mask_dilation_iterations,
        )

    out: list[np.ndarray] = []
    n_attempted = 0
    n_kept = 0
    n_reject_mask = 0
    n_reject_distance = 0

    if include_originals:
        for sl in streamlines:
            out.append(sl.copy())
            if max_streamlines is not None and len(out) >= int(max_streamlines):
                return out

    max_streamlines_int = None if max_streamlines is None else int(max_streamlines)

    for sl in streamlines:
        if max_streamlines_int is not None and len(out) >= max_streamlines_int:
            break

        n_points = sl.shape[0]
        if n_points < 2:
            continue

        u = np.linspace(0.0, 1.0, n_points)
        endpoint_taper = np.sin(np.pi * u)

        for _ in range(int(replicas_per_streamline)):
            if max_streamlines_int is not None and len(out) >= max_streamlines_int:
                break

            n_attempted += 1

            theta = rng.uniform(0.0, 2.0 * np.pi)
            r = float(radius_mm) * np.sqrt(rng.uniform(0.0, 1.0))

            phase = rng.uniform(0.0, 2.0 * np.pi)
            wobble = 0.65 + 0.35 * np.sin(2.0 * np.pi * u + phase)

            replica = np.zeros_like(sl, dtype=np.float64)

            for idx in range(n_points):
                tangent = _local_tangent(sl, idx)
                n1, n2 = _normal_frame_from_tangent(tangent)
                offset_dir = np.cos(theta) * n1 + np.sin(theta) * n2
                offset = r * endpoint_taper[idx] * wobble[idx] * offset_dir
                replica[idx] = sl[idx].astype(np.float64) + offset

            # Preserve exact endpoints so replicas remain anchored to the same
            # graph-derived boundary points.
            replica[0] = sl[0]
            replica[-1] = sl[-1]
            replica = replica.astype(np.float32)

            near = _max_nearest_distance(replica, sl) <= float(max_deviation_mm)
            if not near:
                n_reject_distance += 1
                continue

            if display_mask is not None:
                inside_fraction = _fraction_points_inside_mask(
                    replica,
                    affine=ref_img.affine,
                    valid_mask=display_mask,
                    sample_step_mm=sample_step_mm,
                )
                if inside_fraction < float(min_inside_fraction):
                    n_reject_mask += 1
                    continue

            out.append(replica.astype(np.float32))
            n_kept += 1

    if verbose:
        print("\nSubvoxel display replicas")
        print("-------------------------")
        print("input centerlines:", len(streamlines))
        print("attempted replicas:", n_attempted)
        print("kept replicas:", n_kept)
        print("rejected by mask:", n_reject_mask)
        print("rejected by distance:", n_reject_distance)
        print("returned streamlines:", len(out))
        print("include originals:", include_originals)

    return out


def make_display_dense_streamlines(
    centerlines: list[np.ndarray],
    ref_img: nib.Nifti1Image,
    valid_mask: np.ndarray,
    replicas_per_selected_streamline: int = 6,
    radius_mm: float = 0.30,
    max_extra_replicas: int = 3000,
    seed: int = 12345,
    sample_step_mm: float = 0.50,
    max_deviation_mm: float = 1.25,
    mask_dilation_iterations: int = 1,
    min_inside_fraction: float = 0.95,
    verbose: bool = True,
) -> list[np.ndarray]:
    """Create a dense visualization tractogram from display centerlines.

    Crucial behavior:
    - all input centerlines are always preserved;
    - the replica budget applies only to additional offset replicas;
    - centerlines selected for replication are sampled across the full list,
      not only from the beginning.
    """
    centerlines = [
        np.asarray(sl, dtype=np.float32)
        for sl in centerlines
        if np.asarray(sl).shape[0] >= 2
    ]

    # Always preserve the complete display tractogram first.
    dense: list[np.ndarray] = [sl.copy() for sl in centerlines]

    if len(centerlines) == 0:
        return dense

    replicas_per_selected_streamline = max(1, int(replicas_per_selected_streamline))
    max_extra_replicas = max(0, int(max_extra_replicas))

    if max_extra_replicas == 0:
        return dense

    n_selected = max_extra_replicas // replicas_per_selected_streamline
    n_selected = max(1, min(n_selected, len(centerlines)))

    # Evenly sample across the full tractogram. This avoids the old failure
    # mode where the budget was exhausted by the first few centerlines and the
    # later parts of the tract visually disappeared.
    selected_idx = np.linspace(0, len(centerlines) - 1, n_selected).round().astype(int)
    selected_idx = np.unique(selected_idx)
    selected_centerlines = [centerlines[int(idx)] for idx in selected_idx]

    replicas = make_subvoxel_display_replicas(
        streamlines=selected_centerlines,
        ref_img=ref_img,
        valid_mask=valid_mask,
        replicas_per_streamline=replicas_per_selected_streamline,
        radius_mm=radius_mm,
        max_streamlines=max_extra_replicas,
        seed=seed,
        sample_step_mm=sample_step_mm,
        max_deviation_mm=max_deviation_mm,
        mask_dilation_iterations=mask_dilation_iterations,
        min_inside_fraction=min_inside_fraction,
        include_originals=False,
        verbose=verbose,
    )

    # No endpoint/length duplicate filtering here: display replicas intentionally
    # preserve endpoints, so such filtering would delete valid replicas.
    if len(replicas) > max_extra_replicas:
        replicas = replicas[:max_extra_replicas]

    dense.extend(replicas)

    if verbose:
        print("\nDense display streamlines")
        print("-------------------------")
        print("input centerlines:", len(centerlines))
        print("selected for replication:", len(selected_centerlines))
        print("added replicas:", len(replicas))
        print("output dense streamlines:", len(dense))

    return dense


# -----------------------------------------------------------------------------
# Optional current-bent visualization helpers
# -----------------------------------------------------------------------------

def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=1)
    out = np.zeros_like(v)
    keep = n > eps
    out[keep] = v[keep] / n[keep, None]
    return out


def build_node_current_vectors(
    n_nodes: int,
    edge_i,
    edge_j,
    edge_current,
    node_xyz,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build node-wise current direction vectors from edge currents."""
    edge_i = np.asarray(edge_i, dtype=np.int64)
    edge_j = np.asarray(edge_j, dtype=np.int64)
    current = np.asarray(edge_current, dtype=float)
    node_xyz = np.asarray(node_xyz, dtype=float)

    vec = np.zeros((int(n_nodes), 3), dtype=float)
    weight_sum = np.zeros(int(n_nodes), dtype=float)

    d = node_xyz[edge_j] - node_xyz[edge_i]
    dist = np.linalg.norm(d, axis=1)
    valid = np.isfinite(current) & (np.abs(current) > 0) & (dist > eps)

    unit = np.zeros_like(d)
    unit[valid] = d[valid] / dist[valid, None]

    signed = unit * np.sign(current)[:, None]
    w = np.abs(current)

    np.add.at(vec, edge_i[valid], signed[valid] * w[valid, None])
    np.add.at(vec, edge_j[valid], signed[valid] * w[valid, None])
    np.add.at(weight_sum, edge_i[valid], w[valid])
    np.add.at(weight_sum, edge_j[valid], w[valid])

    keep = weight_sum > eps
    vec[keep] /= weight_sum[keep, None]

    return _normalize_rows(vec).astype(np.float32)


def path_nodes_to_current_bent_curve(
    path_nodes,
    node_xyz,
    node_current_vec,
    current_blend: float = 0.35,
    tangent_scale: float = 0.50,
    points_per_segment: int = 5,
    min_points: int = 2,
) -> np.ndarray | None:
    """Convert a graph path into a piecewise cubic curve bent by current vectors."""
    path_nodes = np.asarray(path_nodes, dtype=np.int64)
    if path_nodes.size < int(min_points):
        return None

    P = np.asarray(node_xyz[path_nodes], dtype=float)
    if P.shape[0] == 2:
        return P.astype(np.float32)

    T_path = np.zeros_like(P)
    T_path[0] = P[1] - P[0]
    T_path[-1] = P[-1] - P[-2]
    T_path[1:-1] = P[2:] - P[:-2]
    T_path = _normalize_rows(T_path)

    T_cur = _normalize_rows(np.asarray(node_current_vec[path_nodes], dtype=float))
    T_cur[np.sum(T_path * T_cur, axis=1) < 0] *= -1.0

    T = _normalize_rows((1.0 - float(current_blend)) * T_path + float(current_blend) * T_cur)

    seg_len = np.linalg.norm(P[1:] - P[:-1], axis=1)
    tangent_len = np.zeros(P.shape[0], dtype=float)
    tangent_len[0] = seg_len[0]
    tangent_len[-1] = seg_len[-1]
    tangent_len[1:-1] = 0.5 * (seg_len[:-1] + seg_len[1:])
    T *= float(tangent_scale) * tangent_len[:, None]

    samples = []
    points_per_segment = max(1, int(points_per_segment))

    for k in range(P.shape[0] - 1):
        P0, P1 = P[k], P[k + 1]
        M0, M1 = T[k], T[k + 1]

        for t in np.linspace(0, 1, points_per_segment, endpoint=False):
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            samples.append(h00 * P0 + h10 * M0 + h01 * P1 + h11 * M1)

    samples.append(P[-1])
    return np.asarray(samples, dtype=np.float32)


def paths_to_current_bent_streamlines(
    paths,
    node_xyz,
    node_current_vec,
    current_blend: float = 0.35,
    tangent_scale: float = 0.50,
    points_per_segment: int = 5,
) -> list[np.ndarray]:
    """Convert graph paths to current-bent streamline-like curves."""
    streamlines = []
    for path in paths:
        sl = path_nodes_to_current_bent_curve(
            path,
            node_xyz,
            node_current_vec,
            current_blend=current_blend,
            tangent_scale=tangent_scale,
            points_per_segment=points_per_segment,
        )
        if sl is not None and sl.shape[0] >= 2:
            streamlines.append(sl)
    return streamlines
