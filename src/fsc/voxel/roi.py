"""Connect functional parcels to white matter through FOD-supported surface voxels."""

from dataclasses import dataclass

import numpy as np

import fsc
from .fod import (
    _neighbor_pair_slices,
    directional_support,
    fod_kernels,
    positive_fod_mean,
    voxelhood_geometry,
)


@dataclass
class ParcellationMapping:
    """Map functional ROIs to graph supernodes and WM interfaces."""

    roi_labels: np.ndarray
    roi_nodes: np.ndarray
    interface_edge_indices: np.ndarray
    contact_voxel_indices: np.ndarray


def labels_in_fod_space(label_image, fod_image):
    """Resample registered parcellation labels into FOD space."""
    from nibabel.processing import resample_from_to

    target = (fod_image.shape[:3], fod_image.affine)
    resampled = resample_from_to(label_image, target, order=0)
    return np.rint(resampled.get_fdata()).astype(np.int32)


def add_roi_nodes(graph, voxel, fod_image, labels, roi_labels=None, *,
                  tissue_mask=None, sphere="repulsion724", basis="tournier07",
                  legacy=True, kappa=8.0, kernel_power=2.0,
                  orientation_floor=1.0, interface_scale=1.0,
                  min_conductance=1e-12):
    """Add one ROI supernode per parcel and connects it to the white-matter graph through FOD-weighted edges.

    Parameters
    ----------
    graph : WeightedGraph
        FOD-weighted white-matter voxel graph.
    voxel : VoxelMapping
        Image-grid to white-matter graph-node mapping.
    fod_image : nibabel.Nifti1Image
        SH FOD image used to construct the white-matter graph.
    labels : ndarray, shape (X, Y, Z)
        Functional parcellation labels in FOD space.
    roi_labels : ndarray, optional
        ROI labels in FC matrix order. Defaults to all positive labels.
    tissue_mask : ndarray, optional
        Mask of valid ROI tissue, excluding CSF, in FOD space.
    sphere, basis, legacy, kappa, kernel_power, orientation_floor
        FOD conventions; use the same settings as graph_from_fod().
    interface_scale : float, default=1.0
        Global multiplier for ROI-to-white-matter contact conductance.
    min_conductance : float, default=1e-12
        Minimum conductance for a retained interface.

    Returns
    -------
    augmented_graph : WeightedGraph
        White-matter graph with one supernode per ROI and FOD-weighted
        ROI-to-white-matter connections.
    mapping : ParcellationMapping
        ROI supernode identifiers and corresponding interface edges.
    """
    fod = fod_image.get_fdata(dtype=np.float32)
    labels = np.asarray(labels)

    if labels.shape != voxel.node_id.shape or fod.shape[:3] != labels.shape:
        raise ValueError("FOD, parcellation and voxel mapping grids must agree.")
    if graph.n_nodes != voxel.n_voxels:
        raise ValueError("Graph nodes do not match the voxel mapping.")
    if not np.allclose(fod_image.affine, voxel.affine):
        raise ValueError("FOD image and voxel mapping affines differ.")

    if roi_labels is None:
        roi_labels = np.unique(labels[labels > 0])
    else:
        roi_labels = np.asarray(roi_labels, dtype=int)
    if len(roi_labels) == 0 or np.any(roi_labels <= 0):
        raise ValueError("ROI labels must be nonempty and positive.")
    if len(np.unique(roi_labels)) != len(roi_labels):
        raise ValueError("ROI labels must be unique.")

    # Tissue voxels and white-matter graph voxels are distinct nodes.
    tissue = np.isin(labels, roi_labels) & (voxel.node_id < 0)
    if tissue_mask is not None:
        if np.shape(tissue_mask) != labels.shape:
            raise ValueError("Tissue mask and parcellation grids differ.")
        tissue &= np.asarray(tissue_mask, dtype=bool)

    offsets, lengths, directions = voxelhood_geometry(fod_image.affine)
    kernels, sh_basis = fod_kernels(
        fod.shape[-1], directions, sphere=sphere, basis=basis,
        legacy=legacy, kappa=kappa, power=kernel_power
    )

    # A voxel identifier in image space lets us collect contacts without
    # constructing a graph node for any interior ROI tissue voxel.
    image_indices = np.arange(labels.size).reshape(labels.shape)
    contact_surface_voxels = []
    contact_wm_nodes = []
    contact_conductances = []

    
    # Collect FOD-supported contacts between ROI tissue and WM voxels.
    for direction_id, offset in enumerate(offsets):
        if np.count_nonzero(offset) != 1:
            continue

        source_slice, target_slice = _neighbor_pair_slices(offset)

        for tissue_slice, wm_slice in (
            (source_slice, target_slice),
            (target_slice, source_slice),
        ):
            contacts = tissue[tissue_slice] & (voxel.node_id[wm_slice] >= 0)

            if not np.any(contacts):
                continue

            coefficients = fod[wm_slice][contacts]
            baseline = positive_fod_mean(coefficients, sh_basis)

            support = directional_support(
                coefficients,
                kernels[direction_id],
                baseline,
                orientation_floor,
            )

            conductance = interface_scale * support / lengths[direction_id]
            valid = np.isfinite(conductance) & (conductance > min_conductance)

            if not np.any(valid):
                continue

            contact_surface_voxels.append(
                image_indices[tissue_slice][contacts][valid]
            )
            contact_wm_nodes.append(
                voxel.node_id[wm_slice][contacts][valid]
            )
            contact_conductances.append(
                conductance[valid].astype(np.float32)
            )

    if not contact_conductances:
        raise ValueError("No FOD-supported ROI-to-WM contacts.")

    contact_surface_voxels = np.concatenate(contact_surface_voxels)
    contact_wm_nodes = np.concatenate(contact_wm_nodes)
    contact_conductances = np.concatenate(contact_conductances)


    # One graph node per functional ROI.
    roi_nodes = graph.n_nodes + np.arange(len(roi_labels))

    # Map each interface voxel to its ROI supernode.
    contact_labels = labels.flat[contact_surface_voxels]

    contact_roi_nodes = np.empty(len(contact_labels), dtype=int)

    for index, label in enumerate(roi_labels):
        selected = contact_labels == label
        contact_roi_nodes[selected] = roi_nodes[index]

        if not np.any(selected):
            raise ValueError(f"ROI {label} has no WM contacts.")

    # Append the ROI-WM edges to the existing structural graph.
    augmented_graph = fsc.WeightedGraph(
        n_nodes=graph.n_nodes + len(roi_nodes),
        edge_i=np.concatenate((graph.edge_i, contact_roi_nodes)),
        edge_j=np.concatenate((graph.edge_j, contact_wm_nodes)),
        conductance=np.concatenate((graph.conductance, contact_conductances)),
    )

    n_wm_edges = len(graph.edge_i)

    mapping = ParcellationMapping(
        roi_labels=roi_labels,
        roi_nodes=roi_nodes,
        interface_edge_indices=np.arange(
            n_wm_edges, len(augmented_graph.edge_i)
        ),
        contact_voxel_indices=contact_surface_voxels,
    )

    return augmented_graph, mapping


def roi_constraints(mapping, pairs, values, n_nodes):
    """Construct functional potential constraints between ROI supernodes."""
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    values = np.asarray(values, dtype=float).reshape(-1)

    if len(pairs) != len(values):
        raise ValueError("FC pairs and values must have the same length.")

    if np.any(pairs < 0) or np.any(pairs >= len(mapping.roi_labels)):
        raise ValueError("FC pair index outside the parcellation.")

    return fsc.functional_connectivity_from_fc_pairs(
        mapping.roi_nodes[pairs],
        values,
        n_nodes,
    )
