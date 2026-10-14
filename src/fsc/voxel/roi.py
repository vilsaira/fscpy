"""Connect functional parcels to white matter through FOD-supported surface voxels."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

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
    """Map functional ROI labels to individual active surface voxels.

    roi_labels gives the functional-connectivity matrix order.
    surface_nodes are separate graph nodes; sharing a label does not
    create an edge between them.
    """

    roi_labels: np.ndarray
    surface_labels: np.ndarray
    surface_nodes: np.ndarray
    surface_edge_indices: np.ndarray
    interface_edge_indices: np.ndarray


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
    """Add active ROI surface voxels and FOD-supported contacts to white matter.

    Only ROI voxels sharing a *face* with a white-matter graph voxel and
    having positive FOD-supported interface conductance become surface nodes.
    Neighbouring surface nodes belonging to the same ROI are connected only
    where both FODs support the interface (26-neighbour geometry).

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
        White-matter graph with ROI surface nodes and their FOD-weighted edges.
    mapping : ParcellationMapping
        Active ROI surface terminals and structural edge indices.
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

    # Face-sharing contacts only: the three unique axial offsets, both ways.
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
                coefficients, kernels[direction_id], baseline, orientation_floor
            )
            conductance = interface_scale * support / lengths[direction_id]
            valid = np.isfinite(conductance) & (conductance > min_conductance)
            if not np.any(valid):
                continue

            contact_surface_voxels.append(image_indices[tissue_slice][contacts][valid])
            contact_wm_nodes.append(voxel.node_id[wm_slice][contacts][valid])
            contact_conductances.append(conductance[valid].astype(np.float32))

    if not contact_conductances:
        raise ValueError("No FOD-supported ROI-to-white-matter contacts.")

    contact_surface_voxels = np.concatenate(contact_surface_voxels)
    contact_wm_nodes = np.concatenate(contact_wm_nodes)
    contact_conductances = np.concatenate(contact_conductances)

    # One graph node per distinct surface voxel with a supported WM contact.
    active_voxels, contact_to_surface = np.unique(
        contact_surface_voxels, return_inverse=True
    )
    surface_nodes = graph.n_nodes + np.arange(len(active_voxels))
    surface_labels = labels.flat[active_voxels]

    missing_rois = np.setdiff1d(roi_labels, surface_labels)
    if len(missing_rois):
        raise ValueError(f"No supported white-matter contact for ROIs {missing_rois.tolist()}.")

    surface_node_id = np.full(labels.shape, -1, dtype=int)
    surface_node_id.flat[active_voxels] = surface_nodes

    # Optional structural flow along the surface follows the same 26-neighbour FOD rule as the white-matter graph.
    surface_baseline = np.zeros(labels.shape, dtype=np.float32)
    surface_baseline.flat[active_voxels] = positive_fod_mean(
        fod.reshape(-1, fod.shape[-1])[active_voxels], sh_basis
    )

    surface_sources = []
    surface_targets = []
    surface_conductances = []

    for direction_id, offset in enumerate(offsets):
        source_slice, target_slice = _neighbor_pair_slices(offset)
        neighbors = (surface_node_id[source_slice] >= 0) & (surface_node_id[target_slice] >= 0)
        neighbors &= labels[source_slice] == labels[target_slice]
        if not np.any(neighbors):
            continue

        source_support = directional_support(
            fod[source_slice][neighbors], kernels[direction_id],
            surface_baseline[source_slice][neighbors], orientation_floor
        )
        target_support = directional_support(
            fod[target_slice][neighbors], kernels[direction_id],
            surface_baseline[target_slice][neighbors], orientation_floor
        )
        conductance = np.sqrt(source_support * target_support) / lengths[direction_id]
        valid = np.isfinite(conductance) & (conductance > min_conductance)
        if not np.any(valid):
            continue

        surface_sources.append(surface_node_id[source_slice][neighbors][valid])
        surface_targets.append(surface_node_id[target_slice][neighbors][valid])
        surface_conductances.append(conductance[valid].astype(np.float32))

    if surface_conductances:
        surface_sources = np.concatenate(surface_sources)
        surface_targets = np.concatenate(surface_targets)
        surface_conductances = np.concatenate(surface_conductances)
    else:
        surface_sources = np.empty(0, dtype=int)
        surface_targets = np.empty(0, dtype=int)
        surface_conductances = np.empty(0, dtype=np.float32)

    n_original_edges = len(graph.edge_i)
    n_surface_edges = len(surface_sources)
    augmented_graph = fsc.WeightedGraph(
        graph.n_nodes + len(surface_nodes),
        np.concatenate((graph.edge_i, surface_sources, surface_nodes[contact_to_surface])),
        np.concatenate((graph.edge_j, surface_targets, contact_wm_nodes)),
        np.concatenate((graph.conductance, surface_conductances, contact_conductances)),
    )

    mapping = ParcellationMapping(
        roi_labels=roi_labels,
        surface_labels=surface_labels,
        surface_nodes=surface_nodes,
        surface_edge_indices=np.arange(n_original_edges, n_original_edges + n_surface_edges),
        interface_edge_indices=np.arange(
            n_original_edges + n_surface_edges, len(augmented_graph.edge_i)
        ),
    )
    return augmented_graph, mapping


def roi_constraints(mapping, pairs, values, n_nodes):
    """Impose equal potentials on each ROI's active surface terminals.

    For every ROI, its surface terminals are constrained to the potential
    of its first terminal. For each FC pair, the two reference terminals
    have the specified potential difference. There are no conductive
    edges between terminals unless provided by add_roi_nodes().

    Parameters
    ----------
    mapping : ParcellationMapping
        Mapping returned by add_roi_nodes().
    pairs : ndarray, shape (m, 2)
        Zero-based indices into mapping.roi_labels / FC matrix order.
    values : ndarray, shape (m,)
        Functional correlation values for the selected pairs.
    n_nodes : int
        Number of nodes in the augmented graph.

    Returns
    -------
    constraint_matrix : scipy.sparse.csr_matrix
        Matrix with one zero-difference constraint for every additional
        terminal of each ROI, followed by the selected FC constraints.
    targets : ndarray
        Zero values for equal-potential constraints followed by FC values.
    """
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(pairs) != len(values):
        raise ValueError("FC pairs and values must have the same length.")
    if np.any(pairs < 0) or np.any(pairs >= len(mapping.roi_labels)):
        raise ValueError("FC pair index outside the parcellation.")

    rows = []
    columns = []
    coefficients = []
    anchors = np.empty(len(mapping.roi_labels), dtype=int)
    constraint_id = 0

    for roi_index in np.unique(pairs):
        label = mapping.roi_labels[roi_index]
        nodes = mapping.surface_nodes[mapping.surface_labels == label]
        if len(nodes) == 0:
            raise ValueError(f"ROI {label} has no surface terminals.")
        anchors[roi_index] = nodes[0]

        for node in nodes[1:]:
            rows.extend((node, nodes[0]))
            columns.extend((constraint_id, constraint_id))
            coefficients.extend((1.0, -1.0))
            constraint_id += 1

    n_equalities = constraint_id

    for source_roi, target_roi in pairs:
        rows.extend((anchors[source_roi], anchors[target_roi]))
        columns.extend((constraint_id, constraint_id))
        coefficients.extend((1.0, -1.0))
        constraint_id += 1

    incidence = sparse.coo_matrix(
        (coefficients, (rows, columns)), shape=(n_nodes, constraint_id)
    ).tocsr()
    targets = np.concatenate((np.zeros(n_equalities), values))
    return incidence, targets
