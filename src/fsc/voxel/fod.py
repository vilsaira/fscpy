"""Construct a weighted voxel graph from spherical-harmonic FOD coefficients."""

from dataclasses import dataclass

import numpy as np
import fsc


@dataclass
class VoxelMapping:
    """Map between image voxels and nodes or edges of the structural graph.

    Attributes
    ----------
    node_id : ndarray, shape (X, Y, Z)
        Graph node index for each image voxel. Voxels excluded from
        the structural graph are marked with -1.

    affine : ndarray, shape (4, 4)
        Image affine transforming voxel indices into physical coordinates.

    edge_indices : ndarray, shape (n_voxel_edges,)
        Indices identifying voxel-to-voxel edges in the WeightedGraph.
        These are used to exclude ROI-supernode interfaces when
        calculating voxel support.
    """

    node_id: np.ndarray
    affine: np.ndarray
    edge_indices: np.ndarray

    @property
    def n_voxels(self):
        """Number of image voxels represented as structural graph nodes."""
        return int(np.count_nonzero(self.node_id >= 0))

    def to_volume(self, node_values, fill=0):
        """Convert graph-node values into a 3D array in the image voxel space.

        Parameters
        ----------
        node_values : array_like
            Values indexed by structural graph node. May also contain
            additional ROI-supernode values, which are ignored.

        fill : float, default=0
            Value assigned to voxels excluded from the structural graph.

        Returns
        -------
        ndarray
            Three-dimensional float32 image with the same spatial
            dimensions as node_id.
        """
        volume = np.full(self.node_id.shape, fill, dtype=np.float32)
        voxel_mask = self.node_id >= 0

        volume[voxel_mask] = np.asarray(node_values)[self.node_id[voxel_mask]]

        return volume


def voxelhood_geometry(affine):
    """Return voxel offsets, physical distances, and unit directions.

    The 13 offsets represent the unique undirected connections in a
    26-neighbour 3D voxel grid.
    """
    offsets = np.array([
        [0, 0, 1],
        [0, 1, -1],
        [0, 1, 0],
        [0, 1, 1],
        [1, -1, -1],
        [1, -1, 0],
        [1, -1, 1],
        [1, 0, -1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, -1],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=int)

    physical_offsets = offsets @ np.asarray(affine)[:3, :3].T
    lengths = np.linalg.norm(physical_offsets, axis=1)
    directions = physical_offsets / lengths[:, None]

    return offsets, lengths, directions

def fod_kernels(n_coefficients, directions, *, sphere="repulsion724",
                basis="tournier07", legacy=True, kappa=8., power=2.):
    """Construct SH kernels for evaluating directional FOD amplitudes.

    The directional kernels calculate angularly weighted FOD amplitudes
    around the supplied voxel-neighbour directions. The SH basis is
    also returned for evaluating the FOD over the complete sampling sphere.

    Parameters
    ----------
    n_coefficients : int
        Number of SH coefficients describing each voxel's FOD.
    directions : ndarray, shape (n_directions, 3)
        Unit vectors defining the voxel-neighbour directions in physical space.
    sphere : str
        DIPY sphere used for sampling the FOD.
    basis : str
        SH basis convention used for the input FOD coefficients.
    legacy : bool
        Use DIPY's legacy convention for the selected SH basis.
    kappa : float
        Controls how strongly angular weights favour aligned directions.
    power : float
        Exponent applied to the absolute directional alignment.

    Returns
    -------
    directional_kernels : ndarray, shape (n_directions, n_coefficients)
        Kernels for angularly averaged directional FOD amplitudes.
    sh_basis : ndarray, shape (n_sphere_directions, n_coefficients)
        SH basis evaluated at each direction of the sampling sphere.
    """
    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf_matrix, calculate_max_order

    sh_order = calculate_max_order(n_coefficients)

    sampling_sphere = get_sphere(name=sphere)
    sh_matrix = sh_to_sf_matrix(
        sampling_sphere,
        sh_order_max=sh_order,
        basis_type=basis,
        legacy=legacy,
        return_inv=False
    )
    sh_basis = np.asarray(sh_matrix).T

    alignment = np.abs(directions @ sampling_sphere.vertices.T)
    angular_weights = np.exp(kappa * alignment**power)
    angular_weights /= angular_weights.sum(axis=1, keepdims=True)

    directional_kernels = angular_weights @ sh_basis

    return directional_kernels, sh_basis

def positive_fod_mean(coefficients, sh_basis):
    """Calculate the mean positive FOD amplitude over the sampling sphere.

    Parameters
    ----------
    coefficients : ndarray, shape (n_voxels, n_coefficients)
        Spherical harmonic FOD coefficients for the selected voxels.
    sh_basis : ndarray, shape (n_directions, n_coefficients)
        Spherical harmonic basis evaluated on the sampling sphere.

    Returns
    -------
    ndarray, shape (n_voxels,)
        Mean positive FOD amplitude for each voxel.
    """
    fod_amplitudes = coefficients @ sh_basis.T
    fod_amplitudes = np.maximum(fod_amplitudes, 0)

    return fod_amplitudes.mean(axis=1)

def directional_support(coefficients, kernel, isotropic_baseline,
                        orientation_floor=1.0):
    """Calculate directional FOD support above the isotropic baseline.

    The directional amplitude is evaluated using an angularly averaged
    SH kernel. Negative amplitudes are clipped to zero before subtracting
    the previously calculated positive-FOD spherical mean.
    """
    directional_amplitude = np.maximum(coefficients @ kernel, 0)

    return np.maximum(
        directional_amplitude - orientation_floor * isotropic_baseline, 0
    )


def _neighbor_pair_slices(offset):
    """Purpose of this helper: find two matching regions of a 3D image such that corresponding voxels are separated by a specified neighbour offset, excluding positions that would fall outside the image. 

    Each component of offset is -1, 0, or +1.
    """
    axis_slices = {
        -1: (slice(1, None), slice(None, -1)),
         0: (slice(None), slice(None)),
         1: (slice(None, -1), slice(1, None)),
    }

    source_x, target_x = axis_slices[offset[0]]
    source_y, target_y = axis_slices[offset[1]]
    source_z, target_z = axis_slices[offset[2]]

    source = (source_x, source_y, source_z)
    target = (target_x, target_y, target_z)

    return source, target


def fod_edges(fod, mask, node_id, offsets, lengths, kernels, isotropic_baseline,
              *, orientation_floor=1.0, min_conductance=1e-12):
    """Construct weighted structural edges between neighbouring voxels.

    For each of the 13 unique neighbour directions, the function identifies
    voxel pairs belonging to the structural graph and calculates their
    interface conductance from directional FOD support at both endpoints.

    Conductance is the geometric mean of endpoint support divided by
    the physical distance between voxel centres. Interfaces with
    conductance below min_conductance are excluded.

    The graph is undirected. Source and target only identify the two
    endpoints of each edge, not the direction of neural information flow.

    Parameters
    ----------
    fod : ndarray, shape (X, Y, Z, n_coefficients)
        Spherical harmonic FOD coefficients.
    mask : ndarray, shape (X, Y, Z)
        Boolean mask identifying voxels included in the structural graph.
    node_id : ndarray, shape (X, Y, Z)
        Graph node index for each voxel. Excluded voxels have index -1.
    offsets : ndarray, shape (13, 3)
        Unique neighbour displacements in voxel coordinates.
    lengths : ndarray, shape (13,)
        Physical distances corresponding to the neighbour offsets.
    kernels : ndarray, shape (13, n_coefficients)
        SH kernels for evaluating directional FOD amplitudes.
    isotropic_baseline : ndarray, shape (X, Y, Z)
        Precomputed mean positive FOD amplitude for each voxel.
    orientation_floor : float, default=1.0
        Scaling factor for isotropic baseline subtraction.
    min_conductance : float, default=1e-12
        Minimum conductance required to retain an edge.

    Returns
    -------
    source_nodes, target_nodes : ndarray, shape (n_edges,)
        Graph node indices of the retained structural edge endpoints.
    conductances : ndarray, shape (n_edges,)
        Conductance of each retained structural edge.
    """
    source_nodes = []
    target_nodes = []
    conductances = []

    for direction_id, offset in enumerate(offsets):
        source_slice, target_slice = _neighbor_pair_slices(offset)

        # Find neighbouring voxel pairs belonging to the structural graph.
        connected = mask[source_slice] & mask[target_slice]

        if not np.any(connected):
            continue

        source_fod = fod[source_slice][connected]
        target_fod = fod[target_slice][connected]

        source_baseline = isotropic_baseline[source_slice][connected]
        target_baseline = isotropic_baseline[target_slice][connected]

        # Calculate directional support at both endpoints.
        source_support = directional_support(
            source_fod, kernels[direction_id],
            source_baseline, orientation_floor
        )
        target_support = directional_support(
            target_fod, kernels[direction_id],
            target_baseline, orientation_floor
        )

        # Calculate conductance across each voxel interface.
        conductance = (
            np.sqrt(source_support * target_support) / lengths[direction_id]
        )

        valid = np.isfinite(conductance) & (conductance > min_conductance)

        if not np.any(valid):
            continue

        source_nodes.append(node_id[source_slice][connected][valid])
        target_nodes.append(node_id[target_slice][connected][valid])
        conductances.append(conductance[valid].astype(np.float32))

    if not conductances:
        raise ValueError("No FOD-supported voxel interfaces.")

    return (
        np.concatenate(source_nodes),
        np.concatenate(target_nodes),
        np.concatenate(conductances),
    )


def graph_from_fod(image, *, wm_mask=None, exclusion_mask=None,
                   fod_norm_threshold=0.25, sphere="repulsion724",
                   basis="tournier07", legacy=True, kappa=8.0,
                   kernel_power=2.0, orientation_floor=1.0,
                   min_conductance=1e-12):
    """Construct a voxel-wise weighted structural graph from an SH FOD image.

    Each included voxel becomes a graph node. Connections are constructed
    between neighbouring voxels in a 26-neighbour lattice, with conductances
    determined by directional FOD support at both endpoints.

    The graph is undirected. Functional parcel supernodes are not included
    here; they are added separately using the parcellation labels.

    Parameters
    ----------
    image : nibabel.Nifti1Image
        Four-dimensional NIfTI image containing SH FOD coefficients.
        The last dimension contains the coefficients for each voxel.
        FOD orientation and SH basis must match the image affine and
        the specified DIPY convention.

    wm_mask : ndarray, optional
        Boolean white matter mask restricting the graph to selected voxels.
        The FOD norm threshold is applied regardless of this mask.

    exclusion_mask : ndarray, optional
        Boolean 3D mask identifying voxels to exclude, regardless
        of their FOD amplitude or inclusion in wm_mask.

    fod_norm_threshold : float, default=0.25
        Minimum Euclidean norm of the SH coefficient vector required
        to include a voxel when wm_mask is not provided.

    sphere : str, default="repulsion724"
        DIPY sphere used to sample FOD amplitudes.

    basis : str, default="tournier07"
        SH basis convention used by the input FOD coefficients.

    legacy : bool, default=True
        DIPY legacy convention for the selected SH basis.

    kappa : float, default=8.0
        Angular selectivity parameter for the directional FOD kernels.

    kernel_power : float, default=2.0
        Exponent applied to directional alignment when constructing
        the angular averaging kernels.

    orientation_floor : float, default=1.0
        Scaling factor for subtracting the isotropic FOD baseline.

    min_conductance : float, default=1e-12
        Minimum conductance required to retain a voxel interface.

    Returns
    -------
    graph : WeightedGraph
        Undirected structural graph containing voxel nodes and
        their FOD-weighted connections.

    mapping : VoxelMapping
        Correspondence between image voxels and graph nodes or edges,
        together with the original image affine.
    """
    fod = image.get_fdata(dtype=np.float32)

    # Select voxels with sufficient FOD amplitude.
    mask = np.linalg.norm(fod, axis=3) > fod_norm_threshold

    # Restrict selection to white matter when a mask is provided.
    if wm_mask is not None:
        if np.shape(wm_mask) != mask.shape:
            raise ValueError("White matter mask dimensions differ.")
        mask &= np.asarray(wm_mask, dtype=bool)

    # Exclude unwanted regions.
    if exclusion_mask is not None:
        if np.shape(exclusion_mask) != mask.shape:
            raise ValueError("Exclusion mask dimensions differ.")
        mask &= ~np.asarray(exclusion_mask, dtype=bool)

    # Assign a graph node index to each included voxel.
    n_voxels = int(mask.sum())
    if n_voxels == 0:
        raise ValueError("Empty voxel graph.")

    node_id = np.full(mask.shape, -1, dtype=np.int64)
    node_id[mask] = np.arange(n_voxels)

    # Calculate voxel-neighbour geometry and FOD kernels.
    offsets, lengths, directions = voxelhood_geometry(image.affine)

    kernels, sh_basis = fod_kernels(
        fod.shape[-1], directions,
        sphere=sphere, basis=basis, legacy=legacy,
        kappa=kappa, power=kernel_power
    )

    # Calculate the positive-FOD spherical mean once for each voxel.
    isotropic_baseline = np.zeros(mask.shape, dtype=np.float32)
    isotropic_baseline[mask] = positive_fod_mean(fod[mask], sh_basis)

    # Construct the FOD-weighted voxel interfaces.
    source_nodes, target_nodes, conductances = fod_edges(
        fod, mask, node_id, offsets, lengths, kernels, isotropic_baseline,
        orientation_floor=orientation_floor,
        min_conductance=min_conductance
    )

    graph = fsc.WeightedGraph(
        n_voxels, source_nodes, target_nodes, conductances
    )

    mapping = VoxelMapping(
        node_id,
        np.asarray(image.affine),
        np.arange(len(conductances))
    )

    return graph, mapping