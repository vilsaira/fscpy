"""Voxelwise FSC on a tiny synthetic, x-aligned FOD fibre bundle.

No input files are needed. Requires the ``voxel`` extra (NiBabel and DIPY).
"""

import nibabel as nib
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh

import fsc
from fsc.voxel import add_roi_nodes, graph_from_fod, sum_pairwise_support

# Create an anisotropic FOD aligned with the x-axis, in Tournier07 SH order.
sphere = get_sphere(name="repulsion724")
profile = 0.2 + 4.0 * np.abs(sphere.vertices[:, 0]) ** 8
coefficients = sf_to_sh(
    profile, sphere, sh_order_max=4, basis_type="tournier07", legacy=True
)

fod = np.zeros((9, 5, 5, len(coefficients)), dtype=np.float32)
wm_mask = np.zeros(fod.shape[:3], dtype=bool)
wm_mask[2:7, 2, 2] = True
fod[wm_mask] = coefficients
fod_image = nib.Nifti1Image(fod, np.eye(4))

# Two non-WM parcel voxels touch the ends of the synthetic bundle.
labels = np.zeros(fod.shape[:3], dtype=int)
labels[1, 2, 2] = 1
labels[7, 2, 2] = 2
roi_labels = [1, 2]  # FC matrix order
fc = np.array([[0.0, 1.0], [1.0, 0.0]])

# A reduced isotropic floor makes this tiny artificial FOD robust as a demo.
wm_graph, voxel_mapping = graph_from_fod(
    fod_image, wm_mask=wm_mask, orientation_floor=0.5
)
graph, parcellation = add_roi_nodes(
    wm_graph, voxel_mapping, fod_image, labels,
    roi_labels=roi_labels, orientation_floor=0.5
)

support = sum_pairwise_support(
    graph, voxel_mapping, parcellation, fc,
    solver="cg", rtol=1e-8
)

assert np.any(support > 0)
print(f"WM nodes: {wm_graph.n_nodes}")
print(f"ROI supernodes: {parcellation.roi_nodes.tolist()}")
print("Support along bundle:", support[2:7, 2, 2])

# Uncomment to inspect the resulting 3D support image in another viewer.
# nib.save(nib.Nifti1Image(support, fod_image.affine), "synthetic_fsc.nii.gz")
