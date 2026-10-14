"""Small FOD example using equal-potential ROI surface constraints."""

import nibabel as nib
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh

from fsc import constraints_from_fc, solve_fsc
from fsc.voxel import (
    add_roi_nodes, graph_from_fod, roi_constraints,
    sum_pairwise_support, voxel_support,
)

sphere = get_sphere(name="repulsion724")
amplitudes = 0.2 + 2 * sphere.vertices[:, 0] ** 4
coefficients = sf_to_sh(
    amplitudes, sphere, sh_order_max=4, basis_type="tournier07", legacy=True
)
fod = np.broadcast_to(coefficients, (5, 3, 3, len(coefficients))).copy()
image = nib.Nifti1Image(fod, np.diag([1.2, 1.5, 2., 1.]))

labels = np.zeros((5, 3, 3), dtype=np.int16)
labels[0] = 1
labels[-1] = 2

wm_mask = np.ones(labels.shape, dtype=bool)
wm_mask[0] = False
wm_mask[-1] = False

voxel_graph, voxel = graph_from_fod(image, wm_mask=wm_mask)
graph, parcellation = add_roi_nodes(
    voxel_graph, voxel, image, labels, roi_labels=[1, 2]
)

fc = np.array([[0, 0.8], [0.8, 0]])
roi_pairs, fc_values = constraints_from_fc(fc)
incidence, targets = roi_constraints(parcellation, roi_pairs, fc_values, graph.n_nodes)
result = solve_fsc(graph, incidence, targets)

print("Voxel nodes:", voxel.n_voxels)
print("ROI surface nodes:", len(parcellation.surface_nodes))
print("ROI-to-WM contacts:", len(parcellation.interface_edge_indices))
print("FOD-supported surface edges:", len(parcellation.surface_edge_indices))
print("Joint WM support:", voxel_support(graph, voxel, result.edge_currents).sum())
print("Pairwise WM support:", sum_pairwise_support(graph, voxel, parcellation, fc).sum())
print("Relative MNA residual:", result.relative_residual)
