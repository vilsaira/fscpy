# fscpy

Function–structure coupling (FSC) on weighted structural graphs using constrained Laplacians and modified nodal analysis. The package provides two structural representations: regional connectivity matrices and voxelwise fibre-orientation-distribution (FOD) graphs.

## Install

From the repository directory:

```bash
python -m pip install -e .             # Regional FSC
python -m pip install -e '.[voxel]'     # Also install NiBabel and DIPY
```

## Regional FSC

```python
import numpy as np
from fsc import solve_from_parcellations

sc = np.array([[0, 1, 0],
               [1, 0, 0.5],
               [0, 0.5, 0]])
fc = np.array([[0, 0, 1],
               [0, 0, 0],
               [1, 0, 0]])

result = solve_from_parcellations(sc, fc)
print(result.potentials)
print(result.edge_currents)
print(result.relative_residual)
```

`sc` supplies edge conductances. Nonzero, finite upper-triangle entries of `fc` define potential-difference constraints; their absolute magnitudes are used. Multiple FC constraints are solved jointly, so inspect residuals for consistency.

## Voxelwise FSC

This example assumes a spherical-harmonic FOD image, a registered parcellation, and white-matter/ROI-tissue masks **already in FOD voxel space**. The FC matrix must follow the label order in `roi_labels.npy`.

```python
import nibabel as nib
import numpy as np
from fsc.voxel import (
    graph_from_fod,
    labels_in_fod_space,
    add_roi_nodes,
    sum_pairwise_support,
)

fod_image = nib.load('fod_sh.nii.gz')
parcellation_image = nib.load('parcellation.nii.gz')
wm_mask = nib.load('wm_mask_fod_space.nii.gz').get_fdata() > 0
tissue_mask = nib.load('roi_tissue_mask_fod_space.nii.gz').get_fdata() > 0

fc = np.load('fc.npy')
roi_labels = np.load('roi_labels.npy')

wm_graph, voxel_mapping = graph_from_fod(fod_image, wm_mask=wm_mask)
labels = labels_in_fod_space(parcellation_image, fod_image)

graph, parcellation = add_roi_nodes(
    wm_graph, voxel_mapping, fod_image, labels,
    roi_labels=roi_labels, tissue_mask=tissue_mask,
)

# Solve each nonzero FC pair independently; sum absolute WM edge currents.
support = sum_pairwise_support(graph, voxel_mapping, parcellation, fc)
nib.save(nib.Nifti1Image(support, fod_image.affine), 'support.nii.gz')
```

Only ROI surface voxels with FOD-supported white-matter contacts enter the augmented graph. Active terminals of a parcel share an imposed potential; this does not add structural edges through parcel interiors, but equality constraints can carry source currents. The support volume sums currents on **WM–WM edges only**; ROI contact currents are excluded. For a single selected FC pair, pass `pairs=[(0, 1)]` to `sum_pairwise_support` (zero-based FC indices).

The input FOD coefficient ordering must match the DIPY `tournier07` SH basis with `legacy=True` unless you explicitly select matching alternatives. Images must be registered; the resampling helper does not perform registration or SH reorientation.
