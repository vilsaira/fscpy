# fscpy

**Function–structure coupling (FSC)** maps functional correlations onto the structural connections that can support them. FSC represents structural connectivity as weighted graph conductances, imposes functional connectivity as potential differences, and measures the resulting currents on structural edges.

The Python package is imported as `fsc` and provides two workflows:

- **Parcellation FSC:** a structural connectivity (SC) matrix and a functional connectivity (FC) matrix produce a symmetric matrix of structural-edge support. Relevant publication for this version is the Network Neuroscience paper 10.1162/NETN.a.589.
- **Voxelwise FSC:** a spherical-harmonic fibre orientation distribution (FOD) defines a weighted white-matter voxel graph. One supernode per functional parcel connects to neighbouring white-matter voxels through FOD-supported interfaces. The output is a voxelwise map of white-matter edge support. This version is explained in the Aperture Neuro paper 10.52294/001c.172225.

These workflows use different structural graphs, so their numerical outputs are **not interchangeable**.

## Installation

From the repository directory, preferably in a virtual environment:

```bash
python -m pip install -e .                  # Parcellation FSC (NumPy, SciPy)
python -m pip install -e '.[voxel]'          # Also NiBabel and DIPY
python -m pip install -e '.[test]'           # Run tests with pytest
python -m pip install -e '.[examples]'       # Matplotlib for the ENIGMA plot
```

On Windows PowerShell, use double quotes around extras, for example `python -m pip install -e ".[voxel,test,examples]"`.

The ENIGMA example also requires the external [ENIGMA Toolbox](https://github.com/MICA-MNI/ENIGMA), which is not a dependency of FSC. Please, refer to their installation instructions if needed.


## Examples

The scripts run from the repository root after installation:

| Script | Data | Output |
| --- | --- | --- |
| `python examples/parcellation.py` | Explicit 5×5 SC and FC matrices | A 5×5 symmetric FSC matrix with a known three-edge path |
| `python examples/enigma.py` | ENIGMA Toolbox's included HCP SC and FC matrices | Side-by-side SC, FC and FSC heatmaps |
| `python examples/voxel.py` | Generated synthetic SH FOD and two parcel voxels | A small 3D white-matter support map |

The synthetic voxel example requires `.[voxel]` but **no downloaded MRI data**. It constructs an x-oriented FOD bundle, attaches two functional ROI supernodes and evaluates the resulting support. The ENIGMA example follows the bundled cortical connectivity matrices returned by `load_sc()` and `load_fc()`.

### Parcellation workflow

```python
import numpy as np
import fsc

sc = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
], dtype=float)

fc = np.zeros((5, 5))
fc[1, 4] = fc[4, 1] = 1

support = fsc.solve_from_parcellations(sc, fc)
print(support)
```

`solve_from_parcellations()` returns the **symmetric support matrix**, not an `FSCResult` object. In this example, the only structural path connecting parcels 1 and 4 is 1–0–2–4. Its three unit-conductance edges each carry support `1/3`; the side branch 2–3 carries zero current.

### Voxel workflow

The runnable demonstration is [`examples/voxel.py`](examples/voxel.py). For actual FOD images and registered parcel labels, the workflow is:

```python
import nibabel as nib
import numpy as np
import fsc
from fsc.voxel import graph_from_fod, labels_in_fod_space, add_roi_nodes, sum_pairwise_support

fod_image = nib.load('fod_sh.nii.gz')
parcel_image = nib.load('parcellation_registered.nii.gz')
labels = labels_in_fod_space(parcel_image, fod_image)

fc = np.load('fc.npy')
roi_labels = np.load('roi_labels.npy')  # must match the FC matrix order

wm_graph, voxel_mapping = graph_from_fod(fod_image, fod_norm_threshold=0.25)
graph, parcellation = add_roi_nodes(
    wm_graph, voxel_mapping, fod_image, labels, roi_labels=roi_labels
)

support = sum_pairwise_support(
    graph, voxel_mapping, parcellation, fc, solver='cg', rtol=1e-6
)
nib.save(nib.Nifti1Image(support, fod_image.affine), 'fsc_support.nii.gz')
```

`labels_in_fod_space()` uses nearest-neighbour resampling and assumes the images are **already spatially registered**. It does not register images or reorient SH coefficients. By default, the FOD coefficient convention is DIPY's `tournier07` basis with `legacy=True`; the supplied image must use the matching convention. An optional `wm_mask` can further restrict graph voxels; the FOD-norm threshold is still applied. ROI voxels must have positive labels, lie outside the selected WM voxel graph, and have FOD-supported WM contacts.

The voxel workflow applies each finite, nonzero FC pair **independently in the mathematical model**. For efficiency, the implementation solves reusable grounded-Laplacian basis fields relative to one ROI per connected component, then reconstructs pair solutions. It accumulates **absolute** structural currents. Voxel support sums currents on **WM–WM edges only**, excluding ROI-supernode interface edges. Pairs without a structural path contribute zero. Specify `pairs=[(0, 1)]` to select one pair by zero-based FC matrix indices.

## Numerical interpretation

In both workflows, nonzero FC magnitudes define imposed potential differences and negative FC values are treated using `abs(FC)`. SC values and FOD interface weights act as conductances, **not lengths or resistances**. FSC reports how the weighted structural graph supports the imposed correlations; it is not a direct measurement of axonal traffic or a uniquely determined anatomical tract.

The low-level `fsc.solve_fsc()` interface solves **simultaneous** functional constraints through modified nodal analysis and returns an `FSCResult` (potentials, currents and residual diagnostics). This is distinct from `solve_from_parcellations()` and `solve_fsc_voxel()`, which evaluate FC pairs separately and sum their absolute structural-edge currents.

## Tests

Run the small deterministic tests without downloading neuroimaging datasets:

```bash
python -m pytest -q
```

The tests cover the five-node analytical solution, general MNA, negative FC, empty FC, ROI-basis accumulation, disconnected ROIs, and WM-only voxel support. An optional synthetic-FOD end-to-end test runs when NiBabel and DIPY are installed.

## Research

The two FSC workflows are described in the following papers:

- **Parcellation workflow:** Sairanen, V. *From nodes to pathways: an edge-centric model of brain function–structure coupling via constrained Laplacians*. **Network Neuroscience** (2026). [doi:10.1162/NETN.a.589](https://doi.org/10.1162/NETN.a.589).

- **Voxel workflow:** Sairanen, V. *Voxel-wise function-structure coupling with diffusion-weighted and functional MRI*. **Aperture Neuro** (2026). [doi:10.52294/001c.172225](https://doi.org/10.52294/001c.172225).

Please, cite the publication corresponding to the workflow used.

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).
