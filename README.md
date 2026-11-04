# fscpy

**Function–structure coupling (FSC)** maps functional correlations onto the structural connections that can support them. FSC represents structural connectivity as weighted graph conductances, imposes functional connectivity as potential differences, and measures the resulting currents on structural edges.

The Python package is imported as `fsc` and provides two workflows:

- **Parcellation FSC:** a structural connectivity (SC) matrix and a functional connectivity (FC) matrix produce a symmetric matrix of structural-edge support. Related publication: the Network Neuroscience paper [doi:10.1162/NETN.a.589](https://doi.org/10.1162/NETN.a.589). See [Relation to the publications](#relation-to-the-publications) for how this implementation differs from the paper's joint formulation.
- **Voxelwise FSC:** a spherical-harmonic fibre orientation distribution (FOD) defines a weighted white-matter voxel graph. One supernode per functional parcel connects to neighbouring white-matter voxels through FOD-supported interfaces. The output is a voxelwise map of white-matter edge support. This version is explained in the Aperture Neuro paper [doi:10.52294/001c.172225](https://doi.org/10.52294/001c.172225).

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

The synthetic voxel example requires `.[voxel]` but **no downloaded MRI data**. It constructs an x-oriented FOD bundle in the MRtrix3 SH convention, attaches two functional ROI supernodes and evaluates the resulting support. The ENIGMA example follows the bundled cortical connectivity matrices returned by `load_sc()` and `load_fc()`.

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

FC must be square and symmetric. Only finite, positive values above the diagonal are used. Negative, zero and non-finite values are ignored. The papers used Fisher *z*-transformed FC; the package uses whatever values it is given, and the output scales linearly with the FC magnitudes.

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

#### Input conventions

- **SH basis.** The FOD image must contain SH coefficients in the MRtrix3 convention, as written by `dwi2fod`. This is DIPY's `tournier07` basis with `legacy=False`, which is the default. FODs in the older MRtrix 0.2 convention need `legacy=True`.
  Since version 2.0.2 the default is `legacy=False`; earlier versions defaulted to `legacy=True`. Pass `legacy=True` explicitly to reproduce results from those versions.
- **Same settings in both calls.** Pass identical `sphere`, `basis`, `legacy`, `kappa`, `kernel_power` and `orientation_floor` values to `graph_from_fod()` and `add_roi_nodes()`. They are not cross-checked.
- **Frame.** Neighbour directions are computed in world coordinates from the image affine, so the SH coefficients must be defined in that same frame, as MRtrix3 does. The image must have a correct affine.
- **Registration.** `labels_in_fod_space()` uses nearest-neighbour resampling and assumes the images are **already spatially registered**. It does not register images or reorient SH coefficients.
- **Masks.** An optional `wm_mask` restricts graph voxels and an optional `exclusion_mask` removes voxels. The FOD-norm threshold is always applied in addition to `wm_mask`.
- **ROIs.** ROI voxels must have positive labels, lie outside the selected WM voxel graph, and every ROI must have at least one FOD-supported WM contact, otherwise `add_roi_nodes()` raises an error. An optional `tissue_mask` excludes non-tissue ROI voxels.

#### How the graph is built

- **Nodes.** Voxels whose SH coefficient vector has a Euclidean norm above `fod_norm_threshold` (default 0.25), inside `wm_mask` if given. This threshold depends on the FOD intensity scale, so check it against your FOD normalisation.
- **Edges.** Each voxel is connected to its 26 neighbours (13 unique directions). The conductance of an interface is `sqrt(s_i · s_j) / ℓ`, where `ℓ` is the physical distance between voxel centres and `s` is the directional support of each endpoint. Interfaces below `min_conductance` (default 1e-12) are dropped.
- **Directional support.** The FOD amplitude along the interface direction is evaluated with an angular averaging kernel, with weights proportional to `exp(κ·|cos θ|^p)` over the sampling sphere (default `repulsion724`, `kappa=8`, `kernel_power=2`). It is clipped at zero after averaging, reduced by `orientation_floor` times the mean positive FOD amplitude of the voxel, and clipped at zero again. This kernel averaging is an implementation detail: the papers evaluate the FOD directly along the interface direction.
- **ROI supernodes.** Each ROI is a graph node. Labelled voxels outside the WM graph are ROI tissue; each one touches the WM graph through its six face neighbours. The conductance of a contact is `interface_scale` times the *WM voxel's* directional support along the face normal, divided by the voxel size. Diagonal contacts are not used, and support is taken from the WM side only. All ROI supernodes are present in the graph when any pair is solved.
- **Support map.** Voxel support is the sum of absolute currents on the WM–WM edges incident to a voxel, accumulated over all FC pairs. ROI–WM interface edges are excluded.

#### Solving

The voxel workflow applies each finite, positive FC pair **independently in the mathematical model**. For efficiency, the implementation solves reusable grounded-Laplacian basis fields relative to one ROI per connected component, then reconstructs pair solutions. It accumulates **absolute** structural currents. Pairs without a structural path contribute zero. Specify `pairs=[(0, 1)]` to select one pair by zero-based FC matrix indices.

`solver` can be `'auto'`, `'cg'` or `'direct'`; the default `rtol` is `1e-4`. Each solution is checked for current conservation, and the check is relative to the injected current, so it does not depend on the scale of the conductances. One potential field is stored per active ROI, and run time grows with the number of FC pairs.

## Numerical interpretation

In both workflows, only positive FC values define imposed potential differences; negative values are ignored. SC values and FOD interface weights act as conductances, **not lengths or resistances**. FSC reports how the weighted structural graph supports the imposed correlations; it is not a direct measurement of axonal traffic or a uniquely determined anatomical tract.

The low-level `fsc.solve_fsc()` interface solves **simultaneous** functional constraints through modified nodal analysis and returns an `FSCResult` (potentials, currents and residual diagnostics). The constraints must be mutually compatible. This holds for a sparse set of constraints, such as the five relationships of the FiberCup experiment. A dense FC matrix generally does not satisfy this, because it imposes far more potential differences than the graph has free potentials. For dense FC, use `solve_from_parcellations()` or `solve_fsc_voxel()`, which evaluate FC pairs separately and sum their absolute structural-edge currents.

## Relation to the publications

- **Network Neuroscience (parcellation workflow).** The paper formulates FSC as one joint modified-nodal-analysis system (its Eq. 11), with all constraints solved together. That corresponds to `fsc.solve_fsc()`. `solve_from_parcellations()` is a pairwise variant: it solves each FC pair separately and sums absolute currents, so its output is not the same quantity. The two agree for a single constraint and generally differ for several.
- **Aperture Neuro (voxel workflow).** The Methods describe imposing each selected FC value separately and accumulating the resulting current magnitudes, which is what `sum_pairwise_support()` does. The directional-support kernel, the six-face ROI contacts and the 0.25 norm threshold are implementation choices not given in the paper.

## Tests

Run the small deterministic tests without downloading neuroimaging datasets:

```bash
python -m pytest -q
```

The tests cover the five-node analytical solution, general MNA, negative FC, empty FC, ROI-basis accumulation, disconnected ROIs, WM-only voxel support, and the scale-invariance of the solver's convergence check. Optional tests run when NiBabel and DIPY are installed: a synthetic-FOD end-to-end example, and a check that the default SH basis reproduces amplitudes computed by MRtrix3 (`sh2amp`) from a reference FiberCup FOD stored in `tests/data/`.

## Research

The two FSC workflows are described in the following papers:

- **Parcellation workflow:** Sairanen, V. *From nodes to pathways: an edge-centric model of brain function–structure coupling via constrained Laplacians*. **Network Neuroscience** (2026). [doi:10.1162/NETN.a.589](https://doi.org/10.1162/NETN.a.589).

- **Voxel workflow:** Sairanen, V. *Voxel-wise function-structure coupling with diffusion-weighted and functional MRI*. **Aperture Neuro** (2026). [doi:10.52294/001c.172225](https://doi.org/10.52294/001c.172225).

Please, cite the publication corresponding to the workflow used.

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).
