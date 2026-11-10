# fscpy
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.CONCEPT.svg)](https://doi.org/10.5281/zenodo.CONCEPT)


**Function–structure coupling (FSC)** maps functional correlations onto the structural connections that can support them. FSC represents structural connectivity as weighted graph conductances, imposes functional connectivity as potential differences, and measures the resulting currents on structural edges.

The Python package is imported as `fsc` and provides two workflows:

- **Parcellation FSC:** a structural connectivity (SC) matrix and a functional connectivity (FC) matrix produce a symmetric matrix of structural-edge support. Related publication: the Network Neuroscience paper [doi:10.1162/NETN.a.589](https://doi.org/10.1162/NETN.a.589). The paper's joint formulation is available as `method="joint"` for FC without cycles, and a pairwise variant for dense FC is the default; see [Relation to the publications](#relation-to-the-publications).
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

`solve_from_parcellations()` returns the **symmetric support matrix**, not an `FSCResult` object. In this example, the only structural path connecting parcels 1 and 4 is 1–0–2–4. Its three unit-conductance edges each carry support `1/3` and the side branch 2–3 carries zero current.

FC must be square and symmetric. Only finite, positive values above the diagonal are used. Negative, zero and non-finite values are ignored. The papers used Fisher *z*-transformed FC. The fscpy uses whatever values it is given. The output scales linearly with the FC magnitudes.

`solve_from_parcellations()` has two methods:

- `method="pairwise"` (default) solves each FC pair separately and sums the absolute edge currents. It works for any FC matrix.
- `method="joint"` solves all FC constraints in one modified-nodal-analysis system as in the Network Neuroscience paper (Eq. 11). Returns absolute edge currents. Requires the FC pairs to contain no cycles, e.g. a set of disjoint pairs like the FiberCup example. Otherwise, error is raised. Three regions with positive FC between all of them already form a cycle. So, a dense FC matrix has likely no exact joint solution.

```python
fc = np.zeros((5, 5))
fc[1, 4] = fc[4, 1] = 1
fc[3, 4] = fc[4, 3] = 1      # two pairs sharing node 4: no cycle

joint = fsc.solve_from_parcellations(sc, fc, method="joint")
pairwise = fsc.solve_from_parcellations(sc, fc)   # different for several pairs
```

The two methods agree for a single FC pair and generally differ for several, because the joint solve combines the constraints in one potential field while the pairwise method sums the magnitudes of separate solutions.

#### Mapping support to streamlines

The support matrix can be mapped back onto a tractogram so that each streamline gets a weight for visualisation, filtering, or analyses. The SC matrix does not have to come from fscpy. Any tool that writes a per-streamline parcel assignment, such as MRtrix3 `tck2connectome -out_assignments`, works.

```python
from fsc.parcellation import streamline_support

assignments = np.loadtxt("assignments.txt", dtype=int)  # two parcel labels per streamline
weights = np.loadtxt("sift2_weights.txt")               # optional, see below

values = streamline_support(support, assignments, weights)
np.savetxt("fsc_streamline_weights.txt", values)        # one value per streamline, input order
```

The support of each parcel pair is split among the streamlines connecting that pair in proportion to their weights. So the values of each pair's streamlines sum to `support[a, b]`. This holds for both methods and for SC from any source or normalisation. Only the relative weights within a pair matter.

Requirements:

- **Assignments:** exactly two labels per streamline, numbered from 1 in SC matrix order, with 0 for an unassigned endpoint. This is the default `tck2connectome -out_assignments` format. Options that write more labels per streamline (e.g. `-assignment_all_voxels`), or that add a row for node 0 to the matrix (`-keep_unassigned`) break this correspondence!
- **Weights:** use the per-streamline weights the SC was built from. Omit them for streamline counts, pass the SIFT2 weights file if the SC used `-tck_weights_in`, or `1 / length` for length-scaled SC. Non-finite or non-positive weights mark a streamline as invalid.
- **Output:** one non-negative value per input streamline, in input order. Unassigned, invalid and self-connecting streamlines get 0. If the SC has an edge with no streamline in `assignments`, that edge's support is not mapped.

For signed streamline currents from a potential vector computed with `fsc.solve_fsc()`, use `streamline_currents()`. If SC is built inside fscpy from streamline resistances, use `sc_from_streamlines()`.


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
- **Same settings in both calls.** Pass identical `sphere`, `basis`, `legacy`, `kappa`, `kernel_power`, `orientation_floor` and `length_power` values to `graph_from_fod()` and `add_roi_nodes()`. They are not cross-checked.
- **Frame.** Neighbour directions are computed in world coordinates from the image affine, so the SH coefficients must be defined in that same frame, as MRtrix3 does. The image must have a correct affine.
- **Registration.** `labels_in_fod_space()` uses nearest-neighbour resampling and assumes the images are **already spatially registered**. It does not register images or reorient SH coefficients.
- **Masks.** An optional `wm_mask` restricts graph voxels and an optional `exclusion_mask` removes voxels. The FOD-norm threshold is always applied in addition to `wm_mask`.
- **ROIs.** ROI voxels must have positive labels and lie outside the selected WM voxel graph. A supernode without any FOD-supported WM contact is kept, so FC indexing is unchanged, but carries no current. Function add_roi_nodes() lists all such labels in one warning, or raises with missing_contacts="raise". An optional `tissue_mask` excludes non-tissue ROI voxels.

#### How the graph is built

- **Nodes.** Voxels whose SH coefficient vector has a Euclidean norm above `fod_norm_threshold` (default 0.25), inside `wm_mask` if given. This threshold depends on the FOD intensity scale, so check it against your FOD normalisation.
- **Edges.** Each voxel is connected to its 26 neighbours (13 unique directions). The conductance of an interface is `sqrt(s_i · s_j) / ℓ^p`, where `ℓ` is the physical distance between voxel centres, `s` is the directional support of each endpoint and `p = length_power` (default 2). With `p = 2` each link gets its share of the cross-section, so tracts aligned with an axis or a lattice diagonal conduct equally. Interfaces below min_conductance (default 1e-12) are dropped.
- **Directional support.** The FOD amplitude along the interface direction is evaluated with an angular averaging kernel, with weights proportional to `exp(κ·|cos θ|^p)` over the sampling sphere (default `repulsion724`, `kappa=8`, `kernel_power=2`). It is clipped at zero after averaging, reduced by `orientation_floor` times the mean positive FOD amplitude of the voxel, and clipped at zero again. This kernel averaging is an implementation detail: the papers evaluate the FOD directly along the interface direction.
- **ROI supernodes.** Each ROI is a graph node. Labelled voxels outside the WM graph are ROI tissue. Each one connects to adjacent WM voxels across all 26 neighbours, like the WM graph. The conductance of a contact is interface_scale times the WM voxel's directional support along the contact direction divided by `ℓ^p`. Support is taken from the WM side only. When an FC pair is solved, only the two ROIs of that pair are in the graph. All other ROIs and their contacts are left out. This is done to avoid supernodes from conducting currents between bundles.
- **Support map.** Voxel support is the sum of absolute currents on the WM–WM edges incident to a voxel, accumulated over all FC pairs. ROI–WM interface edges are excluded.
- **Known limitation: orientation dependence.** Links exist only to the 26 neighbours, so a tract running between lattice directions (for example 10°–30° off an axis) has to zig-zag and appears more resistive than the same tract aligned with the grid. In synthetic tests with sharp FODs this made the resistance per mm 1.5–2.5× higher for most such angles, and up to about 25× in the sharpest case.

#### Solving

The voxel workflow solves each finite, positive FC pair separately. For a pair of ROIs (a, b), the graph consists of the white-matter voxels and the contacts of ROIs a and b only. ROI b is grounded, unit current is injected at ROI a, and the solution is scaled so that the potential difference equals the FC value. Absolute structural currents are accumulated over pairs. Pairs without a structural path contribute zero. Specify `pairs=[(0, 1)]` to select one pair by zero-based FC matrix indices.

`solver` can be `auto`, `cg` or `direct`. The default rtol is 1e-4. Each solution is checked for current conservation, and the check is relative to the injected current, so it does not depend on the scale of the conductances. Each FC pair needs one linear solve. Run time grows linearly with the number of pairs. 

## Numerical interpretation

In both workflows, only positive FC values define imposed potential differences. Negative values are ignored. SC values and FOD interface weights act as conductances, **not lengths or resistances**. FSC reports how the weighted structural graph supports the imposed correlations. It is not a direct measurement of axonal traffic or a uniquely determined anatomical tract.

The low-level `fsc.solve_fsc()` interface solves **simultaneous** functional constraints through modified nodal analysis and returns an `FSCResult` (potentials, currents and residual diagnostics). The constraints must be mutually compatible. The imposed potential differences must sum to zero around every cycle of constrained pairs. Constraints without cycles, e.g. the FiberCup experiment, always satisfy this. Whereas a dense FC matrix likely does not because it imposes far more potential differences than the graph has free potentials. `solve_from_parcellations(..., method="joint")` wraps this interface and checks for cycles before solving. For dense FC, use `solve_from_parcellations()` with the default `method="pairwise"`, or `solve_fsc_voxel()`, which evaluate FC pairs separately and sum their absolute structural-edge currents.

## Relation to the publications

- **Network Neuroscience (parcellation workflow).** The paper formulates FSC as one joint modified-nodal-analysis system (Eq. 11), with all constraints solved together. `solve_from_parcellations(..., method="joint")` implements it for FC pairs without cycles like in the FiberCup specificity test. When FC pairs form cycles, as many dense FC matrices, the constraints cannot all hold at once and the joint system has no exact solution. For such inputs use the default `method="pairwise"`, which solves each pair separately and sums absolute currents. The two methods agree for a single constraint.
- **Aperture Neuro (voxel workflow).** The Methods describe imposing each selected FC value separately and accumulating the resulting current magnitudes, which is what sum_pairwise_support() does. The directional-support kernel and the 0.25 norm threshold are implementation choices not given in the paper. **Since version 2.1.0 the voxel graph differs from the published method:** ROIs connect to white matter across all 26 neighbours (the paper used face neighbours), conductance scales as 1/ℓ² instead of the paper's 1/ℓ to match this change (Eq. 4), and each FC pair is solved with only its own two ROIs in the graph. 

## Tests

Run the small deterministic tests without downloading neuroimaging datasets:

```bash
python -m pytest -q
```

The tests cover the five-node analytical solution, general MNA, the joint parcellation method and its rejection of FC with cycles, mapping support to streamlines, negative FC, empty FC, ROI-basis accumulation, disconnected ROIs, ROI contacts for oblique tracts, equal conduction along axis and diagonal tracts, WM-only voxel support, and the scale-invariance of the solver's convergence check. Optional tests run when NiBabel and DIPY are installed: a synthetic-FOD end-to-end example, and a check that the default SH basis reproduces amplitudes computed by MRtrix3 (`sh2amp`) from a reference FiberCup FOD stored in `tests/data/`.

## Research

The two FSC workflows are described in the following papers:

- **Parcellation workflow:** Sairanen, V. *From nodes to pathways: an edge-centric model of brain function–structure coupling via constrained Laplacians*. **Network Neuroscience** (2026). [doi:10.1162/NETN.a.589](https://doi.org/10.1162/NETN.a.589).

- **Voxel workflow:** Sairanen, V. *Voxel-wise function-structure coupling with diffusion-weighted and functional MRI*. **Aperture Neuro** (2026). [doi:10.52294/001c.172225](https://doi.org/10.52294/001c.172225).

Please, cite the publication corresponding to the workflow used.

- **Software:** Sairanen, V. *fscpy: Function-structure coupling*. Zenodo. [doi:10.5281/zenodo.CONCEPT](https://doi.org/10.5281/zenodo.CONCEPT). Cite the version DOI shown on Zenodo for the release you used.



## License

GNU General Public License v3.0; see [LICENSE](LICENSE).
