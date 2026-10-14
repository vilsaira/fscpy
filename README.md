# fscpy

**Function–structure coupling (FSC)** solves a constrained graph-Laplacian problem to describe how a weighted structural network accommodates pairwise functional-connectivity (FC) constraints. Its primary outputs are graph-node potentials and currents along structural edges. These are *model-derived structural support*, not measured neural signalling or causal flow.

## Two structural representations, one numerical formulation

Both methods operate on connectomes. The distinction is the **spatial representation of the structural graph**, not whether a graph is a connectome.

| | Parcellation-level FSC (`fsc.parcellation`) | Voxel-wise FSC (`fsc.voxel`) |
|---|---|---|
| Nodes | Brain regions | FOD-supported voxels, plus parcellation supernodes |
| Structural edges | Nonzero regional SC entries (typically tractography-derived) | FOD-weighted interfaces on a 26-neighbour voxel lattice |
| Conductance | SC edge weight | Bidirectional directional support divided by physical edge length |
| Results | Currents on interregional structural connections | Currents on voxel interfaces and spatial support maps |

**These are different graphs and are not expected to give identical results.** They share only the constrained-Laplacian/MNA computation:

\[
\begin{bmatrix} L & B_{\mathrm{FC}} \\ B_{\mathrm{FC}}^T & 0 \end{bmatrix}
\begin{bmatrix} \phi \\ i_s \end{bmatrix}
= \begin{bmatrix}0 \\ s_{\mathrm{FC}} \end{bmatrix},
\qquad f=C_GB_G^T\phi.
\]

The `L` matrix, node/edge counts, and resulting currents depend on the input graph.

### Code organization

- `fsc.weighted_graph.WeightedGraph` is **the only numerical graph type**, storing node count, structural edge endpoints, and conductances.
- `fsc.voxel.VoxelMapping` stores voxel grid geometry, voxel-to-node indices, and **explicit voxel-edge indices** into that weighted graph. It is *not* another graph implementation.
- `fsc.voxel.ParcellationMapping` stores atlas labels, their corresponding graph **supernodes**, and explicit indices of their interface edges. It is *not* the ROI supernodes themselves or a competing graph type.
- `fsc.solver.solve_fsc()` accepts a `WeightedGraph` and FC constraints, regardless of which procedure constructed the graph.

The two high-level APIs are intentionally different because an SC matrix and an FOD volume are different kinds of input.

## Installation

Requires Python 3.10 or later. From the extracted source directory:

```bash
python -m pip install .                     # Core solver and parcellation FSC
python -m pip install '.[voxel]'             # Add NiBabel and DIPY for FODs
python -m pip install '.[voxel,paths,test]'  # Optional paths, examples, and tests
python -m pytest -q
```

The core solver and parcellation example need only NumPy and SciPy. The optional `paths` extra installs NetworkX for parcellation route summaries; voxel current-guided paths use SciPy.

## Parcellation-level FSC

A square symmetric SC matrix defines one node per region and one weighted edge per nonzero upper-triangle entry. The weights are used **directly as conductances**, without an intermediate resistance matrix.

```python
import numpy as np
from fsc.parcellation import graph_from_sc, solve_parcellation, edge_currents_to_matrix

sc = np.array([[0, 1, 0], [1, 0, 0.5], [0, 0.5, 0]])
fc = np.array([[0, 0, 3], [0, 0, 0], [3, 0, 0]])

graph = graph_from_sc(sc)
result = solve_parcellation(sc, fc)
current_matrix = edge_currents_to_matrix(graph, result.edge_currents, absolute=True)
print(current_matrix)
```

Here, a functional relationship without a direct SC edge can be supported through the intermediate region. `solve_parcellation()` solves all nonzero upper-triangle FC entries **jointly**; zero FC matrix entries are omitted. To explicitly impose a zero potential difference, pass a zero-valued pair to `solve_fsc()`.

Streamline assignments and filtering utilities live in `fsc.parcellation.streamlines`.

## Voxel-wise FSC

A spherical-harmonic FOD image defines a graph whose nodes are selected voxels. Undirected candidate edges use the 13 unique offsets corresponding to the 26-neighbour 3D lattice. Directional FOD support is evaluated at both ends of an interface; unsupported edges are removed.

Functional parcellations are added as graph supernodes connected to voxel interfaces. FC constraints are imposed **between these supernodes**, not between arbitrary voxels.

```python
import nibabel as nib
import numpy as np
from fsc import constraints_from_fc, solve_fsc
from fsc.voxel import (graph_from_fod, add_roi_nodes, labels_in_fod_space,
                       voxel_support, sum_pairwise_support)

fod_image = nib.load('wmfod.nii.gz')
label_image = nib.load('parcels.nii.gz')

voxel_graph, voxel_mapping = graph_from_fod(fod_image)
labels = labels_in_fod_space(label_image, fod_image)

graph, parcellation_mapping = add_roi_nodes(
    voxel_graph, voxel_mapping, labels, roi_labels=np.load('roi_order.npy')
)

fc = np.load('selected_fc.npy')  # FC rows follow parcellation_mapping.labels
pairs, values = constraints_from_fc(fc, parcellation_mapping.supernodes)
result = solve_fsc(graph, pairs, values)

support = voxel_support(graph, voxel_mapping, result.edge_currents)
print(support.shape, result.relative_residual)
```

`graph_from_fod()` returns `(WeightedGraph, VoxelMapping)`. `add_roi_nodes()` returns an **augmented** `(WeightedGraph, ParcellationMapping)`, leaving the original voxel mapping available for spatial output. `voxel_support()` uses `voxel_mapping.edge_indices` to select the FOD-derived voxel interfaces and deliberately excludes parcellation interface currents. This avoids depending on an implicit edge ordering in the combined graph.

The FOD image must use the specified DIPY SH basis (`tournier07`, `legacy=True` by default). Atlas images must already be registered to FOD image space; `labels_in_fod_space()` resamples with nearest neighbours but does not register images or rotate SH coefficients.

### Joint vs. separately solved FC pairs

The published voxel-wise network support analysis uses **separately solved** FC pairs and sums their absolute support maps:

```python
support = sum_pairwise_support(graph, voxel_mapping, parcellation_mapping, fc)
```

This is not generally equivalent to taking absolute currents from one joint multi-constraint solution. By default the function uses all nonzero upper-triangle pairs; an analysis can supply selected FC pair indices. L1-normalization and network pair selection are separate analysis steps.

## Numerical conventions

- **FC sign:** `solve_fsc(..., sign='legacy')` retains the historical constraint-incidence convention. For an isolated relationship it imposes `abs(FC)` as the potential difference. `sign='signed'` is an explicit alternative, not the published default.
- **Joint constraints:** Several FC constraints may be mutually inconsistent. Inspect `result.constraint_residual` and `result.relative_residual`, not only iterative-solver convergence.
- **FOD positivity:** The inherited implementation clips an angularly averaged FOD amplitude before subtracting the nonnegative isotropic mean. The voxel paper's pointwise-positive FOD expression is not necessarily identical if the FOD contains negative lobes. Verify scientifically before changing this calculation.
- **Parcellation interfaces:** Each atlas label is represented by a supernode connected to nearby voxels. By default, its total interface conductance is the median voxel-edge weight, split among its interfaces. This is a model choice, not a regional tractography edge.
- **Current-guided paths:** `fsc.voxel.paths.extract_roi_paths()` produces an optional representation of the solved current field; it is not a separate FSC solution or a tractography validation.

## Examples and tests

```bash
python examples/parcellation.py
python examples/voxel.py                       # Requires voxel dependencies
python -m pytest -q
python -m pytest tests/test_solver.py
python -m pytest tests/test_voxel_fod.py
python -m pytest tests/test_voxel_roi.py
```

The tests cover mathematical toy graphs, FC constraints, weighted edges, streamline mapping, FOD interfaces, supernode attachment, voxel support, current-guided paths, and minimal end-to-end workflows. `tests/data/legacy_reference.npz` contains numerical reference outputs from the earlier implementation and must not be regenerated using the rewritten implementation.

Routine synthetic tests do not need HCP data. The real-DIPY FOD regression skips automatically if the optional dependencies are not installed. Passing these tests does **not** establish that complete published HCP analyses have been rerun or reproduced.

## Publications

- Sairanen (2026). *From nodes to pathways: An edge-centric model of brain function–structure coupling via constrained Laplacians*. **Network Neuroscience**. https://doi.org/10.1162/NETN.a.589
- Sairanen (2026). *Voxel-wise function–structure coupling with diffusion-weighted and functional MRI*. **Aperture Neuro**.
