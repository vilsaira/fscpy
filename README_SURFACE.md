# Voxelwise FSC with parcel-surface boundary conditions

This update replaces conductive ROI supernodes with individual active surface
voxels. A surface voxel is a labelled, non-WM voxel with at least one **face-
sharing**, FOD-supported contact to a voxel in the WM graph. Each contact
contributes an edge with conductance proportional to directional FOD support at
the WM endpoint. Conductances are **not normalized by parcel**. Only active
surface voxels are represented; the parcel interior is not part of the graph.

Active surface voxels of the **same** parcel have structural edges only when
both FODs support their 26-neighbour interface. Different parcels are never
connected directly by a surface edge.

FC imposes **equal potential at all active terminals of each parcel** and a
potential difference between the respective parcels. Equal-potential
constraints are implemented as zero-difference columns in the MNA constraint
matrix, not as structural edges. The constraint matrix includes separate
columns for the requested FC pairs.

**Consequence:** An FOD-supported edge connecting two terminals of the same
parcel carries zero current under these exact equal-potential constraints.
The ideal voltage sources may also inject/withdraw currents at distinct
terminals; only `result.edge_currents`, not the constraint-source currents,
represent structural currents. This does not model finite-conductivity flow
through a parcel interior.

## Calling the updated API

```python
from fsc import constraints_from_fc, solve_fsc
from fsc.voxel import graph_from_fod, add_roi_nodes, roi_constraints, voxel_support

wm_graph, voxel = graph_from_fod(fod_image, wm_mask=wm_mask)
graph, parcellation = add_roi_nodes(
    wm_graph, voxel, fod_image, labels,
    roi_labels=roi_labels, tissue_mask=tissue_mask,
)
pairs, fc_values = constraints_from_fc(fc)
incidence, targets = roi_constraints(parcellation, pairs, fc_values, graph.n_nodes)
result = solve_fsc(graph, incidence, targets)
support = voxel_support(graph, voxel, result.edge_currents)
```

`roi_labels` must match the FC matrix row/column order. Supply a `tissue_mask`
that excludes CSF when anatomical separation matters: parcellation labels alone
cannot reliably identify CSF. `wm_mask` and the FOD-norm threshold should be
intersected in your existing `graph_from_fod()`.

The FOD sampling sphere, SH convention, and directional support parameters
passed to `add_roi_nodes()` must match those of `graph_from_fod()`.

`voxel_support()` continues to report **white-matter voxel-edge currents** and
does not include ROI-to-WM contacts or the ROI surface edges. Zero-FC entries
are omitted by `constraints_from_fc()` unless explicitly supplied as pairs.
For published-style voxelwise network maps, use `sum_pairwise_support()` to
solve the FC pairs independently and sum absolute currents. A joint solve with
many pairwise FC constraints can be inconsistent; inspect solver residuals.

The original parcellation workflow (`solve_fsc(graph, node_pairs, fc_values)`)
continues to work unchanged.

## Testing

```bash
pip install -e '.[voxel,test]'
python -m pytest -q tests/test_roi_surface.py
python examples/voxel.py
```

The synthetic tests use a stub FOD evaluator and do not require DIPY/NiBabel.
The full NIfTI example requires both libraries and appropriate image data.
