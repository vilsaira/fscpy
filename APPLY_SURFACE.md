# Applying this ROI boundary revision

This bundle is a **set of replacement files**, not a complete repository.
It is based on the latest `fscpy_roi_integration` changes and the completed
`voxel/fod.py` from our refactoring discussion.

Replace these files in your repository:

- `src/fsc/voxel/roi.py` — ROI surface nodes, WM contacts, optional supported surface edges, `roi_constraints()`.
- `src/fsc/voxel/support.py` — pairwise support uses the new constraints.
- `src/fsc/voxel/__init__.py` — exports `roi_constraints()`.
- `examples/voxel.py` — new example calls.
- `src/fsc/solver.py` — copy only if your local solver does not already accept a sparse constraint matrix. The enclosed solver retains support for ordinary node pairs.

Add `tests/test_roi_surface.py`, and adapt the voxel section of your README
using `README_SURFACE.md`.

Your existing `src/fsc/voxel/fod.py` remains untouched. It must provide:
`voxelhood_geometry`, `_neighbor_pair_slices`, `fod_kernels`,
`positive_fod_mean`, `directional_support`, and `VoxelMapping.n_voxels`.

Remove old tests that describe uniform supernode conductance, connected ROI
component supernodes, weighted-average component potentials, or the obsolete
`roi_constraint_matrix()` API. The previous exploratory `voxel/paths.py` also
assumes one node per ROI and is **not compatible**; remove it as previously
planned (and remove its corresponding test) or redesign it separately.

### Changed API

Old:

```python
incidence = roi_constraint_matrix(mapping, pairs, graph.n_nodes)
result = solve_fsc(graph, incidence, fc_values)
```

New:

```python
incidence, targets = roi_constraints(mapping, pairs, fc_values, graph.n_nodes)
result = solve_fsc(graph, incidence, targets)
```

`targets` has zeros for the within-ROI equal-potential constraints, followed
by the FC values. `result.source_currents` therefore contains currents for
*both* kinds of constraints, not just the FC pairs. Do not index the first
`len(pairs)` entries as FC-pair source currents.

### Model scope

Only ROI voxels that make supported face-sharing contacts with the WM graph
become surface nodes. The structural ROI–ROI surface edges are 26-neighbour
and require directional FOD support at both ends. There are **no edges through
the ROI interior**, no common conductive hub, and no per-ROI normalization.

The assumption that all active surface terminals of a functional parcel have
the same potential is enforced through ideal MNA constraints. Such equal-
potential contacts cannot carry currents along an ROI surface edge in a
consistent solution. Nonzero structural currents are measured on the
FOD-supported graph edges, while the ideal sources can still inject/withdraw
current at individual terminals.

### Validation

The new synthetic tests check topology, support gating, unnormalized
conductances, CSF mask restriction, original functional ROI ordering, and
exact equal-potential constraints. A real-data FOD and performance check is
still required; DIPY and NiBabel were not available in the test environment.
