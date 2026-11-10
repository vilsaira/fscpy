## 2.1.1 (2026-11-10)

Results are unchanged within the solver tolerance.

### Added
- Voxel solver: optional algebraic multigrid (AMG) preconditioning. With PyAMG
  installed (`pip install "fscpy[amg]"`), `solve_fsc_voxel(..., solver="auto")`
  uses CG with a smoothed-aggregation AMG preconditioner. On an HCP subject
  (about 315,000 WM voxels) this cut CG iterations per pair from about 740 to
  14–20 and the time per pair from about 7.5 s to about 4 s. Without PyAMG,
  `"auto"` uses Jacobi-preconditioned CG as before; `solver="amg"` requires PyAMG.

### Documentation
- README: running chunks of FC pairs in parallel processes or batch jobs.


## 2.1.0 (2026-11-10)
Archived on Zenodo: [doi:10.5281/zenodo.24104201](https://doi.org/10.5281/zenodo.24104201)

> **Voxel results differ from the Aperture Neuro paper.** The paper's method
> connected ROIs to white matter through face neighbours only, scaled edge
> conductance by 1/ℓ (its Eq. 4), and kept every ROI in the graph for every FC
> pair. Version 2.1.0 changes all three, see below.


### Changed (affects results)
- Voxel workflow: ROI supernodes now connect to white matter across all 26
  neighbours instead of the 6 face neighbours used in the paper. Tracts meeting
  an ROI obliquely were previously connected poorly or not at all, and
  `add_roi_nodes` could fail with "No FOD-supported ROI-to-WM contacts".
- Voxel workflow: edge and ROI-contact conductance now scales as 1/ℓ² by
  default (`length_power=2`) instead of 1/ℓ. With 1/ℓ, tracts along lattice
  diagonals conducted 1.4–1.9× better than identical axis-aligned tracts.
  `length_power=1` restores the paper's distance scaling, but not its
  face-only ROI contacts.
- Voxel workflow: each FC pair is now solved with only its own two ROIs in the
  graph. Previously every ROI stayed connected, so unrelated ROIs acted as
  perfect conductors between bundles. `ground_node` in `solve_fsc_voxel` is now ignored.


### Changed
- `add_roi_nodes()` no longer stops at the first supernode without WM
  contacts. Such supernodes carry no current and are listed in one warning;
  `missing_contacts="raise"` restores the error, now listing all labels.


## 2.0.2

### Changed (affects results)
- Voxel workflow: the default SH basis is now `legacy=False`, which matches
  MRtrix3 `dwi2fod` output. With the old default, MRtrix3 FODs were decoded with
  direction-dependent amplitude errors of roughly −20% to +34% (z unaffected).
  Pass `legacy=True` to reproduce results from earlier versions.

### Fixed
- `solve_fsc_voxel`: the convergence check is now relative to the injected
  current and no longer rejects correct solutions when conductances are large.

### Added
- `solve_from_parcellations(..., method="joint")`: the joint MNA solve of the
  Network Neuroscience paper (Eq. 11) for FC pairs without cycles. Cyclic (dense)
  FC raises an explanatory error; the default `method="pairwise"` is unchanged.
- `streamline_support()`: maps the support matrix onto individual streamlines,
  e.g. MRtrix3 `tck2connectome -out_assignments`, for SC from any source.
- Streamline functions are exported from `fsc.parcellation`.

### Documentation
- README: input conventions, graph-construction defaults, and how the code
  relates to the two publications.
