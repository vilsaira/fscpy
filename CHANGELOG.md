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
