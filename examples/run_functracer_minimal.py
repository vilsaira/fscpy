#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal FunCTracer example. Edit paths before running."""

from pathlib import Path
import numpy as np
import nibabel as nib
from fsc.voxel import FunCTracer

fod_path = Path("subject_wmfod.nii.gz")
roi_label_path = Path("subject_parcellation.nii.gz")
fc_path = Path("subject_fc_matrix.npy")
out_dir = Path("functracer_outputs")
out_dir.mkdir(exist_ok=True)

fod_img = nib.load(str(fod_path))
roi_img = nib.load(str(roi_label_path))
fc = np.load(fc_path)

tracer = FunCTracer.from_images(
    fod_img=fod_img,
    roi_label_img=roi_img,
    fc_matrix=fc,
    fod_norm_threshold=0.25,
    interface_conductance="median",
    normalize_total_interface_conductance=True,
    dilation_iterations=1,
    constraint_mode="fsc",
    verbose=True,
)

result = tracer.solve()
nib.save(nib.Nifti1Image(result.voxel_potential_volume(), fod_img.affine), str(out_dir / "functracer_phi.nii.gz"))
nib.save(nib.Nifti1Image(result.voxel_current_magnitude_volume(), fod_img.affine), str(out_dir / "functracer_current_magnitude.nii.gz"))
np.save(out_dir / "functracer_effective_conductance.npy", result.effective_conductance_matrix())
print("Saved outputs to", out_dir)
