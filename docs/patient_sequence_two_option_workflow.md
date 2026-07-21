# Patient-configurable two-option workflow

The workflow selects one cine sequence from
`processed/acdc/nodeo_roi_splits.jsonl`. The ROI H5 is the only image source;
no external uncertainty encoder, latent H5, or observation checkpoint is
required.

If the manifest is absent, the entry point builds it with
`configs/acdc/nodeo_roi_splits.yaml`.

## Shared stages

Both options perform these operations for the selected patient:

1. Load the complete cropped ROI cine sequence.
2. Obtain the NODEO mean deformation `phi_bar[k]` and predicted frames.
3. Compute registration ambiguity directly from the NODEO residual:

   ```text
   residual[k] = (I[k] - warp(I[0], phi_bar[k]))^2
   U_ambiguity[k] = normalize_per_frame(GaussianSmooth(residual[k]))
   ```

4. Fit a patient-specific SDE-CVGRU with masked-frame MSE against low-rank
   NODEO motion codes. No uncertainty term is optimized.
5. Freeze the fitted network and propagate `U_ambiguity` analytically through
   the CVGRU and deformation decoder Jacobians.
6. Add process covariance from the SDE dynamics and retain the NODEO
   deformation as the final mean trajectory.
7. Save predicted frames, voxel-wise deformation variance, and separate
   ambiguity/process covariance components.
8. Read ACDC ED/ES indices and propagate deformation covariance to an
   uncalibrated EF prediction band.

The reported uncertainty is a registration-ambiguity proxy plus SDE process
uncertainty. It is not scanner-noise covariance or a self-calibrated coverage
guarantee.

## Option 1

Fit NODEO for the selected sequence, then run the shared stages:

```bash
OPTION=1 PATIENT=patient101 SPLIT=test ./run_patient_sequence_workflow.sh
```

Set `OVERWRITE_NODEO=1` to refit an existing on-demand result.

## Option 2 (default)

Reuse an existing NODEO output from the directory configured by
`nodeo.precomputed_dir`, then run the same ambiguity and SDE stages:

```bash
PATIENT=patient101 SPLIT=test ./run_patient_sequence_workflow.sh
```

## Outputs

For `patient101`, outputs are stored under:

```text
runs/acdc/patient_sequence_workflow/test/patient101/
```

The directory contains:

```text
selected_nodeo_summary.jsonl selected or newly fitted NODEO result
sde/test/*.pt                mean deformation and analytical uncertainty
ef_prediction_band.json      EF, variance, standard error, and prediction band
workflow_result.json         paths, uncertainty source, and final EF summary
```

The SDE `.pt` stores:

```text
mean_deformation
total_displacement
predicted_frames
residual_squared
ambiguity_map
deformation_variance_diag
ambiguity_deformation_variance_diag
process_deformation_variance_diag
motion_basis
motion_covariance_factor
ambiguity_motion_covariance_factor
process_motion_covariance_factor
hidden_covariance
hidden_ambiguity_covariance
hidden_process_covariance
```

The diagonal decomposition is exact in the saved output:

```text
deformation_variance_diag
  = ambiguity_deformation_variance_diag
  + process_deformation_variance_diag
```
