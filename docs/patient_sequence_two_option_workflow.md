# Patient-configurable two-option workflow

The workflow selects exactly one cine patient from
`processed/acdc/nodeo_roi_splits.jsonl` and executes all remaining stages in a
single command.

## Shared stages

Both options perform the following operations for the selected patient:

1. Read the complete cropped ROI cine sequence from its H5 split.
2. Run the trained score C-UNSURE checkpoint on every frame.
3. Run CineMA and propagate C-UNSURE image uncertainty into latent covariance.
4. Package `z[k]` and `Sigma_z[k]` into a patient-only H5 file.
5. Fit the patient-specific mean SDE-CVGRU using masked-frame MSE against NODEO
   motion codes.
6. Freeze the model and propagate analytical covariance with Jacobians.
7. Save mean deformation, predicted cine frames, and voxel/frame uncertainty.
8. Read ACDC ED/ES indices from `Info.cfg` and produce the uncalibrated EF
   prediction band.

By default `observation.resume: false`, so C-UNSURE and CineMA are executed
again on every invocation. Set it to `true` only when intentionally resuming a
partially completed patient inference.

## Option 1

NODEO is fitted for the selected cine sequence before the shared stages. Its
configuration is selected through `nodeo.run_config`.

```bash
OPTION=1 PATIENT=patient101 SPLIT=test ./run_patient_sequence_workflow.sh
```

Set `OVERWRITE_NODEO=1` to refit an existing on-demand NODEO result.

## Option 2 (default)

The existing NODEO result is selected from:

```text
runs/acdc/nodeo_dir_euler/test
```

Then C-UNSURE/CineMA and the SDE-CVGRU stages are executed for that same
patient:

```bash
PATIENT=patient101 SPLIT=test ./run_patient_sequence_workflow.sh
```

The same values can be stored directly in
`configs/acdc/patient_sequence_workflow.yaml`.

## Outputs

For `patient101`, outputs are stored under:

```text
runs/acdc/patient_sequence_workflow/test/patient101/
```

The directory contains:

```text
observation_frames/          per-frame C-UNSURE/CineMA outputs
latent_observations.h5       patient-only z[k] and Sigma_z[k]
selected_nodeo_summary.jsonl selected or newly fitted NODEO result
sde/test/*.pt                mean deformation and analytical uncertainty
ef_prediction_band.json      EF, variance, standard error, lower and upper band
workflow_result.json         paths and final EF summary
```

The final `.pt` stores compact low-rank covariance and direct voxel variance:

```text
mean_deformation
total_displacement
predicted_frames
deformation_variance_diag
motion_basis
motion_covariance_factor
hidden_mean
hidden_covariance
process_covariance
```
