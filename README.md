# Bidirectional NODEO-DIR + decomposed ambiguity + post-hoc SDE-CVGRU

The active workflow estimates a NODEO deformation trajectory for each cropped
4D cine CMR sequence and derives deformation uncertainty directly from the
registration ambiguity map `U_ambiguity`.

```text
cropped ROI cine sequence
  -> sequential NODEO-DIR with configurable ODE integration
  -> mean deformation phi_bar[k]
  -> forward and inverse deformation with bidirectional consistency
  -> local affine intensity correction and structural evidence maps
  -> separate image-quality and deformation-ambiguity maps
  -> per-sequence masked + full-trajectory MSE fit of SDE-CVGRU mean dynamics
  -> frozen-network analytical Jacobian propagation
  -> ambiguity + SDE process + NODEO model covariance
  -> voxel-wise deformation variance and uncalibrated EF prediction band
```

No external image-uncertainty encoder, latent packaging stage, or observation
checkpoint is used.

## Uncertainty definition

The raw photometric mismatch is not used as deformation uncertainty. The
pipeline first fits a local affine intensity model, then combines normalized
structural, gradient-orientation, inverse-consistency, and Jacobian-violation
evidence:

```text
I_corrected = gain * warp(I[0], phi_bar[k]) + bias
U_deformation = 0.4 U_structural + 0.2 U_gradient
              + 0.3 U_inverse    + 0.1 U_jacobian
U_image       = 0.5 U_intensity  + 0.5 U_artifact
```

`U_deformation[0]` is zero. Each evidence family uses a robust sequence-level
scale. Only `U_deformation` is used as an isotropic
voxel-space covariance proxy and projected into the NODEO motion basis. The
SDE-CVGRU mean is fitted with masked-frame prediction MSE plus full-trajectory
reconstruction MSE. Frozen-network analytical Jacobians then propagate two
dynamical streams, while a third NODEO term is estimated from late optimization
checkpoints:

```text
R_phi_total[k] = R_phi_ambiguity[k]
               + R_phi_process[k]
               + R_phi_NODEO-model[k]
```

The final mean deformation remains exactly the NODEO result. NODEO uses a
periodic time encoding, forward and backward image losses, inverse consistency,
a two-sided Jacobian determinant penalty, and a soft end-of-cycle closure loss.
`R_phi_NODEO-model` is a practical late-checkpoint
ensemble proxy for local optimization/model variability. It is not an exact
Bayesian parameter posterior, scanner-noise covariance, or a calibrated
coverage guarantee.

## Setup

```bash
python -m pip install -r requirements.txt
```

No external foundation-model installation or image-noise checkpoint is required.

## Full ACDC workflow

Prepare ROI H5 files and the sequence manifest:

```bash
./run_acdc_workflow.sh prepare
./run_acdc_workflow.sh manifest
```

Run NODEO for all splits:

```bash
./run_acdc_workflow.sh nodeo-all
```

The default profile uses RK4 with `step_size: 0.05`. Euler and adaptive
Dopri5 profiles are also available; all profiles apply Gaussian velocity
smoothing at every ODE evaluation.

Run patient-specific post-hoc SDE-CVGRU uncertainty propagation:

```bash
./run_sde_sequence_posthoc_workflow.sh all
```

The same preprocessing, NODEO, and post-hoc SDE sequence is available as one
command:

```bash
./run_acdc_workflow.sh full
```

This bulk command stops after producing per-sequence deformation uncertainty.
Use the single-patient workflow below to derive EF and its analytical band.

## Single patient

Select the NODEO solver in `configs/acdc/patient_sequence_workflow.yaml`:

```yaml
nodeo:
  solver: rk4  # euler, rk4, or dopri5
```

The selected profile controls both NODEO and the matching post-hoc SDE output.
For a one-command override without editing YAML, set `SOLVER`, for example
`SOLVER=dopri5`.

Option 1 fits NODEO for the requested patient before uncertainty propagation:

```bash
OPTION=1 PATIENT=patient101 SPLIT=test ./run_patient_sequence_workflow.sh
```

For example, run Option 1 with Dopri5:

```bash
SOLVER=dopri5 OPTION=1 PATIENT=patient101 SPLIT=test ./run_patient_sequence_workflow.sh
```

Option 2, the default, reuses the configured precomputed NODEO result:

```bash
PATIENT=patient101 SPLIT=test ./run_patient_sequence_workflow.sh
```

No score checkpoint argument is accepted or needed.

Existing NODEO files from the former one-way workflow do not contain
`inverse_displacement`. Re-run NODEO before SDE propagation. Option 1 detects
and refits such stale patient outputs automatically; Option 2 reports a clear
error instead of silently using incompatible uncertainty inputs.

## Main outputs

Each SDE result stores the NODEO mean and an exact diagonal decomposition:

```text
mean_deformation
total_displacement
predicted_frames
residual_squared
image_quality_map
deformation_ambiguity_map
intensity_change_map
artifact_residual_map
structural_residual_map
gradient_residual_map
inverse_consistency_map
jacobian_violation_map
ambiguity_map
deformation_variance_diag
ambiguity_deformation_variance_diag
process_deformation_variance_diag
nodeo_model_deformation_variance_diag
ambiguity_frame_scale
ambiguity_sequence_scale
motion_covariance_factor
ambiguity_motion_covariance_factor
process_motion_covariance_factor
```

The single-patient workflow also writes `ambiguity_ed_es_validation.json`,
which compares ED-to-ES label-propagation surface errors with uncertainty using
Spearman correlation, error-detection ROC-AUC, and sparsification error (AUSE).

For the patient workflow, `ef_prediction_band.json` contains EF, propagated
variance, standard error, and the analytical interval. Validate its empirical
coverage on an independent labelled cohort before interpreting it as a 95%
prediction interval.

See:

- `docs/patient_sequence_two_option_workflow.md`
- `docs/per_sequence_posthoc_sde_workflow.md`
