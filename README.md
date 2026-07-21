# NODEO-DIR + residual ambiguity + post-hoc SDE-CVGRU

The active workflow estimates a NODEO deformation trajectory for each cropped
4D cine CMR sequence and derives deformation uncertainty directly from the
registration ambiguity map `U_ambiguity`.

```text
cropped ROI cine sequence
  -> sequential NODEO-DIR
  -> mean deformation phi_bar[k]
  -> predicted frame warp(I[0], phi_bar[k])
  -> U_ambiguity[k] from the smoothed squared registration residual
  -> per-sequence MSE fit of SDE-CVGRU mean dynamics
  -> frozen-network analytical Jacobian propagation
  -> ambiguity covariance + SDE process covariance
  -> voxel-wise deformation variance and uncalibrated EF prediction band
```

No external image-uncertainty encoder, latent packaging stage, or observation
checkpoint is used.

## Uncertainty definition

For frame `k`:

```text
r[k] = (I[k] - warp(I[0], phi_bar[k]))^2
U_ambiguity[k] = normalize_per_frame(GaussianSmooth(r[k]))
```

`U_ambiguity[0]` is zero. The map is used directly as an isotropic voxel-space
covariance proxy and projected into the NODEO motion basis. After the
per-sequence SDE-CVGRU has been fitted only with masked-frame MSE, analytical
Jacobians propagate two separate streams:

```text
R_phi_total[k] = R_phi_ambiguity[k] + R_phi_process[k]
```

The final mean deformation remains exactly the NODEO result. The uncertainty
is registration ambiguity plus model-dynamics process uncertainty; it is not
scanner-noise covariance, parameter-posterior uncertainty, or a calibrated
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

Option 1 fits NODEO for the requested patient before uncertainty propagation:

```bash
OPTION=1 PATIENT=patient101 SPLIT=test ./run_patient_sequence_workflow.sh
```

Option 2, the default, reuses the configured precomputed NODEO result:

```bash
PATIENT=patient101 SPLIT=test ./run_patient_sequence_workflow.sh
```

No score checkpoint argument is accepted or needed.

## Main outputs

Each SDE result stores the NODEO mean and an exact diagonal decomposition:

```text
mean_deformation
total_displacement
predicted_frames
residual_squared
ambiguity_map
deformation_variance_diag
ambiguity_deformation_variance_diag
process_deformation_variance_diag
motion_covariance_factor
ambiguity_motion_covariance_factor
process_motion_covariance_factor
```

For the patient workflow, `ef_prediction_band.json` contains EF, propagated
variance, standard error, and the analytical interval. Validate its empirical
coverage on an independent labelled cohort before interpreting it as a 95%
prediction interval.

See:

- `docs/patient_sequence_two_option_workflow.md`
- `docs/per_sequence_posthoc_sde_workflow.md`
