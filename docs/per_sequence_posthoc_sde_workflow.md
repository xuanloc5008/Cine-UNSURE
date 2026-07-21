# Per-sequence MSE + post-hoc analytical SDE-CVGRU

This workflow operates independently on each cine sequence and needs only the
NODEO output: cropped images, time indices, and mean deformation trajectory.

## Ambiguity input

For every frame, the runner first corrects local affine intensity variation and
then computes separate evidence families:

```text
U_image = combine(intensity change, non-edge artifact residual)
U_deformation = combine(local structure, gradient orientation,
                        inverse consistency, Jacobian violation)
```

Frame zero is set to zero ambiguity. The shared sequence scale preserves
between-frame amplitude, unlike independent per-frame max normalization.
Only `U_deformation` is used as an isotropic voxel covariance proxy, repeated
over the three displacement channels and projected into the NODEO motion
basis. `U_image` is saved for diagnostics but does not enter SDE deformation
covariance. No external observation file or learned ambiguity conversion is used.

## Mean fitting

NODEO displacement is represented in a low-rank motion basis. A small
continuous-time hidden model and CVGRU are fitted separately for each sequence.
Random target frames are hidden. Optimization uses their prediction MSE plus a
weighted full-trajectory reconstruction MSE, both against NODEO motion codes.

No covariance, NLL, coverage, clinical target, or segmentation target enters
the fit. The final reported mean remains locked to NODEO; the fitted model is
used for sensitivity and covariance propagation.

## Post-hoc uncertainty

After selecting the lowest held-out-frame MSE, all weights are frozen. The
runner propagates two covariance streams and adds one NODEO model term:

1. Ambiguity covariance from `U_ambiguity`, through CVGRU observation
   Jacobians and the deformation decoder.
2. Process covariance from SDE dynamics, through drift and update Jacobians.
3. NODEO model variance estimated from a late-checkpoint displacement ensemble.

The total is their sum. Compact low-rank factors are stored for the first two
terms, while the NODEO model term is stored as an exact voxel-wise diagonal:

```text
L_phi[k] = motion_basis @ motion_covariance_factor[k]
R_phi[k] = L_phi[k] @ L_phi[k].T
```

Separate ambiguity/process factors and exact voxel-wise diagonal components
are also saved. The late-checkpoint term is a local model-variability proxy,
not an exact Bayesian posterior.

## Run

Run one sequence:

```bash
LIMIT=1 ./run_sde_sequence_posthoc_workflow.sh train
```

Run all splits:

```bash
./run_sde_sequence_posthoc_workflow.sh all
```

Each `.pt` is a patient-specific fitted model and analytical uncertainty
result under `runs/acdc/sde_sequence_posthoc/<split>/`.

## EF prediction band

Use the ACDC LV label, ED mask, and ES index:

```bash
DEFORMATION=runs/acdc/sde_sequence_posthoc/val/000000_<id>.pt \
REFERENCE_MASK=datasets/ACDC/database/training/patient001/patient001_frame01_gt.nii.gz \
ES_INDEX=11 \
EF_OUTPUT=runs/acdc/sde_sequence_posthoc/val/000000_<id>_ef.json \
./run_sde_sequence_posthoc_workflow.sh ef
```

EF variance is obtained by delta propagation from the low-rank deformation
covariance. The Gaussian band is analytical and uncalibrated; it must not be
interpreted as guaranteed 95% empirical coverage without an independent
coverage study.
