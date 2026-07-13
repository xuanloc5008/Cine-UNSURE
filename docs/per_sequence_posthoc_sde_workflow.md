# Per-sequence MSE + post-hoc analytical SDE-CVGRU

This workflow operates independently on each cine sequence after the three
upstream inputs already exist:

1. CineMA observations `z[k]`.
2. C-UNSURE-derived latent observation covariance `Sigma_z[k]`.
3. NODEO mean deformation `phi_bar[k]` and its cropped ROI image sequence.

## Mean fitting

NODEO displacement and CineMA observations are represented in separate linear
bases. A small continuous-time hidden model and CVGRU are fitted for one
sequence at a time. Random target frames are hidden from the CVGRU and the only
optimization objective is mean squared error between predicted and NODEO
motion codes at those hidden frames.

Frame zero is constrained to the NODEO reference code during fitting and is
hard-anchored to identity deformation with zero deformation variance in the
saved result.

No covariance, NLL, coverage, or clinical target contributes to this fit.

## Post-hoc uncertainty

After selecting the lowest held-out-frame MSE, all weights are frozen. The
script then computes:

- SDE prediction covariance with the drift Jacobian and fixed `G = I`; dynamic
  `Q` is estimated from hidden innovations after subtracting the covariance
  already explained by observation noise;
- CVGRU observation update covariance with Jacobians with respect to hidden
  state and CineMA observation;
- deformation covariance with the decoder Jacobian.

The full voxel covariance is retained as a low-rank factor:

```text
L_phi[k] = motion_basis @ motion_covariance_factor[k]
R_phi[k] = L_phi[k] @ L_phi[k].T
```

`deformation_variance_diag` is also saved as `[T,3,D,H,W]` for direct
visualization.

## Run

Run one sequence first:

```bash
LIMIT=1 ./run_sde_sequence_posthoc_workflow.sh train
```

Run all available NODEO train and validation sequences:

```bash
./run_sde_sequence_posthoc_workflow.sh train-val
```

Outputs are written under `runs/acdc/sde_sequence_posthoc/<split>/`. Each `.pt`
file is a separate patient-specific model and analytical uncertainty result.

## EF prediction band

Use the LV label (`3` for ACDC), the ED mask, and the ES position in the output
sequence:

```bash
DEFORMATION=runs/acdc/sde_sequence_posthoc/val/000000_<id>.pt \
REFERENCE_MASK=datasets/ACDC/database/training/patient001/patient001_frame01_gt.nii.gz \
ES_INDEX=11 \
EF_OUTPUT=runs/acdc/sde_sequence_posthoc/val/000000_<id>_ef.json \
./run_sde_sequence_posthoc_workflow.sh ef
```

The EF variance is obtained by delta propagation from the low-rank deformation
covariance. The reported Gaussian prediction band is not self-calibrated. ED
and ES deformation errors are currently treated as independent in the EF
delta approximation, and this assumption is recorded in the JSON output.
