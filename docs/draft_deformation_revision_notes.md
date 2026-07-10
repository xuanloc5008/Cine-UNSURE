# Draft Revision Notes: Deformation Field Computation and Training

## Problem in the current draft

The current draft defines CineMA observations, C-UNSURE observation covariance, SDE/CVRNN propagation, and output uncertainty. However, the deformation field itself is under-specified:

- The state variable alternates between dense deformation `phi` and hidden state `h` without a clear bridge.
- The image warping convention is not defined.
- The training objective does not explain how `c(h)` learns a meaningful deformation field.
- Registration losses and deformation regularizers are missing.
- Section 3.5.3 states `R_k = Delta_h c P_k Delta_h c^T`, but does not specify the practical diagonal/block-diagonal computation.

## Files added

- `docs/draft_deformation_field_training_insert.tex`

This is a LaTeX-ready insert for the paper.

## Suggested edits

1. Replace the current Section 3.3 with the new `Neural SDE for Cardiac Deformation`.
2. Replace the current Section 3.4 with the clarified `CVRNN Update at Observed Frames`.
3. Replace or expand Section 3.5 with the clarified `Analytical Uncertainty Propagation`.
4. Insert the new `Training Objective for Deformation Fields` before Section 4.
5. Update Algorithm 1 so that output includes:
   - deformation mean `phi_hat_k`
   - diagonal or block deformation covariance `R_k`
   - warped image `I_hat_k`
   - clinical metric uncertainty.

## Corrected workflow after revision

1. Freeze CineMA and C-UNSURE.
2. Encode every cine frame:
   `z_k = E_CineMA(I_k)`.
3. Estimate observation covariance:
   `Sigma_k = J_E Sigma_img J_E^T` using MC finite differences.
4. Train SDE-RNN:
   - propagate hidden state by neural SDE,
   - update hidden state by CVRNN at observed frames,
   - decode dense deformation `phi_k = Id + c(h_k)`,
   - warp reference frame to target frame,
   - optimize image registration loss + latent NLL + deformation regularizers.
5. Inference returns:
   - deformation field `phi_hat_k`,
   - deformation covariance `R_k` in diagonal/block form,
   - latent innovation covariance `S_k`,
   - clinical metric uncertainty via delta method.

## Key theoretical correction

The deformation decoder `c(h)` cannot learn meaningful deformation from latent NLL alone. It must be trained through image-space registration losses and deformation regularization:

```text
L_total =
  lambda_img  L_img
+ lambda_lat  L_lat
+ lambda_seg  L_seg
+ lambda_Jdet L_Jdet
+ lambda_mag  L_mag
+ lambda_smt  L_smt
+ lambda_temp L_temp
```

This is the missing bridge between the draft's uncertainty theory and an actual learned cardiac deformation field.
