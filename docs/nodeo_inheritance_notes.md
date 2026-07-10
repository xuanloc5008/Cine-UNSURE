# NODEO-DIR Inheritance Notes

Local reference repository:

```text
/Users/xuanloc/Documents/NODE/NODEO-DIR
```

The downstream deformation code should inherit NODEO-DIR's registration semantics rather than introducing a new registration objective.

## Components inherited

- `LocalNCC3D`: local normalized cross-correlation image similarity, corresponding to NODEO's `NCC`.
- `SpatialTransformer3D`: voxel-flow warping through `grid_sample`, corresponding to NODEO's `SpatialTransformer`.
- `negative_jacobian_loss`: fold prevention from finite-difference Jacobian determinant.
- `smoothness_loss`: spatial smoothness penalty on the displacement field.
- `velocity_magnitude_loss`: velocity/displacement magnitude regularization.
- `euler_integrate` and `rk4_integrate`: NODEO-style ODE integration choices.

These are implemented in:

```text
src/cunsure_monai3d/nodeo_ops.py
```

## Loss to use for deformation training

The main deformation objective should stay close to NODEO:

```text
L_NODEO = L_img + lambda_J L_Jdet + lambda_v L_mag + lambda_df L_smt
```

where:

```text
L_img  = 1 - LNCC(W(I0, phi_k), I_k)
L_Jdet = negative Jacobian determinant penalty
L_mag  = velocity/displacement magnitude penalty
L_smt  = spatial smoothness penalty
```

CineMA and C-UNSURE should be used for observed latent states and uncertainty propagation, not as extra primary registration losses.
