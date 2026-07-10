# Full Paper Workflow Commands

This repo now separates the paper workflow into three train/inference levels.

## 1. C-UNSURE observation-noise training

Already completed if this checkpoint exists:

```bash
runs/selected/cunsure_best_epoch10.pt
```

Re-run only when preprocessing, image normalization, or dataset domain changes.

```bash
python scripts/prepare_hdf5.py --config configs/prepare_hdf5.yaml
python scripts/train_cunsure.py --config configs/train_cunsure.yaml
```

## 2. CineMA latent observation covariance

Use the frozen CineMA encoder and C-UNSURE eta to produce:

```bash
runs/selected/latent_observations_cinema_cunsure_mc.h5
```

Typical commands:

```bash
python scripts/infer_foundation_jacobian_batch.py \
  --config configs/infer_cinema_mc_covariance_all_datasets.yaml

python scripts/package_latent_observations.py \
  --input-dir runs/selected/cinema_mc_all_datasets \
  --output runs/selected/latent_observations_cinema_cunsure_mc.h5

python scripts/export_sde_sequence_index.py \
  --h5 runs/selected/latent_observations_cinema_cunsure_mc.h5 \
  --output runs/selected/sde_sequence_index.jsonl
```

## 3. NODEO-style Neural SDE/CVRNN deformation training

This is the main deformation training stage of the paper.

```bash
python scripts/train_nodeo_sde_deformation.py \
  --config configs/train_nodeo_sde_deformation.yaml
```

The loss follows NODEO:

```text
L_NODEO = L_img + lambda_J L_Jdet + lambda_v L_mag + lambda_df L_smt
```

CineMA/C-UNSURE provide observed latent states and covariance for CVRNN update; they are not extra registration losses.

## 4. Deformation inference

```bash
python scripts/infer_nodeo_sde_deformation.py \
  --checkpoint runs/nodeo_sde_deformation/best.pt \
  --h5 runs/selected/latent_observations_cinema_cunsure_mc.h5 \
  --output runs/nodeo_sde_deformation/deformation_sequence0.pt \
  --sequence-index 0 \
  --covariance diag \
  --device auto
```

Output contains:

```text
displacement [T,3,D,H,W]
phi [T,3,D,H,W]
warped [T,1,D,H,W]
deformation_covariance_diag [T,3,D,H,W] when enabled
```

## 5. Clinical metrics and uncertainty

Requires a reference segmentation mask, typically ED.

```bash
python scripts/compute_clinical_metrics.py \
  --deformation runs/nodeo_sde_deformation/deformation_sequence0.pt \
  --reference-mask path/to/reference_ed_mask.nii.gz \
  --output runs/nodeo_sde_deformation/clinical_metrics_sequence0.json \
  --labels 1 \
  --ed-index 0 \
  --es-index -1
```

Output contains:

```text
volume curve + variance
EF + variance
mean wall motion + variance
mean Green-Lagrange strain + variance
```
