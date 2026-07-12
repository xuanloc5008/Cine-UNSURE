# Current Workflow Commands

This repo now uses Score C-UNSURE for frame-wise observation uncertainty and a
separate NODEO mean branch for deformation learning.

## 1. ROI H5 preprocessing

```bash
python scripts/verify_roi_masks.py --config configs/prepare_hdf5.yaml --max-missing 50
python scripts/prepare_hdf5.py --config configs/prepare_hdf5.yaml
python scripts/verify_hdf5_splits.py --config configs/prepare_hdf5.yaml
```

## 2. Score C-UNSURE

```bash
python scripts/train_cunsure_score.py --config configs/train_cunsure_score.yaml
python scripts/verify_score_cunsure_frames.py \
  --checkpoint runs/cunsure_score_monai3d_roi/best.pt \
  --h5 processed/val_cunsure_roi.h5 \
  --num-frames 32 \
  --device cuda
```

## 3. CineMA latent covariance

```bash
python scripts/infer_score_cunsure_cinema_batch.py \
  --config configs/infer_cinema_score_cunsure_all_datasets.yaml

python scripts/package_latent_observations.py \
  --input-dir runs/selected/cinema_score_cunsure_roi_all_datasets \
  --output runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --compression lzf

python scripts/inspect_latent_observations.py \
  --h5 runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --min-length 2 \
  --random-checks 10
```

## 4. NODEO mean deformation

```bash
python scripts/export_sde_sequence_index.py \
  --h5 runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --output runs/selected/sde_sequence_index_roi.jsonl \
  --min-length 2 \
  --val-fraction 0.1 \
  --test-fraction 0.1

python scripts/verify_deformation_training_inputs.py \
  --config configs/train_nodeo_mean_deformation.yaml \
  --num-sequences 5 \
  --random

python scripts/train_nodeo_mean_deformation.py \
  --config configs/train_nodeo_mean_deformation.yaml

python scripts/infer_nodeo_mean_deformation.py \
  --checkpoint runs/nodeo_mean_deformation_roi/best.pt \
  --h5 runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --output runs/nodeo_mean_deformation_roi/mean_deformation_sequence0.pt \
  --sequence-index 0 \
  --covariance diag \
  --device auto
```

## 5. Optional clinical metrics

```bash
python scripts/compute_clinical_metrics.py \
  --deformation runs/nodeo_mean_deformation_roi/mean_deformation_sequence0.pt \
  --reference-mask path/to/reference_ed_mask.nii.gz \
  --output runs/nodeo_mean_deformation_roi/clinical_metrics_sequence0.json \
  --labels 1 \
  --ed-index 0 \
  --es-index -1 \
  --volume-size 16 96 96
```
