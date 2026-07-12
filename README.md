# Score C-UNSURE + CineMA + NODEO Mean Deformation

This repository uses the current workflow:

```text
ROI cine H5
  -> Score C-UNSURE per-frame image uncertainty
  -> CineMA latent observation covariance
  -> packaged latent H5
  -> NODEO mean deformation training
  -> deformation inference and clinical metrics
```

The deformation mean is learned with a NODEO-style velocity field over the full
voxel grid. The SDE-RNN deformation decoder baseline has been removed from the
active workflow.

## Setup

```bash
python -m pip install -r requirements.txt
```

Install CineMA at the relative path configured under `foundation.repo_path`,
normally `../CineMA`.

## Main Workflow

```bash
python scripts/verify_roi_masks.py --config configs/prepare_hdf5.yaml --max-missing 50
python scripts/prepare_hdf5.py --config configs/prepare_hdf5.yaml
python scripts/verify_hdf5_splits.py --config configs/prepare_hdf5.yaml

python scripts/train_cunsure_score.py --config configs/train_cunsure_score.yaml

python scripts/verify_score_cunsure_frames.py \
  --checkpoint runs/cunsure_score_monai3d_roi/best.pt \
  --h5 processed/val_cunsure_roi.h5 \
  --num-frames 32 \
  --device cuda

python scripts/infer_score_cunsure_cinema_batch.py \
  --config configs/infer_cinema_score_cunsure_all_datasets.yaml

python scripts/package_latent_observations.py \
  --input-dir runs/selected/cinema_score_cunsure_roi_all_datasets \
  --output runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --compression lzf

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

The same workflow can be run through:

```bash
./run_deformation_workflow.sh full
```

For deformation-only runs after latent packaging:

```bash
./run_deformation_workflow.sh index
./run_deformation_workflow.sh verify
./run_deformation_workflow.sh train-mean
./run_deformation_workflow.sh infer-mean
```

See [docs/score_cunsure_workflow.md](docs/score_cunsure_workflow.md) for the
Score C-UNSURE module notes.
