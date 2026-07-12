# Current Workflow Commands

This repo now uses Score C-UNSURE for frame-wise observation uncertainty, a
separate NODEO mean branch for deformation learning, and a Neural SDE-RNN
branch for analytical uncertainty propagation around the NODEO mean trajectory.

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

## 4. Independent NODEO-DIR mean deformation

This stage does not read latent observations, eta, or covariance. Following the
original optimization-based NODEO method, it fits one velocity network per cine
sequence. The cohort split is used for the experimental protocol and
hyperparameter selection; it is not a global supervised NODEO train/val split.

```bash
./run_nodeo_dir_workflow.sh split
./run_nodeo_dir_workflow.sh train
./run_nodeo_dir_workflow.sh val
./run_nodeo_dir_workflow.sh test
./run_nodeo_dir_workflow.sh summarize-test

python scripts/export_sde_sequence_index.py \
  --h5 runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --output runs/selected/sde_sequence_index_roi.jsonl \
  --min-length 2 \
  --split-manifest processed/nodeo_roi_splits.jsonl
```

## 5. Neural SDE-RNN uncertainty propagation

```bash
python scripts/train_sde_rnn_uncertainty.py \
  --config configs/train_sde_rnn_uncertainty.yaml

python scripts/infer_sde_rnn_uncertainty.py \
  --checkpoint runs/sde_rnn_uncertainty_roi/best.pt \
  --h5 runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --output runs/sde_rnn_uncertainty_roi/sde_rnn_uncertainty_sequence0.pt \
  --sequence-index 0 \
  --device auto
```

The output stores the full hidden covariance `P_k` and an exact low-rank
deformation covariance factor `L_phi`, where `R_phi = L_phi @ L_phi.T`.

## 6. Optional clinical metrics

```bash
python scripts/compute_clinical_metrics.py \
  --deformation runs/sde_rnn_uncertainty_roi/sde_rnn_uncertainty_sequence0.pt \
  --reference-mask path/to/reference_ed_mask.nii.gz \
  --output runs/sde_rnn_uncertainty_roi/clinical_metrics_sequence0.json \
  --labels 1 \
  --ed-index 0 \
  --es-index -1 \
  --volume-size 16 96 96
```
