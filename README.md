# Score C-UNSURE + CineMA + NODEO-DIR + Neural SDE-RNN

This repository uses the current workflow:

```text
ROI cine H5
  -> Score C-UNSURE per-frame image uncertainty
  -> CineMA latent observation covariance
  -> packaged latent H5
ROI cine H5 -> independent per-sequence NODEO-DIR optimization
  -> NODEO mean trajectories phi_bar
  -> Neural SDE-RNN uncertainty propagation
  -> deformation inference, deformation covariance, and clinical metrics
  -> uncertainty-aware ACDC pathology probabilities (NOR/DCM/HCM/MINF/RV)
```

NODEO-DIR is deliberately independent of C-UNSURE and CineMA. It reads only
`y`, `source_path`, and `time_index` from the pre-cropped ROI H5 files and
optimizes a fresh velocity network for every cine sequence, matching the
optimization-based formulation of the original NODEO repository. The Neural
SDE-RNN branch follows the mean/covariance propagation
formulation: SDE propagation between observed frames, CVRNN update at observed
frames using CineMA latent covariance, and output covariance
`R_phi = J_c P J_c^T` around the NODEO mean trajectory.

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

# Controlled evaluation following the UNSURE paper metrics/protocol.
./run_acdc_workflow.sh evaluate-score

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

python scripts/build_nodeo_roi_splits.py \
  --config configs/nodeo_roi_splits.yaml

python scripts/run_nodeo_dir.py \
  --config configs/train_nodeo_dir_roi.yaml \
  --split train

python scripts/run_nodeo_dir.py \
  --config configs/train_nodeo_dir_roi.yaml \
  --split val

python scripts/run_nodeo_dir.py \
  --config configs/train_nodeo_dir_roi.yaml \
  --split test

python scripts/summarize_nodeo_dir.py \
  --summary runs/nodeo_dir_roi/test/summary.jsonl \
  --output runs/nodeo_dir_roi/test/metrics_summary.json

python scripts/export_sde_sequence_index.py \
  --h5 runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --output runs/selected/sde_sequence_index_roi.jsonl \
  --min-length 2 \
  --split-manifest processed/nodeo_roi_splits.jsonl

python scripts/train_sde_rnn_uncertainty.py \
  --config configs/train_sde_rnn_uncertainty.yaml

python scripts/infer_sde_rnn_uncertainty.py \
  --checkpoint runs/sde_rnn_uncertainty_roi/best.pt \
  --h5 runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --output runs/sde_rnn_uncertainty_roi/sde_rnn_uncertainty_sequence0.pt \
  --sequence-index 0 \
  --device auto

./run_acdc_clinical_evaluation.sh
./run_acdc_pathology.sh all
```

The `evaluate-score` stage applies the Theorem 3 estimator
`f(y) = y + Sigma_eta s_theta(y)` to held-out frames with known synthetic
Gaussian noise. It reports test PSNR mean/std, estimated versus injected noise
variance, and covariance-kernel ablations (`1`, `3`, `5`) under
`runs/acdc/cunsure_score/unsure_protocol_evaluation/`.

These cine frames are held-out references, not physically noise-free MRI.
Exact reproduction of the paper's FastMRI Table 4 additionally requires raw
k-space, 2x undersampling, and the EI reconstruction objective.

The pathology classifier is trained only from predicted clinical trajectories
and their propagated standard errors. It uses ACDC `Group` metadata as the
supervised target, selects its checkpoint on the validation split, and reports
accuracy, balanced accuracy, macro-F1, NLL, Brier score, and pathology
probability bands on the independent test split. These probability bands cover
uncertainty propagated from the clinical metrics; they are not self-calibrated
and do not represent all possible diagnostic uncertainty.

The same workflow can be run through:

```bash
./run_nodeo_dir_workflow.sh all
./run_deformation_workflow.sh full
```

For deformation-only runs after latent packaging:

```bash
./run_deformation_workflow.sh index
./run_deformation_workflow.sh verify
./run_nodeo_dir_workflow.sh split
./run_nodeo_dir_workflow.sh train
./run_nodeo_dir_workflow.sh val
./run_nodeo_dir_workflow.sh test
./run_deformation_workflow.sh train-uncertainty
./run_deformation_workflow.sh infer-uncertainty
```

See [docs/score_cunsure_workflow.md](docs/score_cunsure_workflow.md) for the
Score C-UNSURE module notes.
