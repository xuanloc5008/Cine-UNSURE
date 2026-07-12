# Score C-UNSURE workflow

The primary pipeline now follows the score-based route in the public UNSURE
implementation.

The AR-DAE objective and annealing below are direct adaptations of the public
code. Frame-wise correlated `eta_k` is the C-UNSURE extension used by this
project. The public fastMRI experiment also adds an MRI-physics equivariance
loss; that term is intentionally absent here because these inputs are already
reconstructed cine volumes and no k-space forward operator is available.

## 1. Prepare real noisy ROI frames

```bash
python scripts/verify_roi_masks.py --config configs/prepare_hdf5.yaml --max-missing 50
python scripts/prepare_hdf5.py --config configs/prepare_hdf5.yaml
python scripts/verify_hdf5_splits.py --config configs/prepare_hdf5.yaml
```

No clean target is stored. Each `y` is a real noisy cardiac MR frame.

## 2. Train the score network

```bash
python scripts/train_cunsure_score.py --config configs/train_cunsure_score.yaml
```

The MONAI 3D U-Net learns `S(y)`, the score of the noisy frame distribution,
using the official AR-DAE objective

`||epsilon + sigma S(y + sigma epsilon)||^2`,

where `sigma ~ N(0, delta^2)` independently per sample and `delta` is annealed
linearly from `0.1` to `0.001`.

Before the expensive CineMA pass, verify that `eta_k` changes across frames:

```bash
python scripts/verify_score_cunsure_frames.py \
  --checkpoint runs/cunsure_score_monai3d_roi/best.pt \
  --h5 processed/val_cunsure_roi.h5 \
  --num-frames 32 --device cuda
```

## 3. Estimate frame-wise noise and propagate through CineMA

```bash
python scripts/infer_score_cunsure_cinema_batch.py \
  --config configs/infer_cinema_score_cunsure_all_datasets.yaml
```

For every frame `k`, the script computes score autocorrelation `h_k`, then
`eta_k = F^-1(1 / F h_k)` with the configured spectral floor. It samples
`n_k ~ N(0, Sigma_img,k)` through the covariance square root and estimates
latent covariance from finite differences of CineMA. It does not materialize a
full CineMA Jacobian. The default validation config reads `processed/val_cunsure_roi.h5`
directly, so inference does not load masks or crop NIfTI volumes again.

Final test inference uses a separate config and output directory:

```bash
python scripts/infer_score_cunsure_cinema_batch.py \
  --config configs/infer_cinema_score_cunsure_test.yaml
```

## 4. Package and train deformation

```bash
python scripts/package_latent_observations.py \
  --input-dir runs/selected/cinema_score_cunsure_roi_all_datasets \
  --output runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --compression lzf

python scripts/inspect_latent_observations.py \
  --h5 runs/selected/latent_observations_cinema_score_cunsure_roi.h5 \
  --min-length 2 --random-checks 10

./run_deformation_workflow.sh index
./run_deformation_workflow.sh verify
./run_deformation_workflow.sh train
./run_deformation_workflow.sh infer
```

The packaged H5 stores `eta[N,1,K,K,K]` and, by default,
`latent_covariance_diag[N,768]`. NODEO mean deformation uses this H5 to recover
sequence metadata and source paths; the mean deformation loss itself is the
image-registration loss.

## Mathematical verification

```bash
python scripts/verify_score_cunsure_math.py
```

This checks the isotropic `eta = 1/E[S(y)^2]` special case, covariance-square-
root sampling, and propagation through a known linear encoder.
