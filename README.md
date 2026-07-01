# Score-Based C-UNSURE Observation Covariance

This implementation turns Section 3.2 of the draft into runnable code for either
2D slices or true 3D cine-MRI frames from a 4D sequence:

1. Train a score network on cine-MRI frames. For 4D cine MRI, each sample is one 3D time frame `I_k`.
2. Compute score autocorrelation `h`.
3. Obtain the C-UNSURE covariance kernel `eta_hat` with FFT.
4. Sample correlated image-space noise probes.
5. Push those probes through a frozen CineMA or MedSAM2 encoder to estimate latent observation covariance.

The external repositories were cloned separately under:

- `work/external/CineMA`
- `work/external/MedSAM2`
- `work/external/unsure`

This package keeps the adapted code separate and imports the cloned encoders through adapter classes.

## Formulas

Observed cine-MRI frame:

```text
I_k = I_k^* + sigma_k * epsilon_k,      epsilon_k ~ N(0, I)
```

Score network:

```text
s_psi(I) ~= grad_I log p_I(I)
```

Train the score model with the AR-DAE / UNSURE score objective:

```text
xi ~ N(0, I)
I_tau = I + tau xi

min_psi E || xi + tau s_psi(I_tau) ||_2^2
```

Score autocorrelation for offsets `Delta` in the correlation window `[-r, r]`.
For a 3D frame `I_k in R^{H x W x D}`:

```text
h_k[Delta_x, Delta_y, Delta_z]
  = (1 / |Omega|) sum_{x,y,z,c}
    s_psi(I_k)[c,x,y,z]
    s_psi(I_k)[c,x + Delta_x, y + Delta_y, z + Delta_z]
```

C-UNSURE closed-form covariance kernel:

```text
eta_hat_k = F^{-1}(1 / (F h_k + eps))
Sigma_img,k = circ(eta_hat_k)
```

Square-root covariance kernel in Fourier domain:

```text
F kappa_k = sqrt(clip(F eta_hat_k, spectral_floor, infinity))
```

Correlated probes:

```text
xi_k^(s) ~ N(0, I)
n_k^(s) = kappa_k * xi_k^(s)
```

Frozen encoder linearization:

```text
z_k = E(I_k)
E(I_k + n_k) ~= E(I_k) + J_E n_k
Sigma_z,k = J_E Sigma_img,k J_E^T
```

Matrix-free finite-difference estimate:

```text
delta z_k^(s) = [E(I_k + tau_fd n_k^(s)) - E(I_k)] / tau_fd

Sigma_hat_z,k =
  1 / S sum_s delta z_k^(s) (delta z_k^(s))^T
```

The output `Sigma_hat_z,k` replaces the scalar heuristic `sigma_obs^2 I` in the CV-GRU update and the innovation covariance.

## Install

From this folder:

```bash
python -m pip install -e .
```

For actual CineMA/MedSAM2 weights, install their dependencies as needed:

```bash
python -m pip install -e ".[cinema]"
python -m pip install -e ".[medsam2]"
python -m pip install -e ".[medical-io]"  # nibabel/SimpleITK for .nii/.nii.gz
```

The adapters import from the cloned repos in `work/external`.

## Download Foundation Checkpoints

Download/cache all foundation-model checkpoints used by the adapters:

```bash
python scripts/download_checkpoints.py
```

Preview without downloading large files:

```bash
python scripts/download_checkpoints.py --dry-run
```

Download only one group:

```bash
python scripts/download_checkpoints.py --only cinema
python scripts/download_checkpoints.py --only medsam2
python scripts/download_checkpoints.py --only efficienttam sam2-base
```

By default, MedSAM2/EfficientTAM/SAM2 files are saved under:

```text
work/external/MedSAM2/checkpoints
```

CineMA files are cached by HuggingFace Hub. You can override locations:

```bash
python scripts/download_checkpoints.py \
  --cinema-cache-dir /path/to/hf_cache \
  --medsam2-checkpoint-dir /path/to/medsam2_checkpoints
```

The script writes:

```text
work/external/MedSAM2/checkpoints/download_manifest.json
```

For MedSAM2, the upstream script is still available:

```bash
cd work/external/MedSAM2
bash download.sh
```

## Train Score Network On 3D Cine Frames

For your ACDC/M&M-style NIfTI folders, point `--data` to one or more dataset roots and filter image files. Based on the folder layout shown in Finder, the clean training roots are:

```bash
export ACDC_TRAIN="/Volumes/Transcend/ACDC/database/training"
export MM1_TRAIN="/Volumes/Transcend/M&M1/Training"
export MNM2_ROOT="/Volumes/Transcend/MnM2/dataset"
export SCORE_CKPT="outputs/score_unet_cine3d.pt"

python scripts/train_score.py \
  --data "$ACDC_TRAIN" "$MM1_TRAIN" "$MNM2_ROOT" \
  --output "$SCORE_CKPT" \
  --channels 1 \
  --spatial-dims 3 \
  --time-axis 0 \
  --frame-layout dhw \
  --include "*_4d.nii.gz" "*_sa.nii.gz" "*_SA_CINE.nii.gz" \
  --exclude "*_gt.nii.gz" "*_ED.nii.gz" "*_ES.nii.gz" "*_LA_*.nii.gz" \
  --image-size 192 \
  --depth-size 16 \
  --epochs 50 \
  --batch-size 1 \
  --base-channels 32 \
  --depth 3 \
  --log-every 100
```

The filters match the structures:

```text
ACDC: patientXXX_4d.nii.gz
M&M1: ID_sa.nii.gz under Training/Labeled or Training/Unlabeled
MnM2: ID_SA_CINE.nii.gz
```

and skip masks/static labels:

```text
*_gt.nii.gz
*_ED.nii.gz
*_ES.nii.gz
*_LA_*.nii.gz
```

## Optional: Preprocess Frames First

For faster repeated training, cache normalized/resized 3D frames once:

```bash
export PREPROC_DIR="outputs/preprocessed_cine3d_192x192x16"

python scripts/preprocess_cine_frames.py \
  --data "$ACDC_TRAIN" "$MM1_TRAIN" "$MNM2_ROOT" \
  --output "$PREPROC_DIR" \
  --channels 1 \
  --time-axis 0 \
  --frame-layout dhw \
  --include "*_4d.nii.gz" "*_sa.nii.gz" "*_SA_CINE.nii.gz" \
  --exclude "._*" "*_gt.nii.gz" "*_ED.nii.gz" "*_ES.nii.gz" "*_LA_*.nii.gz" \
  --image-size 192 \
  --depth-size 16 \
  --dtype float16 \
  --log-every 500
```

Then train directly from cached `.pt` frames:

```bash
python scripts/train_score.py \
  --data "$PREPROC_DIR" \
  --preprocessed \
  --output "$SCORE_CKPT" \
  --channels 1 \
  --spatial-dims 3 \
  --include "*.pt" \
  --exclude "manifest.json" \
  --epochs 10 \
  --batch-size 4 \
  --base-channels 32 \
  --depth 3 \
  --device cuda \
  --multi-gpu \
  --gpu-ids 0,1 \
  --val-fraction 0.05 \
  --augment \
  --log-every 100
```

This writes:

```text
outputs/score_unet_cine3d.pt          # last checkpoint
outputs/score_unet_cine3d.best.pt     # best validation checkpoint
outputs/score_unet_cine3d.metrics.csv # per-epoch train/val loss
```

Resume training from the last checkpoint:

```bash
python scripts/train_score.py \
  --data "$PREPROC_DIR" \
  --preprocessed \
  --output "$SCORE_CKPT" \
  --channels 1 \
  --spatial-dims 3 \
  --include "*.pt" \
  --epochs 20 \
  --batch-size 4 \
  --base-channels 32 \
  --depth 3 \
  --device cuda \
  --multi-gpu \
  --gpu-ids 0,1 \
  --val-fraction 0.05 \
  --augment \
  --resume "$SCORE_CKPT" \
  --log-every 100
```

If you want to reserve M&M1 validation as explicit validation data:

```bash
python scripts/train_score.py \
  --data "$ACDC_TRAIN" "$MM1_TRAIN" "$MNM2_ROOT" \
  --val-data "$MM1_VAL" \
  --output "$SCORE_CKPT" \
  --channels 1 \
  --spatial-dims 3 \
  --time-axis 0 \
  --frame-layout dhw \
  --include "*_4d.nii.gz" "*_sa.nii.gz" "*_SA_CINE.nii.gz" \
  --exclude "._*" "*_gt.nii.gz" "*_ED.nii.gz" "*_ES.nii.gz" "*_LA_*.nii.gz" \
  --image-size 192 \
  --depth-size 16 \
  --epochs 10 \
  --batch-size 4 \
  --base-channels 32 \
  --depth 3 \
  --device cuda \
  --multi-gpu \
  --gpu-ids 0,1 \
  --augment \
  --log-every 100
```

`--batch-size` is the global batch size. With `--multi-gpu --gpu-ids 0,1`, `--batch-size 8` is split roughly as 4 samples per GPU.

For a folder of 4D arrays shaped `[H, W, D, T]`, train on every 3D frame:

```bash
python scripts/train_score.py \
  --data /path/to/4d_cine_arrays \
  --output outputs/score_unet_cine3d.pt \
  --channels 1 \
  --spatial-dims 3 \
  --time-axis -1 \
  --frame-layout hwd \
  --image-size 192 \
  --depth-size 16 \
  --epochs 50 \
  --batch-size 8
```

For NIfTI files normalized internally as `[T, D, H, W]`, use:

```bash
python scripts/train_score.py \
  --data /path/to/cine_4d_nii_folder \
  --output outputs/score_unet_cine3d.pt \
  --spatial-dims 3 \
  --time-axis 0 \
  --frame-layout dhw \
  --image-size 192 \
  --depth-size 16
```

This saves a checkpoint containing the score network weights and loss settings.

## Estimate Covariance For One Encoder

Debug with the lightweight identity encoder:

```bash
python scripts/estimate_covariance.py \
  --image /path/to/frame.npy \
  --score-checkpoint outputs/score_unet_cine.pt \
  --encoder identity \
  --output outputs/debug_covariance.pt \
  --radius 5 \
  --n-probes 32
```

CineMA:

```bash
python scripts/estimate_covariance.py \
  --image /path/to/cine4d.npy \
  --score-checkpoint outputs/score_unet_cine3d.pt \
  --encoder cinema \
  --spatial-dims 3 \
  --time-index 0 \
  --time-axis -1 \
  --frame-layout hwd \
  --cinema-view sax \
  --cinema-pool cls \
  --output outputs/cinema_covariance.pt
```

MedSAM2:

```bash
python scripts/estimate_covariance.py \
  --image /path/to/cine4d.npy \
  --score-checkpoint outputs/score_unet_cine3d.pt \
  --encoder medsam2 \
  --spatial-dims 3 \
  --time-index 0 \
  --time-axis -1 \
  --frame-layout hwd \
  --medsam2-config configs/sam2.1_hiera_t512.yaml \
  --medsam2-checkpoint work/external/MedSAM2/checkpoints/MedSAM2_latest.pt \
  --medsam2-volume-pool mean \
  --output outputs/medsam2_covariance.pt
```

MedSAM2 is image-backbone based in this adapter. For a 3D frame, it encodes all slices as a batch and then pools slice features across depth. CineMA SAX consumes the 3D frame directly as `[B, 1, 192, 192, 16]`.

If you have not downloaded MedSAM2 checkpoints yet:

```bash
cd work/external/MedSAM2
bash download.sh
```

## Compare CineMA And MedSAM2

```bash
python scripts/compare_encoders.py \
  --image /path/to/cine4d.npy \
  --score-checkpoint outputs/score_unet_cine3d.pt \
  --output-dir outputs/foundation_compare \
  --encoders cinema medsam2 \
  --spatial-dims 3 \
  --time-index 0 \
  --time-axis -1 \
  --frame-layout hwd \
  --radius 5 \
  --n-probes 32 \
  --cinema-view sax \
  --cinema-pool cls \
  --medsam2-volume-pool mean \
  --medsam2-checkpoint work/external/MedSAM2/checkpoints/MedSAM2_latest.pt
```

Outputs:

- `cinema_covariance.pt`
- `medsam2_covariance.pt`
- `comparison_metrics.csv`

The CSV reports latent dimension, trace, diagonal statistics, Frobenius norm, approximate rank, and latent norm.

## Verify C-UNSURE Inference

The verification script implements two checks:

1. Synthetic covariance alignment with known injected Gaussian noise.
2. Clinical sensitivity trace monotonicity over raw/light/heavy noisy inputs.

Synthetic test with `sigma = 0.1`:

```bash
python scripts/verify_cunsure.py \
  --mode synthetic \
  --image /path/to/cine4d.npy \
  --score-checkpoint outputs/score_unet_cine3d.pt \
  --encoder cinema \
  --spatial-dims 3 \
  --time-index 0 \
  --time-axis -1 \
  --frame-layout hwd \
  --cinema-view sax \
  --synthetic-sigma 0.1 \
  --mc-samples 256 \
  --cosine-threshold 0.95 \
  --output-dir outputs/verification_cinema
```

It computes:

```text
Sigma_CUNSURE = score-based C-UNSURE covariance
Sigma_MC      = Monte Carlo Cov(E(I + n_s)), n_s ~ N(0, 0.1^2 I)

cos_sim = <vec(Sigma_CUNSURE), vec(Sigma_MC)>
          / (||Sigma_CUNSURE|| ||Sigma_MC||)
```

Pass condition:

```text
cos_sim >= 0.95
```

The script also reports:

```text
relative_trace_error = |tr(Sigma_CUNSURE) - tr(Sigma_MC)| / tr(Sigma_MC)
```

Clinical sensitivity test:

```bash
python scripts/verify_cunsure.py \
  --mode sensitivity \
  --image /path/to/cine4d.npy \
  --score-checkpoint outputs/score_unet_cine3d.pt \
  --encoder cinema \
  --spatial-dims 3 \
  --time-index 0 \
  --time-axis -1 \
  --frame-layout hwd \
  --cinema-view sax \
  --sensitivity-levels 0.0 0.05 0.20 \
  --trials 5 \
  --output-dir outputs/sensitivity_cinema
```

Pass condition:

```text
Trace_raw < Trace_0.05 < Trace_0.20
```

Outputs:

- `verification_summary.json`
- `synthetic_alignment.pt` for covariance tensors
- `sensitivity_traces.csv`

Use `--mode all` to run both tests in one command. For MedSAM2, replace `--encoder cinema --cinema-view sax` with `--encoder medsam2 --medsam2-checkpoint ... --medsam2-volume-pool mean`.

## Notes For The Draft Pipeline

- Keep the score network trained on the same normalized image space used before the encoder adapters.
- For 4D cine MRI, the observation unit is one 3D time frame: `I_k = I[..., k]`, not an independent 2D slice.
- `.npy/.npz` arrays are assumed to be `[H, W, D, T]` when you use `--time-axis -1 --frame-layout hwd`.
- NIfTI reads are normalized internally to `[T, D, H, W]`; use `--time-axis 0 --frame-layout dhw`.
- CineMA SAX uses true 3D input `[B, 1, 192, 192, 16]`.
- MedSAM2 encodes the 3D frame slice-wise and aggregates slice latents with `--medsam2-volume-pool`.
- `n_probes` controls the rank and stability of `Sigma_hat_z`. Use `32` for quick experiments and `64+` for paper-grade estimates.
- Store low-rank deltas if full `d x d` covariance is too large; default C-UNSURE uses the paper form `Sigma = D^T D / S`.
