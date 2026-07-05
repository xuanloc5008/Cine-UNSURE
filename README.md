# MONAI 3D U-Net + Minimax C-UNSURE

This project trains only the C-UNSURE denoiser/noise model:

```text
min_theta max_eta E ||f_theta(y)-y||^2 + 2 tr(Sigma_eta J_f(y))
```

The backbone is a MONAI 3D U-Net. CineMA and MedSAM2 are frozen and used only
after training to compute full Jacobians and push the learned image-space noise
operator into foundation-model latent space.

## Setup

```bash
python -m pip install -r requirements.txt
```

Install CineMA and MedSAM2 as local repos under the relative paths configured in
`configs/infer_*_jacobian.yaml`, for example `../CineMA` and `../MedSAM2`.

## Prepare Real NIfTI Data

Put this project next to `ACDC`, `M&M1`, and `MnM2`, or edit
`configs/prepare_hdf5.yaml` so the relative globs point to those folders. Then run:

```bash
python scripts/prepare_hdf5.py --config configs/prepare_hdf5.yaml
```

This writes HDF5 files with noisy real frames only. No clean target is stored.

## Train C-UNSURE

```bash
python scripts/train_cunsure.py --config configs/train_cunsure.yaml
```

The important output is:

```text
runs/cunsure_monai3d/best.pt
```

It contains the trained MONAI U-Net weights and the learned `eta` kernel.

## Inference With Frozen Foundation Models

CineMA:

```bash
python scripts/infer_foundation_jacobian.py --config configs/infer_cinema_jacobian.yaml
```

MedSAM2:

```bash
python scripts/infer_foundation_jacobian.py --config configs/infer_medsam2_jacobian.yaml
```

Each inference script:

```text
1. loads eta from the C-UNSURE checkpoint
2. loads one real NIfTI frame
3. freezes the foundation model
4. computes the full Jacobian J
5. computes Sigma_z = J Sigma_eta J^T
6. saves foundation_noise_covariance.pt
```

No script trains CineMA or MedSAM2.
