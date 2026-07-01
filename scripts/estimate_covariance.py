#!/usr/bin/env python3
"""Estimate score C-UNSURE latent observation covariance for one encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_cunsure.checkpoint import load_score_checkpoint
from score_cunsure.cunsure import CUNSUREConfig, estimate_latent_covariance
from score_cunsure.data import load_frame
from score_cunsure.encoders import build_encoder


def default_external_root() -> Path:
    return Path(__file__).resolve().parents[3] / "work" / "external"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input cine-MRI frame")
    parser.add_argument("--score-checkpoint", required=True, help="ScoreUNet checkpoint from train_score.py")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--encoder", choices=["identity", "cinema", "medsam2"], default="identity")
    parser.add_argument("--external-root", default=str(default_external_root()))
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--spatial-dims", type=int, choices=[2, 3], default=2)
    parser.add_argument("--npz-key", default=None)
    parser.add_argument("--time-index", type=int, default=None, help="Frame index k for 4D cine MRI input")
    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis for 4D cine input when --spatial-dims 3")
    parser.add_argument("--frame-layout", default="auto", help="Frame layout after optional time extraction")
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--n-probes", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1.0e-2)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--spectral-floor", type=float, default=1.0e-8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-deltas", action="store_true")

    parser.add_argument("--identity-dim", type=int, default=16)
    parser.add_argument("--cinema-view", default="lax_4c")
    parser.add_argument("--cinema-pool", default="cls")
    parser.add_argument("--cinema-cache-dir", default=None)
    parser.add_argument("--medsam2-config", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--medsam2-checkpoint", default=None)
    parser.add_argument("--medsam2-pool", default="vision_mean")
    parser.add_argument("--medsam2-volume-pool", default="mean", choices=["mean", "max", "mean_std", "flatten"])
    return parser.parse_args()


def build_requested_encoder(args: argparse.Namespace) -> torch.nn.Module:
    if args.encoder == "identity":
        return build_encoder("identity", out_dim=args.identity_dim)
    if args.encoder == "cinema":
        return build_encoder(
            "cinema",
            external_root=args.external_root,
            view=args.cinema_view,
            pool=args.cinema_pool,
            device=args.device,
            cache_dir=args.cinema_cache_dir,
        )
    return build_encoder(
        "medsam2",
        external_root=args.external_root,
        config_name=args.medsam2_config,
        checkpoint=args.medsam2_checkpoint,
        pool=args.medsam2_pool,
        volume_pool=args.medsam2_volume_pool,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    score_model, score_ckpt = load_score_checkpoint(args.score_checkpoint, device=device)
    image = load_frame(
        args.image,
        npz_key=args.npz_key,
        channels=args.channels,
        normalize=True,
        spatial_dims=args.spatial_dims,
        frame_index=args.time_index,
        time_axis=args.time_axis if args.time_index is not None else None,
        frame_layout=args.frame_layout,
    )
    image = image.unsqueeze(0).to(device)
    encoder = build_requested_encoder(args).to(device).eval()

    config = CUNSUREConfig(
        radius=args.radius,
        eps=args.eps,
        spectral_floor=args.spectral_floor,
        n_probes=args.n_probes,
        finite_difference_tau=args.tau,
    )
    result = estimate_latent_covariance(image, score_model, encoder, config, generator=generator)

    payload = {
        "z": result.z.cpu(),
        "covariance": result.covariance.cpu(),
        "score": result.score.cpu(),
        "autocorrelation": result.autocorrelation.cpu(),
        "eta_hat": result.eta_hat.cpu(),
        "sqrt_spectrum": result.sqrt_spectrum.cpu(),
        "args": vars(args),
        "score_checkpoint_meta": {
            "model_config": score_ckpt.get("model_config"),
            "loss_config": score_ckpt.get("loss_config"),
            "step": score_ckpt.get("step"),
        },
    }
    if args.save_deltas:
        payload["deltas"] = result.deltas.cpu()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    cov = result.covariance[0]
    metrics = {
        "encoder": args.encoder,
        "latent_dim": int(cov.shape[0]),
        "trace": float(torch.trace(cov).detach().cpu()),
        "diag_mean": float(cov.diagonal().mean().detach().cpu()),
        "fro_norm": float(torch.linalg.matrix_norm(cov).detach().cpu()),
        "z_norm": float(result.z[0].norm().detach().cpu()),
    }
    print(json.dumps(metrics, indent=2))
    print(f"saved covariance payload to {output}")


if __name__ == "__main__":
    main()
