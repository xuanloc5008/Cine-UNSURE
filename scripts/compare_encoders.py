#!/usr/bin/env python3
"""Compare CineMA and MedSAM2 under the same score C-UNSURE image noise model."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--image", required=True)
    parser.add_argument("--score-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-root", default=str(default_external_root()))
    parser.add_argument("--encoders", nargs="+", default=["cinema", "medsam2"], choices=["identity", "cinema", "medsam2"])
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

    parser.add_argument("--identity-dim", type=int, default=16)
    parser.add_argument("--cinema-view", default="lax_4c")
    parser.add_argument("--cinema-pool", default="cls")
    parser.add_argument("--cinema-cache-dir", default=None)
    parser.add_argument("--medsam2-config", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--medsam2-checkpoint", default=None)
    parser.add_argument("--medsam2-pool", default="vision_mean")
    parser.add_argument("--medsam2-volume-pool", default="mean", choices=["mean", "max", "mean_std", "flatten"])
    return parser.parse_args()


def build_one_encoder(name: str, args: argparse.Namespace) -> torch.nn.Module:
    if name == "identity":
        return build_encoder("identity", out_dim=args.identity_dim)
    if name == "cinema":
        return build_encoder(
            "cinema",
            external_root=args.external_root,
            view=args.cinema_view,
            pool=args.cinema_pool,
            device=args.device,
            cache_dir=args.cinema_cache_dir,
        )
    if name == "medsam2":
        return build_encoder(
            "medsam2",
            external_root=args.external_root,
            config_name=args.medsam2_config,
            checkpoint=args.medsam2_checkpoint,
            pool=args.medsam2_pool,
            volume_pool=args.medsam2_volume_pool,
            device=args.device,
        )
    raise ValueError(name)


def summarize(name: str, result) -> dict[str, float | int | str]:
    cov = result.covariance[0]
    diag = cov.diagonal()
    metric: dict[str, float | int | str] = {
        "encoder": name,
        "latent_dim": int(cov.shape[0]),
        "trace": float(torch.trace(cov).detach().cpu()),
        "diag_mean": float(diag.mean().detach().cpu()),
        "diag_min": float(diag.min().detach().cpu()),
        "diag_max": float(diag.max().detach().cpu()),
        "fro_norm": float(torch.linalg.matrix_norm(cov).detach().cpu()),
        "z_norm": float(result.z[0].norm().detach().cpu()),
        "eta_mean": float(result.eta_hat.mean().detach().cpu()),
        "eta_center": float(result.eta_hat.flatten(start_dim=1)[0, result.eta_hat[0].numel() // 2].detach().cpu()),
    }
    if cov.shape[0] <= 2048:
        eig = torch.linalg.eigvalsh(cov.float()).detach().cpu()
        metric["eig_min"] = float(eig.min())
        metric["eig_max"] = float(eig.max())
        metric["rank_tol_1e-6"] = int((eig > 1.0e-6).sum())
    return metric


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    score_model, _ = load_score_checkpoint(args.score_checkpoint, device=device)
    image = load_frame(
        args.image,
        npz_key=args.npz_key,
        channels=args.channels,
        normalize=True,
        spatial_dims=args.spatial_dims,
        frame_index=args.time_index,
        time_axis=args.time_axis if args.time_index is not None else None,
        frame_layout=args.frame_layout,
    ).unsqueeze(0).to(device)
    config = CUNSUREConfig(
        radius=args.radius,
        eps=args.eps,
        spectral_floor=args.spectral_floor,
        n_probes=args.n_probes,
        finite_difference_tau=args.tau,
    )

    rows = []
    for idx, name in enumerate(args.encoders):
        print(f"running encoder={name}")
        encoder = build_one_encoder(name, args).to(device).eval()
        generator = torch.Generator(device=device).manual_seed(args.seed + idx)
        result = estimate_latent_covariance(image, score_model, encoder, config, generator=generator)
        torch.save(
            {
                "z": result.z.cpu(),
                "covariance": result.covariance.cpu(),
                "autocorrelation": result.autocorrelation.cpu(),
                "eta_hat": result.eta_hat.cpu(),
                "args": vars(args),
            },
            out_dir / f"{name}_covariance.pt",
        )
        rows.append(summarize(name, result))

    csv_path = out_dir / "comparison_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved comparison metrics to {csv_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
