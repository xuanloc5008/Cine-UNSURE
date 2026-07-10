#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import project_root, resolve_path, select_device
from cunsure_monai3d.data import H5NoisyVolumeDataset
from cunsure_monai3d.foundation import covariance_sanity_metrics, project_covariance_psd, symmetrize_covariance
from cunsure_monai3d.losses import MinimaxCUNSURE3DLoss
from cunsure_monai3d.models import build_monai_unet3d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint_model(
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, MinimaxCUNSURE3DLoss, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_monai_unet3d(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loss_cfg = cfg["loss"]
    loss_fn = MinimaxCUNSURE3DLoss(
        kernel_size=int(loss_cfg["kernel_size"]),
        eta_init=float(loss_cfg["eta_init"]),
        tau=float(loss_cfg["tau"]),
        eta_step_size=float(loss_cfg["eta_step_size"]),
        eta_momentum=float(loss_cfg["eta_momentum"]),
        eta_grad_clip=None if loss_cfg.get("eta_grad_clip") is None else float(loss_cfg["eta_grad_clip"]),
        eta_max_norm=None if loss_cfg.get("eta_max_norm") is None else float(loss_cfg["eta_max_norm"]),
        device=device,
    ).to(device)
    loss_fn.eta.data.copy_(ckpt["eta"].to(device))
    loss_fn.eval()
    return model, loss_fn, ckpt


@torch.no_grad()
def denoise_metrics(model: torch.nn.Module, loader: DataLoader, *, device: torch.device) -> dict[str, float]:
    total = {
        "input_mean": 0.0,
        "input_std": 0.0,
        "output_mean": 0.0,
        "output_std": 0.0,
        "delta_abs_mean": 0.0,
        "delta_rmse": 0.0,
        "output_min": 0.0,
        "output_max": 0.0,
    }
    n = 0
    for y in tqdm(loader, desc="denoise", leave=False):
        y = y.to(device, non_blocking=True)
        x = model(y)
        delta = x - y
        b = y.shape[0]
        total["input_mean"] += float(y.mean().cpu()) * b
        total["input_std"] += float(y.std().cpu()) * b
        total["output_mean"] += float(x.mean().cpu()) * b
        total["output_std"] += float(x.std().cpu()) * b
        total["delta_abs_mean"] += float(delta.abs().mean().cpu()) * b
        total["delta_rmse"] += float(delta.pow(2).mean().sqrt().cpu()) * b
        total["output_min"] += float(x.min().cpu()) * b
        total["output_max"] += float(x.max().cpu()) * b
        n += b
    return {k: v / max(n, 1) for k, v in total.items()}


def cunsure_metrics(
    model: torch.nn.Module,
    loss_fn: MinimaxCUNSURE3DLoss,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    total = {"loss": 0.0, "residual": 0.0, "divergence": 0.0}
    n = 0
    model.eval()
    for y in tqdm(loader, desc="cunsure", leave=False):
        y = y.to(device, non_blocking=True)
        with torch.enable_grad():
            out = loss_fn(model, y)
        b = y.shape[0]
        total["loss"] += float(out.loss.mean().detach().cpu()) * b
        total["residual"] += float(out.residual.mean().cpu()) * b
        total["divergence"] += float(out.divergence.mean().cpu()) * b
        n += b
    metrics = {k: v / max(n, 1) for k, v in total.items()}
    residual = max(metrics["residual"], 1.0e-12)
    metrics["abs_divergence_to_residual"] = abs(metrics["divergence"]) / residual
    return metrics


def eta_metrics(eta: torch.Tensor) -> dict[str, float]:
    eta = eta.detach().cpu()
    return {
        "sum": float(eta.sum()),
        "norm": float(eta.norm()),
        "mean": float(eta.mean()),
        "min": float(eta.min()),
        "max": float(eta.max()),
        "std": float(eta.std()),
    }


def covariance_metrics(path: Path | None) -> dict[str, dict[str, float]] | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = "latent_covariance_psd" if "latent_covariance_psd" in payload else "latent_covariance"
    cov = payload[key]
    cov_sym = symmetrize_covariance(cov)
    cov_psd = project_covariance_psd(cov_sym)
    return {
        "source_key": {"latent_covariance_psd": float(key == "latent_covariance_psd")},
        "raw": covariance_sanity_metrics(cov),
        "sym": covariance_sanity_metrics(cov_sym),
        "psd": covariance_sanity_metrics(cov_psd),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--covariance", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    root = project_root()
    set_seed(args.seed)
    device = select_device(args.device)
    model, loss_fn, ckpt = load_checkpoint_model(resolve_path(args.checkpoint, root), device=device)

    dataset = H5NoisyVolumeDataset(resolve_path(args.h5, root))
    indices = list(range(len(dataset)))
    if args.limit > 0:
        indices = indices[: min(args.limit, len(indices))]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "h5": str(args.h5),
        "num_samples": len(indices),
        "device": str(device),
        "eta": eta_metrics(loss_fn.eta),
        "cunsure": cunsure_metrics(model, loss_fn, loader, device=device),
        "denoise": denoise_metrics(model, loader, device=device),
    }
    cov_path = resolve_path(args.covariance, root)
    cov_report = covariance_metrics(cov_path)
    if cov_report is not None:
        report["covariance"] = cov_report

    output = resolve_path(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
