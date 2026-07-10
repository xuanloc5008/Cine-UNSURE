#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import load_yaml, project_root, resolve_path, select_device
from cunsure_monai3d.data import H5NoisyVolumeDataset
from cunsure_monai3d.losses import MinimaxCUNSURE3DLoss
from cunsure_monai3d.models import build_monai_unet3d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: Path, model: torch.nn.Module, loss_fn: MinimaxCUNSURE3DLoss, cfg: dict, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "eta": loss_fn.eta.detach().cpu(),
            "eta_grad_momentum": loss_fn.eta_grad_momentum.detach().cpu(),
            "config": cfg,
        },
        path,
    )


def run_epoch(
    model: torch.nn.Module,
    loss_fn: MinimaxCUNSURE3DLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    eta_update_every: int,
) -> dict[str, float]:
    model.train()
    total = {"loss": 0.0, "residual": 0.0, "divergence": 0.0}
    n = 0
    for step, y in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        y = y.to(device, non_blocking=True)
        out = loss_fn(model, y)
        loss_mean = out.loss.mean()
        update_eta = step % eta_update_every == 0
        grad_eta = None
        if update_eta:
            grad_eta = torch.autograd.grad(loss_mean, loss_fn.eta, retain_graph=True)[0]

        optimizer.zero_grad(set_to_none=True)
        loss_mean.backward()
        optimizer.step()
        loss_fn.eta.grad = None
        if grad_eta is not None:
            loss_fn.ascend_eta(grad_eta.detach())

        b = y.shape[0]
        total["loss"] += float(loss_mean.detach().cpu()) * b
        total["residual"] += float(out.residual.mean().cpu()) * b
        total["divergence"] += float(out.divergence.mean().cpu()) * b
        n += b
    return {k: v / max(n, 1) for k, v in total.items()}


@torch.no_grad()
def validate(model: torch.nn.Module, loss_fn: MinimaxCUNSURE3DLoss, loader: DataLoader, *, device: torch.device) -> dict[str, float]:
    model.eval()
    total = {"loss": 0.0, "residual": 0.0, "divergence": 0.0}
    n = 0
    for y in tqdm(loader, desc="val", leave=False):
        y = y.to(device, non_blocking=True)
        out = loss_fn(model, y)
        b = y.shape[0]
        total["loss"] += float(out.loss.mean().cpu()) * b
        total["residual"] += float(out.residual.mean().cpu()) * b
        total["divergence"] += float(out.divergence.mean().cpu()) * b
        n += b
    return {k: v / max(n, 1) for k, v in total.items()}


def is_stable_best_candidate(
    val_metrics: dict[str, float],
    eta_norm: float,
    *,
    max_divergence_ratio: float,
    max_residual: float | None,
    eta_max_norm: float | None,
) -> bool:
    residual = max(float(val_metrics["residual"]), 1.0e-12)
    if max_residual is not None and residual > max_residual:
        return False
    divergence = abs(float(val_metrics["divergence"]))
    if divergence > max_divergence_ratio * residual:
        return False
    if eta_max_norm is not None and eta_norm > eta_max_norm:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_cunsure.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    set_seed(int(cfg["seed"]))
    device = select_device(cfg.get("device", "auto"))

    train_ds = H5NoisyVolumeDataset(resolve_path(cfg["data"]["train_h5"], root))
    val_ds = H5NoisyVolumeDataset(resolve_path(cfg["data"]["val_h5"], root))
    with h5py.File(resolve_path(cfg["data"]["train_h5"], root), "r") as h5:
        if "volume_size" in h5.attrs:
            cfg["data"]["volume_size"] = [int(v) for v in h5.attrs["volume_size"]]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_monai_unet3d(cfg["model"]).to(device)
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
    eta_update_every = max(int(loss_cfg.get("eta_update_every", 1)), 1)
    best_divergence_ratio = float(cfg["output"].get("best_divergence_ratio", 0.3))
    best_max_residual = cfg["output"].get("best_max_residual")
    best_max_residual = None if best_max_residual is None else float(best_max_residual)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )
    run_dir = resolve_path(cfg["output"]["run_dir"], root)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val_residual = float("inf")

    for epoch in range(1, int(cfg["optim"]["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            loss_fn,
            train_loader,
            optimizer,
            device=device,
            eta_update_every=eta_update_every,
        )
        val_metrics = validate(model, loss_fn, val_loader, device=device)
        eta = loss_fn.eta.detach().cpu()
        eta_norm = float(eta.norm())
        stable_best_candidate = is_stable_best_candidate(
            val_metrics,
            eta_norm,
            max_divergence_ratio=best_divergence_ratio,
            max_residual=best_max_residual,
            eta_max_norm=loss_fn.eta_max_norm,
        )
        log = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "eta_mean": float(eta.mean()),
            "eta_norm": eta_norm,
            "stable_best_candidate": stable_best_candidate,
        }
        print(json.dumps(log, indent=2))
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")

        if stable_best_candidate and val_metrics["residual"] < best_val_residual:
            best_val_residual = val_metrics["residual"]
            save_checkpoint(run_dir / "best.pt", model, loss_fn, cfg, epoch)
        if epoch % int(cfg["output"]["checkpoint_every"]) == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", model, loss_fn, cfg, epoch)

    save_checkpoint(run_dir / "last.pt", model, loss_fn, cfg, int(cfg["optim"]["epochs"]))


if __name__ == "__main__":
    main()
