#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import load_yaml, project_root, resolve_path, select_device
from cunsure_monai3d.data import H5NoisyVolumeDataset
from cunsure_monai3d.models import build_monai_unet3d
from cunsure_monai3d.score_cunsure import ardae_score_loss, linear_delta


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset: H5NoisyVolumeDataset, cfg: dict, *, shuffle: bool, device: torch.device) -> DataLoader:
    workers = int(cfg.get("num_workers", 0))
    kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(cfg.get("pin_memory", device.type == "cuda")) and device.type == "cuda",
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: dict,
    epoch: int,
    global_step: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": "unsure_ardae_score",
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": metrics,
            "config": cfg,
        },
        path,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    global_step: int,
    total_steps: int,
    delta_min: float,
    delta_max: float,
    grad_clip: float | None,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_score_norm = 0.0
    count = 0
    last_delta = delta_max
    for y in tqdm(loader, desc="score train", leave=False):
        y = y.to(device, non_blocking=True)
        last_delta = linear_delta(global_step + 1, total_steps, delta_min=delta_min, delta_max=delta_max)
        loss_per_item, score = ardae_score_loss(model, y, delta=last_delta)
        loss = loss_per_item.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch = y.shape[0]
        total_loss += float(loss.detach()) * batch
        total_score_norm += float(score.detach().square().flatten(start_dim=1).mean(dim=1).mean()) * batch
        count += batch
        global_step += 1
    return {
        "loss": total_loss / max(count, 1),
        "score_square_mean": total_score_norm / max(count, 1),
        "delta": float(last_delta),
    }, global_step


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    delta_min: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_score_norm = 0.0
    count = 0
    for y in tqdm(loader, desc="score val", leave=False):
        y = y.to(device, non_blocking=True)
        loss_per_item, _ = ardae_score_loss(model, y, delta=delta_min)
        score = model(y)
        batch = y.shape[0]
        total_loss += float(loss_per_item.mean()) * batch
        total_score_norm += float(score.square().flatten(start_dim=1).mean(dim=1).mean()) * batch
        count += batch
    score_square_mean = total_score_norm / max(count, 1)
    return {
        "loss": total_loss / max(count, 1),
        "score_square_mean": score_square_mean,
        "unsure_scalar_variance": 1.0 / max(score_square_mean, 1.0e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_cunsure_score.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    set_seed(int(cfg.get("seed", 2026)))
    device = select_device(cfg.get("device", "auto"))

    train_path = resolve_path(cfg["data"]["train_h5"], root)
    val_path = resolve_path(cfg["data"]["val_h5"], root)
    train_ds = H5NoisyVolumeDataset(train_path)
    val_ds = H5NoisyVolumeDataset(val_path)
    with h5py.File(train_path, "r") as h5:
        if "volume_size" in h5.attrs:
            cfg["data"]["volume_size"] = [int(v) for v in h5.attrs["volume_size"]]
    train_loader = make_loader(train_ds, cfg["data"], shuffle=True, device=device)
    val_loader = make_loader(val_ds, cfg["data"], shuffle=False, device=device)

    model = build_monai_unet3d(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"].get("weight_decay", 0.0)),
    )
    epochs = int(cfg["optim"]["epochs"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(int(0.8 * epochs), 1),
        gamma=float(cfg["optim"].get("lr_gamma", 0.1)),
    )
    total_steps = epochs * len(train_loader)
    delta_min = float(cfg["score"].get("delta_min", 0.001))
    delta_max = float(cfg["score"].get("delta_max", 0.1))
    grad_clip = cfg["optim"].get("grad_clip")
    grad_clip = None if grad_clip is None else float(grad_clip)

    start_epoch = 1
    global_step = 0
    if args.resume:
        checkpoint = torch.load(resolve_path(args.resume, root), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", (start_epoch - 1) * len(train_loader)))

    run_dir = resolve_path(cfg["output"]["run_dir"], root)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if start_epoch == 1 and metrics_path.exists():
        metrics_path.unlink()
    best_val = (
        float(checkpoint.get("metrics", {}).get("val", {}).get("loss", float("inf")))
        if args.resume
        else float("inf")
    )
    stale_epochs = 0
    patience = int(cfg["optim"].get("early_stopping_patience", 0))
    min_delta = float(cfg["optim"].get("early_stopping_min_delta", 0.0))

    final_epoch = start_epoch - 1
    log = checkpoint.get("metrics", {}) if args.resume else {}
    for epoch in range(start_epoch, epochs + 1):
        train_metrics, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            global_step=global_step,
            total_steps=total_steps,
            delta_min=delta_min,
            delta_max=delta_max,
            grad_clip=grad_clip,
        )
        val_metrics = validate(model, val_loader, device=device, delta_min=delta_min)
        used_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        log = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": used_lr,
            "next_lr": float(optimizer.param_groups[0]["lr"]),
            "global_step": global_step,
        }
        print(json.dumps(log, indent=2))
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")

        improved = val_metrics["loss"] < best_val - min_delta
        if improved:
            best_val = val_metrics["loss"]
            stale_epochs = 0
            save_checkpoint(
                run_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                metrics=log,
            )
        else:
            stale_epochs += 1
        if epoch % int(cfg["output"].get("checkpoint_every", 10)) == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
                metrics=log,
            )
        final_epoch = epoch
        if patience > 0 and stale_epochs >= patience:
            print(f"early stopping at epoch {epoch}; no validation improvement for {patience} epochs")
            break

    save_checkpoint(
        run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        epoch=final_epoch,
        global_step=global_step,
        metrics=log,
    )


if __name__ == "__main__":
    main()
