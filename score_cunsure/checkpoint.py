"""Checkpoint helpers for score models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from score_cunsure.score_model import ScoreLossConfig, ScoreUNet


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_score_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    model_config: dict[str, Any],
    loss_config: ScoreLossConfig,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    best_val_loss: float | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    unwrapped = unwrap_model(model)
    checkpoint: dict[str, Any] = {
        "model": unwrapped.state_dict(),
        "model_config": model_config,
        "loss_config": loss_config.__dict__,
        "step": step,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss
    if metrics is not None:
        checkpoint["metrics"] = metrics
    torch.save(checkpoint, path)


def load_score_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[ScoreUNet, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config", {})
    model = ScoreUNet(**model_config).to(device)
    state_dict = ckpt["model"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt
