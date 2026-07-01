"""Checkpoint helpers for score models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from score_cunsure.score_model import ScoreLossConfig, ScoreUNet


def save_score_checkpoint(
    path: str | Path,
    model: ScoreUNet,
    *,
    model_config: dict[str, Any],
    loss_config: ScoreLossConfig,
    step: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model_config,
            "loss_config": loss_config.__dict__,
            "step": step,
        },
        path,
    )


def load_score_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> tuple[ScoreUNet, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config", {})
    model = ScoreUNet(**model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt

