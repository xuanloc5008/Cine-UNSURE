#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_for_display(image: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((image - low) / max(high - low, 1.0e-8), 0.0, 1.0)


def select_motion_slice(images: np.ndarray) -> int:
    temporal_variance = images[:, 0].var(axis=0)
    scores = temporal_variance.mean(axis=(1, 2))
    return int(np.argmax(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Per-sequence NODEO .pt output")
    parser.add_argument("--output", required=True, help="Output GIF path")
    parser.add_argument("--slice-index", type=int)
    parser.add_argument("--fps", type=float, default=4.0)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    images = payload["images"].float().numpy()
    warped = payload["warped"].float().numpy()
    if images.ndim != 5 or images.shape[1] != 1:
        raise ValueError(f"expected images [T,1,D,H,W], got {images.shape}")
    if warped.shape[0] != images.shape[0] - 1:
        raise ValueError(f"expected warped sequence length T-1, got images={images.shape}, warped={warped.shape}")

    predicted = np.concatenate([images[0:1], warped], axis=0)
    depth = images.shape[2]
    slice_index = select_motion_slice(images) if args.slice_index is None else int(args.slice_index)
    if not 0 <= slice_index < depth:
        raise ValueError(f"slice-index must be in [0, {depth - 1}], got {slice_index}")

    values = images[:, 0, slice_index]
    low, high = np.percentile(values, [1.0, 99.0])
    fixed = images[0, 0, slice_index]
    absolute_errors = np.abs(predicted[:, 0, slice_index] - images[:, 0, slice_index])
    error_high = max(float(np.percentile(absolute_errors, 99.0)), 1.0e-6)
    frames: list[np.ndarray] = []

    for frame_index in range(images.shape[0]):
        target = images[frame_index, 0, slice_index]
        prediction = predicted[frame_index, 0, slice_index]
        error = np.abs(prediction - target)
        figure, axes = plt.subplots(1, 4, figsize=(12, 3.2), dpi=120)
        panels = (
            (normalize_for_display(fixed, low, high), "Fixed I0", "gray", 0.0, 1.0),
            (normalize_for_display(prediction, low, high), f"Predicted k={frame_index}", "gray", 0.0, 1.0),
            (normalize_for_display(target, low, high), f"Target k={frame_index}", "gray", 0.0, 1.0),
            (error, "|Predicted - Target|", "magma", 0.0, error_high),
        )
        for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels, strict=True):
            axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        figure.suptitle(
            f"{payload['dataset']} | slice {slice_index} | time {int(payload['raw_time_indices'][frame_index])}",
            fontsize=11,
        )
        figure.tight_layout()
        figure.canvas.draw()
        frames.append(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(figure)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, duration=1.0 / float(args.fps), loop=0)

    fixed_sequence = np.repeat(images[0:1], images.shape[0], axis=0)
    baseline_mae = float(np.abs(fixed_sequence[1:] - images[1:]).mean())
    predicted_mae = float(np.abs(predicted[1:] - images[1:]).mean())
    improvement = 100.0 * (baseline_mae - predicted_mae) / max(baseline_mae, 1.0e-8)
    print(
        json.dumps(
            {
                "input": str(Path(args.input)),
                "output": str(output),
                "sequence_id": payload["sequence_id"],
                "dataset": payload["dataset"],
                "num_frames": int(images.shape[0]),
                "slice_index": slice_index,
                "fixed_to_target_mae": baseline_mae,
                "predicted_to_target_mae": predicted_mae,
                "mae_improvement_percent": improvement,
                "best_epoch": int(payload["best_epoch"]),
                "metrics": payload["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
