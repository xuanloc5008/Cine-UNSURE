#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nodeo_uq.config import project_root, resolve_path, select_device
from cardiac_nodeo_uq.nodeo_ops import LocalNCC3D, jacobian_det_3d, smoothness_loss


def global_ncc(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, left.ndim))
    left_centered = left - left.mean(dim=dims, keepdim=True)
    right_centered = right - right.mean(dim=dims, keepdim=True)
    numerator = (left_centered * right_centered).sum(dim=dims)
    denominator = (
        left_centered.square().sum(dim=dims).sqrt()
        * right_centered.square().sum(dim=dims).sqrt()
    ).clamp_min(1.0e-12)
    return numerator / denominator


def image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    local_ncc: LocalNCC3D,
) -> dict[str, float]:
    difference = prediction - target
    mae = difference.abs().mean()
    mse = difference.square().mean()
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1.0e-12))
    ncc = global_ncc(prediction, target).mean()
    nodeo_lncc_squared = 1.0 - local_ncc(prediction, target)
    return {
        "mae": float(mae.cpu()),
        "mse": float(mse.cpu()),
        "psnr_db": float(psnr.cpu()),
        "global_ncc": float(ncc.cpu()),
        "nodeo_local_ncc_squared": float(nodeo_lncc_squared.cpu()),
    }


def mean_metrics(rows: list[dict], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / max(len(rows), 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--ncc-window", type=int, default=21)
    args = parser.parse_args()

    root = project_root()
    payload = torch.load(resolve_path(args.input, root), map_location="cpu", weights_only=False)
    device = select_device(args.device)
    images = payload["images"].float().to(device)
    warped = payload["warped"].float().to(device)
    displacement = payload["displacement"].float().to(device)
    phi = payload["phi_bar"].float().to(device)
    times = payload["times"].float().to(device)
    raw_times = [int(v) for v in payload["raw_time_indices"]]
    if warped.shape[0] != images.shape[0] - 1:
        raise ValueError(f"warped must contain T-1 frames, got images={images.shape}, warped={warped.shape}")

    local_ncc = LocalNCC3D(win=int(args.ncc_window)).to(device)
    fixed = images[0:1]
    frame_rows: list[dict] = []
    baseline_rows: list[dict] = []
    with torch.no_grad():
        for sequence_index in range(1, int(images.shape[0])):
            target = images[sequence_index : sequence_index + 1]
            prediction = warped[sequence_index - 1 : sequence_index]
            predicted_metrics = image_metrics(prediction, target, local_ncc=local_ncc)
            baseline_metrics = image_metrics(fixed, target, local_ncc=local_ncc)
            determinant = jacobian_det_3d(phi[sequence_index : sequence_index + 1])
            frame_rows.append(
                {
                    "sequence_index": sequence_index,
                    "raw_time_index": raw_times[sequence_index],
                    **predicted_metrics,
                    "spatial_smoothness": float(
                        smoothness_loss(displacement[sequence_index : sequence_index + 1]).cpu()
                    ),
                    "displacement_rms_voxels": float(
                        displacement[sequence_index].square().mean().sqrt().cpu()
                    ),
                    "jacobian_mean": float(determinant.mean().cpu()),
                    "jacobian_std": float(determinant.std().cpu()),
                    "jacobian_min": float(determinant.min().cpu()),
                    "jacobian_max": float(determinant.max().cpu()),
                    "abs_jacobian_minus_one": float((determinant - 1.0).abs().mean().cpu()),
                    "fold_fraction": float((determinant <= 0.0).float().mean().cpu()),
                    "below_minimum_jacobian_fraction": float((determinant < 0.5).float().mean().cpu()),
                }
            )
            baseline_rows.append(baseline_metrics)

        dt = (times[1:] - times[:-1]).clamp_min(1.0e-6).view(-1, 1, 1, 1, 1)
        trajectory_velocity = (displacement[1:] - displacement[:-1]) / dt
        velocity_rms = float(trajectory_velocity.square().mean().sqrt().cpu())
        if trajectory_velocity.shape[0] > 1:
            velocity_times = 0.5 * (times[1:] + times[:-1])
            velocity_dt = (velocity_times[1:] - velocity_times[:-1]).clamp_min(1.0e-6).view(-1, 1, 1, 1, 1)
            acceleration = (trajectory_velocity[1:] - trajectory_velocity[:-1]) / velocity_dt
            acceleration_rms = float(acceleration.square().mean().sqrt().cpu())
        else:
            acceleration_rms = math.nan

    metric_keys = ("mae", "mse", "psnr_db", "global_ncc", "nodeo_local_ncc_squared")
    predicted_mean = {key: mean_metrics(frame_rows, key) for key in metric_keys}
    baseline_mean = {key: mean_metrics(baseline_rows, key) for key in metric_keys}
    report = {
        "input": args.input,
        "dataset": payload["dataset"],
        "source_path": payload["source_path"],
        "sequence_id": payload["sequence_id"],
        "num_frames": int(images.shape[0]),
        "best_epoch": int(payload["best_epoch"]),
        "intensity_summary": {
            "baseline_fixed_to_target": baseline_mean,
            "predicted_to_target": predicted_mean,
            "mae_improvement_percent": 100.0
            * (baseline_mean["mae"] - predicted_mean["mae"])
            / max(baseline_mean["mae"], 1.0e-12),
        },
        "trajectory_summary": {
            "spatial_smoothness_mean": mean_metrics(frame_rows, "spatial_smoothness"),
            "temporal_velocity_rms_voxels_per_normalized_time": velocity_rms,
            "temporal_acceleration_rms_voxels_per_normalized_time2": acceleration_rms,
            "abs_jacobian_minus_one_mean": mean_metrics(frame_rows, "abs_jacobian_minus_one"),
            "fold_fraction_mean": mean_metrics(frame_rows, "fold_fraction"),
            "fold_fraction_max": max(float(row["fold_fraction"]) for row in frame_rows),
            "jacobian_min_over_sequence": min(float(row["jacobian_min"]) for row in frame_rows),
            "jacobian_max_over_sequence": max(float(row["jacobian_max"]) for row in frame_rows),
        },
        "per_frame": frame_rows,
    }
    output = resolve_path(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
