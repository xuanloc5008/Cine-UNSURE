#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import project_root, resolve_path, select_device
from cunsure_monai3d.models import build_monai_unet3d
from cunsure_monai3d.score_cunsure import estimate_frame_noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--spectral-floor", type=float, default=1.0e-6)
    parser.add_argument("--relative-spectral-floor", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    root = project_root()
    device = select_device(args.device)
    checkpoint = torch.load(resolve_path(args.checkpoint, root), map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "unsure_ardae_score":
        raise ValueError("checkpoint is not an UNSURE AR-DAE score model")
    model = build_monai_unet3d(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    h5_path = resolve_path(args.h5, root)
    with h5py.File(h5_path, "r") as h5:
        count = min(max(args.num_frames, 1), len(h5["y"]))
        indices = np.linspace(0, len(h5["y"]) - 1, count, dtype=int)
        records = []
        for index in indices:
            y = torch.from_numpy(h5["y"][index : index + 1]).float().to(device)
            estimate = estimate_frame_noise(
                model,
                y,
                kernel_size=args.kernel_size,
                spectral_floor=args.spectral_floor,
                relative_spectral_floor=args.relative_spectral_floor,
                covariance_floor=0.0,
            )
            records.append(
                {
                    "index": int(index),
                    "eta_mean": float(estimate.eta.mean()),
                    "eta_norm": float(estimate.eta.norm()),
                    "covariance_trace": float(estimate.covariance_spectrum.sum()),
                    "score_square_mean": float(estimate.score.square().mean()),
                }
            )

    eta_means = np.asarray([record["eta_mean"] for record in records])
    traces = np.asarray([record["covariance_trace"] for record in records])
    report = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "h5": str(h5_path),
        "num_frames": len(records),
        "eta_mean": {
            "min": float(eta_means.min()),
            "mean": float(eta_means.mean()),
            "max": float(eta_means.max()),
            "std": float(eta_means.std()),
        },
        "image_covariance_spectral_sum": {
            "min": float(traces.min()),
            "mean": float(traces.mean()),
            "max": float(traces.max()),
        },
        "frame_dependent_eta": bool(eta_means.std() > 0.0),
        "records": records,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
