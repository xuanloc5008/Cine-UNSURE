#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import as_tuple_int, load_yaml, project_root, resolve_path
from cunsure_monai3d.deformation_data import DeformationSequenceDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_nodeo_mean_deformation.yaml")
    parser.add_argument("--num-sequences", type=int, default=5)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    data_cfg = cfg["data"]
    image_shape_key = "image_shape" if "image_shape" in cfg["model"] else "deformation_shape"
    deformation_shape = as_tuple_int(cfg["model"][image_shape_key], name=image_shape_key)

    dataset = DeformationSequenceDataset(
        resolve_path(data_cfg["h5"], root),
        root=root,
        min_length=int(data_cfg.get("min_length", 2)),
        covariance=str(data_cfg.get("covariance", "diag")),
        normalize_time=bool(data_cfg.get("normalize_time", True)),
        time_axis=int(data_cfg.get("time_axis", -1)),
        volume_size=as_tuple_int(data_cfg["volume_size"], name="volume_size"),
        image_size=deformation_shape,
        normalize=str(data_cfg.get("normalize", "percentile")),
        percentile_low=float(data_cfg.get("percentile_low", 1.0)),
        percentile_high=float(data_cfg.get("percentile_high", 99.0)),
        source_path_remap=list(data_cfg.get("source_path_remap", [])),
        cache_data=False,
        roi_mask_crop=bool(data_cfg.get("roi_mask_crop", False)),
        roi_mask_margin=as_tuple_int(data_cfg.get("roi_mask_margin", [0, 12, 12]), name="roi_mask_margin"),
        require_roi_mask=bool(data_cfg.get("require_roi_mask", False)),
    )
    if len(dataset) == 0:
        raise ValueError("no deformation sequences found")

    indices = list(range(len(dataset)))
    if args.random:
        random.Random(int(cfg.get("seed", 2026))).shuffle(indices)
    indices = indices[: max(1, int(args.num_sequences))]

    rows = []
    for idx in indices:
        sample = dataset[idx]
        images = sample["images"]
        z = sample["z"]
        covariance = sample["R"]
        rows.append(
            {
                "sequence_index": int(idx),
                "dataset": sample["dataset"],
                "source_path": sample["source_path"],
                "num_frames": int(images.shape[0]),
                "images_shape": list(images.shape),
                "z_shape": list(z.shape),
                "covariance_shape": list(covariance.shape),
                "time_indices": sample["raw_time_indices"].tolist(),
                "image_min": float(torch.min(images)),
                "image_mean": float(torch.mean(images)),
                "image_max": float(torch.max(images)),
                "finite": bool(torch.isfinite(images).all() and torch.isfinite(z).all() and torch.isfinite(covariance).all()),
            }
        )

    print(
        json.dumps(
            {
                "config": str(root / args.config),
                "h5": str(resolve_path(data_cfg["h5"], root)),
                "num_sequences_total": len(dataset),
                "checked": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
