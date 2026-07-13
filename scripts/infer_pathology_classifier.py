#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import project_root, resolve_path, select_device
from cunsure_monai3d.pathology import (
    PathologyClassifier,
    clinical_features,
    pathology_probability_bands,
    read_acdc_pathology,
)


def expand_patterns(patterns: list[str], root: Path) -> list[Path]:
    paths = []
    for pattern in patterns:
        path = Path(pattern)
        query = str(path if path.is_absolute() else root / path)
        paths.extend(Path(value) for value in glob.glob(query))
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clinical", nargs="+", required=True, help="clinical JSON paths or glob patterns")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--coverage", type=float, default=0.95)
    args = parser.parse_args()

    root = project_root()
    checkpoint = torch.load(resolve_path(args.checkpoint, root), map_location="cpu", weights_only=False)
    device = select_device(args.device)
    classes = [str(value) for value in checkpoint["classes"]]
    model_cfg = checkpoint["model"]
    feature_mean = checkpoint["feature_mean"].float().to(device)
    feature_std = checkpoint["feature_std"].float().to(device)
    model = PathologyClassifier(
        input_dim=feature_mean.numel(),
        hidden_dims=tuple(int(value) for value in model_cfg["hidden_dims"]),
        num_classes=len(classes),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    coverage = float(args.coverage)
    if not 0.0 < coverage < 1.0:
        raise ValueError("--coverage must be between 0 and 1")
    multiplier = NormalDist().inv_cdf(0.5 + coverage / 2.0)
    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = expand_patterns(args.clinical, root)
    if not paths:
        raise ValueError("no clinical prediction files matched --clinical")

    for path in paths:
        clinical = json.loads(path.read_text(encoding="utf-8"))
        raw_features, raw_variance = clinical_features(clinical, time_points=int(checkpoint["time_points"]))
        features = (torch.from_numpy(raw_features).to(device) - feature_mean) / feature_std
        feature_variance = torch.from_numpy(raw_variance).to(device) / feature_std.square()
        probabilities, probability_variance, lower, upper = pathology_probability_bands(
            model, features, feature_variance, gaussian_multiplier=multiplier
        )
        predicted_index = int(probabilities.argmax())
        target = None
        try:
            target = read_acdc_pathology(str(clinical["source_path"]), root)
        except (FileNotFoundError, ValueError):
            pass
        result = {
            "clinical_file": str(path),
            "source_path": clinical["source_path"],
            "predicted_pathology": classes[predicted_index],
            "target_pathology": target,
            "coverage": coverage,
            "uncertainty_method": "clinical_covariance_delta_method",
            "uncertainty_scope": "propagated clinical-metric uncertainty only",
            "pathology_probabilities": {
                name: {
                    "mean": float(probabilities[index].detach().cpu()),
                    "variance": float(probability_variance[index].detach().cpu()),
                    "lower": float(lower[index].detach().cpu()),
                    "upper": float(upper[index].detach().cpu()),
                }
                for index, name in enumerate(classes)
            },
        }
        output = output_dir / path.name
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output), "prediction": classes[predicted_index], "target": target}))


if __name__ == "__main__":
    main()
