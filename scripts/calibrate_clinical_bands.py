#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np


def logit(value: float) -> float:
    clipped = float(np.clip(value, 1.0e-5, 1.0 - 1.0e-5))
    return float(np.log(clipped / (1.0 - clipped)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True, help="validation clinical metric JSON files")
    parser.add_argument("--targets", required=True, help="JSONL rows with source_path and target ef")
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage", type=float, default=0.95)
    args = parser.parse_args()

    targets: dict[str, float] = {}
    with Path(args.targets).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                targets[str(row["source_path"])] = float(row["ef"])

    scores: dict[str, list[float]] = {"ef": [], "volume_curve": []}
    used: list[str] = []
    for pattern in args.predictions:
        paths = sorted(Path().glob(pattern)) if any(char in pattern for char in "*?[") else [Path(pattern)]
        for path in paths:
            row = json.loads(path.read_text(encoding="utf-8"))
            source = str(row["source_path"])
            variance = row.get("ef_variance")
            if source not in targets or variance is None or float(variance) <= 0:
                continue
            ef_mean = float(np.clip(float(row["ef"]), 1.0e-5, 1.0 - 1.0e-5))
            ef_se_logit = np.sqrt(float(variance)) / (ef_mean * (1.0 - ef_mean))
            scores["ef"].append(abs(logit(targets[source]) - logit(ef_mean)) / ef_se_logit)
            target_row = None
            with Path(args.targets).open("r", encoding="utf-8") as target_handle:
                for target_line in target_handle:
                    candidate = json.loads(target_line)
                    if str(candidate["source_path"]) == source:
                        target_row = candidate
                        break
            if target_row is not None:
                for frame_index, target_key in (
                    (int(row["ed_index"]), "ed_volume_ml_gt"),
                    (int(row["es_index"]), "es_volume_ml_gt"),
                ):
                    frame_variance = row["volume_variance"][frame_index]
                    if target_key in target_row and frame_variance is not None and float(frame_variance) > 0:
                        volume_mean = max(float(row["volume_curve"][frame_index]), 1.0e-6)
                        volume_target = max(float(target_row[target_key]), 1.0e-6)
                        volume_se_log = np.sqrt(float(frame_variance)) / volume_mean
                        scores["volume_curve"].append(abs(np.log(volume_target) - np.log(volume_mean)) / volume_se_log)
            used.append(source)
    if not scores["ef"]:
        raise ValueError("no matched validation EF targets with positive predicted variance")

    gaussian_multiplier = NormalDist().inv_cdf(0.5 + float(args.coverage) / 2.0)
    scales = {}
    details = {}
    for metric, metric_scores in scores.items():
        if not metric_scores:
            continue
        n_metric = len(metric_scores)
        quantile_level = min(1.0, np.ceil((n_metric + 1) * float(args.coverage)) / n_metric)
        conformal_multiplier = float(np.quantile(np.asarray(metric_scores), quantile_level, method="higher"))
        scales[metric] = conformal_multiplier / gaussian_multiplier
        details[metric] = {
            "num_calibration_values": n_metric,
            "quantile_level": float(quantile_level),
            "conformal_multiplier": conformal_multiplier,
        }
    result = {
        "definition": "split-conformal scale from |target-mean|/predicted_standard_error",
        "transforms": {"ef": "logit", "volume_curve": "log"},
        "coverage": float(args.coverage),
        "num_validation_sequences": len(set(used)),
        "gaussian_multiplier": gaussian_multiplier,
        "calibration_scale": scales["ef"],
        "calibration_scales": scales,
        "details": details,
        "uncalibrated_metrics": ["wall_motion", "strain"],
        "sources": used,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
