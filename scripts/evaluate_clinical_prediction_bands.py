#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    targets = {}
    for line in Path(args.targets).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            targets[str(row["source_path"])] = row

    ef_errors = []
    ef_covered = []
    ef_widths = []
    volume_errors = []
    volume_covered = []
    volume_widths = []
    rows = []
    for path in sorted(glob.glob(args.predictions)):
        prediction = json.loads(Path(path).read_text(encoding="utf-8"))
        source = str(prediction["source_path"])
        if source not in targets:
            continue
        target = targets[source]
        ef_band = prediction["prediction_bands"]["ef"]
        ef_target = float(target["ef"])
        ef_errors.append(abs(float(prediction["ef"]) - ef_target))
        ef_covered.append(float(ef_band["lower"]) <= ef_target <= float(ef_band["upper"]))
        ef_widths.append(float(ef_band["upper"]) - float(ef_band["lower"]))

        for frame_index, target_key in (
            (int(prediction["ed_index"]), "ed_volume_ml_gt"),
            (int(prediction["es_index"]), "es_volume_ml_gt"),
        ):
            volume_target = float(target[target_key])
            volume_mean = float(prediction["volume_curve"][frame_index])
            volume_band = prediction["prediction_bands"]["volume_curve"][frame_index]
            volume_errors.append(abs(volume_mean - volume_target))
            volume_covered.append(float(volume_band["lower"]) <= volume_target <= float(volume_band["upper"]))
            volume_widths.append(float(volume_band["upper"]) - float(volume_band["lower"]))
        rows.append({"source_path": source, "prediction_file": path})

    if not rows:
        raise ValueError("no matched independent clinical evaluation predictions")

    result = {
        "independent_evaluation": True,
        "num_sequences": len(rows),
        "ef": {
            "mae": float(np.mean(ef_errors)),
            "coverage": float(np.mean(ef_covered)),
            "mean_band_width": float(np.mean(ef_widths)),
        },
        "volume_ed_es_ml": {
            "mae": float(np.mean(volume_errors)),
            "coverage": float(np.mean(volume_covered)),
            "mean_band_width": float(np.mean(volume_widths)),
        },
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
