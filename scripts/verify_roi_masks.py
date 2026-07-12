#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import load_yaml, project_root
from cunsure_monai3d.preprocess import candidate_mask_paths, scan_nifti_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prepare_hdf5.yaml")
    parser.add_argument("--max-missing", type=int, default=30)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    data = cfg["data"]
    exclude = list(data["exclude_substrings"])
    refs = scan_nifti_frames(
        root,
        list(data.get("train_globs", []))
        + list(data.get("val_globs", []))
        + list(data.get("test_globs", []))
        + list(data.get("mnm2_globs", [])),
        exclude,
        split_4d=False,
        time_axis=int(data["time_axis"]),
    )

    unique_paths = sorted({ref.path for ref in refs})
    rows = []
    missing = []
    for path in unique_paths:
        masks = candidate_mask_paths(path)
        row = {
            "source_path": str(path),
            "num_masks": len(masks),
            "mask_examples": [str(mask) for mask in masks[: args.max_examples]],
        }
        rows.append(row)
        if not masks:
            missing.append(str(path))

    by_dataset: dict[str, dict[str, int]] = {}
    for row in rows:
        source = row["source_path"]
        if "/ACDC/" in source:
            name = "ACDC"
        elif "/M&M1/" in source:
            name = "M&M1"
        elif "/MnM2/" in source:
            name = "MnM2"
        else:
            name = "unknown"
        stats = by_dataset.setdefault(name, {"total": 0, "with_mask": 0, "missing_mask": 0})
        stats["total"] += 1
        if int(row["num_masks"]) > 0:
            stats["with_mask"] += 1
        else:
            stats["missing_mask"] += 1

    print(
        json.dumps(
            {
                "config": str(root / args.config),
                "num_source_files": len(unique_paths),
                "num_with_mask": len(unique_paths) - len(missing),
                "num_missing_mask": len(missing),
                "by_dataset": by_dataset,
                "examples": rows[: args.max_examples],
                "missing_examples": missing[: args.max_missing],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
