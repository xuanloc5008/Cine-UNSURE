#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nodeo_uq.config import load_yaml, project_root, resolve_path


def decode(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def dataset_name(source: str) -> str:
    parts = set(Path(source).parts)
    if "ACDC" in parts:
        return "ACDC"
    if "M&M1" in parts:
        return "M&M1"
    if "MnM2" in parts:
        return "MnM2"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acdc/prepare_hdf5.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    data = cfg["data"]
    report: dict[str, object] = {"splits": {}}
    sources_by_split: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        path = resolve_path(data[f"{split}_output"], root)
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as h5:
            sources = [decode(value) for value in h5["source_path"][:]]
            shape = list(h5["y"].shape)
            attrs = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in h5.attrs.items()}
        unique_sources = set(sources)
        sources_by_split[split] = unique_sources
        report["splits"][split] = {
            "path": str(path),
            "shape": shape,
            "frames_by_dataset": dict(Counter(dataset_name(source) for source in sources)),
            "sources_by_dataset": dict(Counter(dataset_name(source) for source in unique_sources)),
            "attrs": attrs,
        }

    overlaps = {}
    names = list(sources_by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            common = sorted(sources_by_split[left] & sources_by_split[right])
            overlaps[f"{left}_vs_{right}"] = {"count": len(common), "examples": common[:5]}
    report["source_overlaps"] = overlaps
    report["passed"] = all(item["count"] == 0 for item in overlaps.values())
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
