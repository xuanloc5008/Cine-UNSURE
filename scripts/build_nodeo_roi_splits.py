#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import load_yaml, project_root, resolve_path
from cunsure_monai3d.nodeo_roi_data import canonical_source_key, decode_h5_string


def classify_source(path: str) -> tuple[str, str]:
    normalized = path.replace("\\", "/").lower()
    if "/acdc/" in normalized or "/acdcdata/" in normalized:
        subset = "training" if "/training/" in normalized else "testing" if "/testing/" in normalized else "unknown"
        return "ACDC", subset
    if "/m&m1/" in normalized or "/mm1" in normalized:
        for subset in ("training", "validation", "testing"):
            if f"/{subset}/" in normalized:
                return "M&M1", subset
        return "M&M1", "unknown"
    if "/mnm2/" in normalized or "m-and-m-candy" in normalized:
        return "MnM2", "all"
    raise ValueError(f"cannot classify dataset from source_path: {path}")


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def scan_h5(path: Path, root: Path, *, expected_shape: tuple[int, int, int]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with h5py.File(path, "r") as h5:
        required = {"y", "source_path", "time_index"}
        missing = required - set(h5.keys())
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        if h5["y"].shape[0] != h5["source_path"].shape[0] or h5["y"].shape[0] != h5["time_index"].shape[0]:
            raise ValueError(f"inconsistent row counts in {path}")
        if tuple(int(v) for v in h5["y"].shape[2:]) != expected_shape:
            raise ValueError(f"expected ROI shape {expected_shape}, got {h5['y'].shape[2:]} in {path}")
        if not bool(h5.attrs.get("roi_mask_crop", False)):
            raise ValueError(f"{path} is not marked as ROI-cropped")
        for idx, (source, time_index) in enumerate(zip(h5["source_path"], h5["time_index"], strict=True)):
            grouped[decode_h5_string(source)].append((idx, int(time_index)))

    rows: dict[str, dict[str, object]] = {}
    for source, values in grouped.items():
        values.sort(key=lambda item: item[1])
        dataset, official_subset = classify_source(source)
        rows[source] = {
            "dataset": dataset,
            "official_subset": official_subset,
            "source_path": source,
            "h5_path": relative_to_root(path, root),
            "indices": [idx for idx, _ in values],
            "time_indices": [time for _, time in values],
        }
    return rows


def shuffled_split(items: list[str], fractions: tuple[float, ...], seed: int) -> list[list[str]]:
    order = sorted(items)
    random.Random(seed).shuffle(order)
    counts = [int(round(len(order) * fraction)) for fraction in fractions[:-1]]
    counts.append(len(order) - sum(counts))
    output: list[list[str]] = []
    start = 0
    for count in counts:
        output.append(order[start : start + count])
        start += count
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nodeo_roi_splits.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    data_cfg = cfg["data"]
    seed = int(cfg.get("seed", 2026))
    sequences: dict[str, dict[str, object]] = {}
    expected_shape = tuple(int(v) for v in data_cfg["volume_size"])
    for configured_path in data_cfg["roi_h5_files"]:
        path = resolve_path(configured_path, root)
        for source, row in scan_h5(path, root, expected_shape=expected_shape).items():
            if source in sequences:
                raise ValueError(f"source appears in multiple ROI H5 files: {source}")
            sequences[source] = row

    pools: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source, row in sequences.items():
        pools[(str(row["dataset"]), str(row["official_subset"]))].append(source)

    assignments: dict[str, str] = {}
    acdc_train, acdc_val = shuffled_split(pools[("ACDC", "training")], (0.80, 0.20), seed + 1)
    for source in acdc_train:
        assignments[source] = "train"
    for source in acdc_val:
        assignments[source] = "val"
    for source in pools[("ACDC", "testing")]:
        assignments[source] = "test"

    for source in pools[("M&M1", "training")]:
        assignments[source] = "train"
    for source in pools[("M&M1", "validation")]:
        assignments[source] = "val"
    for source in pools[("M&M1", "testing")]:
        assignments[source] = "test"

    mnm2_train, mnm2_val, mnm2_test = shuffled_split(pools[("MnM2", "all")], (0.70, 0.10, 0.20), seed + 3)
    for source in mnm2_train:
        assignments[source] = "train"
    for source in mnm2_val:
        assignments[source] = "val"
    for source in mnm2_test:
        assignments[source] = "test"

    unknown = sorted(set(sequences) - set(assignments))
    if unknown:
        raise ValueError(f"unassigned sources ({len(unknown)}), examples: {unknown[:3]}")

    output_path = resolve_path(data_cfg["manifest"], root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for source in sorted(sequences):
        row = dict(sequences[source])
        row["split"] = assignments[source]
        row["sequence_id"] = hashlib.sha1(canonical_source_key(source).encode("utf-8")).hexdigest()[:16]
        rows.append(row)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    counts = Counter((str(row["dataset"]), str(row["split"])) for row in rows)
    report = {
        "manifest": str(output_path),
        "num_sequences": len(rows),
        "counts": {f"{dataset}/{split}": count for (dataset, split), count in sorted(counts.items())},
        "rules": {
            "ACDC": "80% official training -> train; 20% official training -> val; official testing -> test",
            "M&M1": "official Training -> train; official Validation -> val; official Testing -> test",
            "MnM2": "70% train; 10% val; 20% test",
        },
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
