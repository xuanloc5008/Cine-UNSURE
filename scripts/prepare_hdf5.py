#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import as_tuple_int, load_yaml, project_root, resolve_path
from cunsure_monai3d.preprocess import scan_nifti_frames, write_hdf5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prepare_hdf5.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    data = cfg["data"]
    volume_size = as_tuple_int(data["volume_size"], name="volume_size")
    exclude = list(data["exclude_substrings"])

    train_refs = scan_nifti_frames(
        root,
        list(data["train_globs"]),
        exclude,
        split_4d=bool(data["split_4d"]),
        time_axis=int(data["time_axis"]),
    )
    val_refs = scan_nifti_frames(
        root,
        list(data["val_globs"]),
        exclude,
        split_4d=bool(data["split_4d"]),
        time_axis=int(data["time_axis"]),
    )
    print(f"found train frames: {len(train_refs)}")
    print(f"found val frames: {len(val_refs)}")

    write_hdf5(
        train_refs,
        resolve_path(data["train_output"], root),
        volume_size=volume_size,
        channels=int(data["channels"]),
        normalize=str(data["normalize"]),
        percentile_low=float(data["percentile_low"]),
        percentile_high=float(data["percentile_high"]),
        time_axis=int(data["time_axis"]),
        limit=data.get("train_limit"),
        compression=data.get("compression", "lzf"),
    )
    write_hdf5(
        val_refs,
        resolve_path(data["val_output"], root),
        volume_size=volume_size,
        channels=int(data["channels"]),
        normalize=str(data["normalize"]),
        percentile_low=float(data["percentile_low"]),
        percentile_high=float(data["percentile_high"]),
        time_axis=int(data["time_axis"]),
        limit=data.get("val_limit"),
        compression=data.get("compression", "lzf"),
    )


if __name__ == "__main__":
    main()
