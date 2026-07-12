#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import as_tuple_int, load_yaml, project_root, resolve_path
from cunsure_monai3d.preprocess import FrameRef, scan_nifti_frames, write_hdf5


def scan(data: dict, root: Path, key: str, exclude: list[str]) -> list[FrameRef]:
    return scan_nifti_frames(
        root,
        list(data.get(key, [])),
        exclude,
        split_4d=bool(data["split_4d"]),
        time_axis=int(data["time_axis"]),
    )


def split_mnm2_by_source(
    refs: list[FrameRef], *, test_fraction: float, seed: int
) -> tuple[list[FrameRef], list[FrameRef]]:
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("mnm2_test_fraction must be in [0, 1)")
    source_paths = sorted({ref.path for ref in refs})
    shuffled = source_paths.copy()
    random.Random(seed).shuffle(shuffled)
    test_count = int(round(len(shuffled) * test_fraction))
    test_sources = set(shuffled[:test_count])
    train_refs = [ref for ref in refs if ref.path not in test_sources]
    test_refs = [ref for ref in refs if ref.path in test_sources]
    return train_refs, test_refs


def assert_disjoint_splits(splits: dict[str, list[FrameRef]]) -> None:
    sources = {name: {ref.path for ref in refs} for name, refs in splits.items()}
    names = list(sources)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sources[left] & sources[right]
            if overlap:
                examples = ", ".join(str(path) for path in sorted(overlap)[:3])
                raise ValueError(f"source leakage between {left} and {right}: {examples}")


def write_split(
    refs: list[FrameRef], data: dict, root: Path, name: str, volume_size: tuple[int, int, int]
) -> None:
    output_key = f"{name}_output"
    output = data.get(output_key)
    if not output:
        print(f"skip {name}: {output_key} is not set")
        return
    write_hdf5(
        refs,
        resolve_path(output, root),
        volume_size=volume_size,
        channels=int(data["channels"]),
        normalize=str(data["normalize"]),
        percentile_low=float(data["percentile_low"]),
        percentile_high=float(data["percentile_high"]),
        time_axis=int(data["time_axis"]),
        limit=data.get(f"{name}_limit"),
        compression=data.get("compression", "lzf"),
        roi_mask_crop=bool(data.get("roi_mask_crop", False)),
        roi_mask_margin=as_tuple_int(data.get("roi_mask_margin", [0, 12, 12]), name="roi_mask_margin"),
        require_roi_mask=bool(data.get("require_roi_mask", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prepare_hdf5.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    data = cfg["data"]
    volume_size = as_tuple_int(data["volume_size"], name="volume_size")
    exclude = list(data["exclude_substrings"])

    train_refs = scan(data, root, "train_globs", exclude)
    val_refs = scan(data, root, "val_globs", exclude)
    test_refs = scan(data, root, "test_globs", exclude)
    mnm2_refs = scan(data, root, "mnm2_globs", exclude)
    mnm2_train, mnm2_test = split_mnm2_by_source(
        mnm2_refs,
        test_fraction=float(data.get("mnm2_test_fraction", 0.30)),
        seed=int(cfg.get("seed", 2026)),
    )
    train_refs.extend(mnm2_train)
    test_refs.extend(mnm2_test)
    splits = {"train": train_refs, "val": val_refs, "test": test_refs}
    assert_disjoint_splits(splits)

    print(f"found train frames: {len(train_refs)}")
    print(f"found val frames: {len(val_refs)}")
    print(f"found test frames: {len(test_refs)}")
    print(f"MnM2 source split: train={len({r.path for r in mnm2_train})}, test={len({r.path for r in mnm2_test})}")

    for name, refs in splits.items():
        write_split(refs, data, root, name, volume_size)


if __name__ == "__main__":
    main()
