#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import project_root, resolve_path
from cunsure_monai3d.nodeo_roi_data import canonical_source_key, load_nodeo_manifest
from cunsure_monai3d.sde_data import build_sequence_refs


def split_name(position: int, total: int, val_fraction: float, test_fraction: float) -> str:
    if total <= 0:
        return "train"
    frac = position / total
    if frac < test_fraction:
        return "test"
    if frac < test_fraction + val_fraction:
        return "val"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--split-manifest")
    args = parser.parse_args()

    root = project_root()
    h5_path = resolve_path(args.h5, root)
    output = resolve_path(args.output, root)
    refs = build_sequence_refs(h5_path, min_length=args.min_length)

    refs = list(refs)
    split_by_source: dict[str, str] | None = None
    if args.split_manifest:
        manifest = load_nodeo_manifest(resolve_path(args.split_manifest, root))
        split_by_source = {
            canonical_source_key(row.source_path): row.split
            for row in manifest
            if row.split in {"train", "val", "test"}
        }
    else:
        order = list(range(len(refs)))
        random.Random(args.seed).shuffle(order)
        random_split = {
            ref_index: split_name(position, len(refs), args.val_fraction, args.test_fraction)
            for position, ref_index in enumerate(order)
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        written = 0
        for sequence_index, ref in enumerate(refs):
            if split_by_source is not None:
                key = canonical_source_key(ref.source_path)
                if key not in split_by_source:
                    continue
                split = split_by_source[key]
            else:
                split = random_split[sequence_index]
            row = {
                "sequence_id": sequence_index,
                "split": split,
                "dataset": ref.dataset,
                "source_path": ref.source_path,
                "indices": list(ref.indices),
                "time_indices": list(ref.time_indices),
                "length": len(ref.indices),
            }
            f.write(json.dumps(row) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "output": str(output),
                "num_sequences": written,
                "min_length": args.min_length,
                "split_manifest": args.split_manifest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
