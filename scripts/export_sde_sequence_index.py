#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import project_root, resolve_path
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
    args = parser.parse_args()

    root = project_root()
    h5_path = resolve_path(args.h5, root)
    output = resolve_path(args.output, root)
    refs = build_sequence_refs(h5_path, min_length=args.min_length)

    rng = random.Random(args.seed)
    refs = list(refs)
    rng.shuffle(refs)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for position, ref in enumerate(refs):
            row = {
                "sequence_id": position,
                "split": split_name(position, len(refs), args.val_fraction, args.test_fraction),
                "dataset": ref.dataset,
                "source_path": ref.source_path,
                "indices": list(ref.indices),
                "time_indices": list(ref.time_indices),
                "length": len(ref.indices),
            }
            f.write(json.dumps(row) + "\n")

    print(
        json.dumps(
            {
                "output": str(output),
                "num_sequences": len(refs),
                "min_length": args.min_length,
                "val_fraction": args.val_fraction,
                "test_fraction": args.test_fraction,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
