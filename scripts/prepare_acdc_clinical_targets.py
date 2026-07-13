#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import load_yaml, project_root, resolve_path
from cunsure_monai3d.sde_data import build_sequence_refs


def local_source(source: str, root: Path) -> Path:
    path = Path(source)
    if path.exists():
        return path
    normalized = source.replace("\\", "/")
    marker = "/datasets/"
    if marker in normalized:
        candidate = root / "datasets" / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    marker = "/ACDC/"
    if marker in normalized:
        candidate = root / "datasets" / "ACDC" / normalized.split(marker, 1)[1]
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve source path: {source}")


def read_acdc_phase_indices(patient_dir: Path) -> tuple[int, int]:
    info = patient_dir / "Info.cfg"
    text = info.read_text(encoding="utf-8")
    ed_match = re.search(r"^ED\s*:\s*(\d+)", text, flags=re.MULTILINE)
    es_match = re.search(r"^ES\s*:\s*(\d+)", text, flags=re.MULTILINE)
    if ed_match is None or es_match is None:
        raise ValueError(f"cannot parse ED/ES from {info}")
    return int(ed_match.group(1)), int(es_match.group(1))


def lv_volume_ml(mask_path: Path, label: int) -> float:
    image = nib.load(str(mask_path))
    mask = np.asarray(image.get_fdata())
    voxel_volume_ml = float(np.prod(image.header.get_zooms()[:3])) / 1000.0
    return float(np.count_nonzero(np.isclose(mask, float(label)))) * voxel_volume_ml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_sde_rnn_uncertainty.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", type=int, default=3, help="ACDC LV cavity label")
    parser.add_argument("--checkpoint")
    parser.add_argument("--sequence-index-file")
    parser.add_argument("--h5")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(resolve_path(args.config, root))
    h5_path = args.h5 or cfg["data"]["h5"]
    refs = build_sequence_refs(resolve_path(h5_path, root), min_length=int(cfg["data"].get("min_length", 2)))
    split_indices: list[int]
    if args.sequence_index_file:
        split_indices = []
        with resolve_path(args.sequence_index_file, root).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") == args.split:
                    split_indices.append(int(row["sequence_id"]))
    elif args.checkpoint:
        checkpoint = torch.load(resolve_path(args.checkpoint, root), map_location="cpu", weights_only=False)
        selection = checkpoint["config"].get("pilot_selection", {})
        index_key = f"{args.split}_indices"
        if index_key not in selection:
            raise KeyError(f"checkpoint does not contain {index_key}; pass --sequence-index-file for full mode")
        split_indices = [int(index) for index in selection[index_key]]
    else:
        parser.error("either --checkpoint or --sequence-index-file is required")

    rows = []
    for sequence_index in split_indices:
        ref = refs[sequence_index]
        if ref.dataset != "ACDC":
            continue
        source = local_source(ref.source_path, root)
        ed_raw, es_raw = read_acdc_phase_indices(source.parent)
        stem = source.name.replace("_4d.nii.gz", "").replace("_4d.nii", "")
        ed_mask = source.parent / f"{stem}_frame{ed_raw:02d}_gt.nii.gz"
        es_mask = source.parent / f"{stem}_frame{es_raw:02d}_gt.nii.gz"
        if not ed_mask.exists() or not es_mask.exists():
            raise FileNotFoundError(f"missing ACDC ED/ES masks for {source}")
        raw_times = list(ref.time_indices)
        ed_index = raw_times.index(ed_raw - 1)
        es_index = raw_times.index(es_raw - 1)
        ed_volume = lv_volume_ml(ed_mask, int(args.label))
        es_volume = lv_volume_ml(es_mask, int(args.label))
        rows.append(
            {
                "sequence_index": sequence_index,
                "source_path": ref.source_path,
                "resolved_source_path": str(source),
                "reference_mask": str(ed_mask),
                "ed_index": ed_index,
                "es_index": es_index,
                "ed_volume_ml_gt": ed_volume,
                "es_volume_ml_gt": es_volume,
                "ef": (ed_volume - es_volume) / max(ed_volume, 1.0e-6),
            }
        )
    if not rows:
        raise ValueError(f"{args.split} split contains no ACDC sequences with clinical labels")
    output = resolve_path(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"output": str(output), "split": args.split, "split_sequences": len(split_indices), "acdc_labelled_sequences": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
