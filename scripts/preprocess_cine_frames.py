#!/usr/bin/env python3
"""Preprocess cine-MRI volumes into cached 3D tensor frames for faster training."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_cunsure.data import VolumeFrameDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, nargs="+", help="One or more raw dataset roots/files")
    parser.add_argument("--output", required=True, help="Output folder for cached .pt frames")
    parser.add_argument("--npz-key", default=None)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--spatial-dims", type=int, choices=[3], default=3)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--depth-size", type=int, default=16)
    parser.add_argument("--time-axis", type=int, default=0)
    parser.add_argument("--frame-layout", default="dhw")
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def safe_stem(path: Path) -> str:
    name = path.name
    name = re.sub(r"\.nii(\.gz)?$|\.pt$|\.pth$|\.npy$|\.npz$", "", name, flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def resize_frame(frame: torch.Tensor, image_size: int, depth_size: int) -> torch.Tensor:
    return F.interpolate(
        frame.unsqueeze(0),
        size=(image_size, image_size, depth_size),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    manifest: dict[str, object] = {
        "data": args.data,
        "image_size": args.image_size,
        "depth_size": args.depth_size,
        "time_axis": args.time_axis,
        "frame_layout": args.frame_layout,
        "include": args.include,
        "exclude": args.exclude,
        "dtype": args.dtype,
        "roots": [],
    }

    total_saved = 0
    for data_root in args.data:
        dataset = VolumeFrameDataset(
            data_root,
            channels=args.channels,
            npz_key=args.npz_key,
            normalize=True,
            time_axis=args.time_axis,
            frame_layout=args.frame_layout,
            limit=args.limit,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
        )
        root_record = {"root": data_root, "files": len(dataset.paths), "frames": len(dataset)}
        manifest["roots"].append(root_record)  # type: ignore[index]
        print(f"loaded root={data_root} files={len(dataset.paths)} frames={len(dataset)}", flush=True)

        for idx, (path, frame_index) in enumerate(dataset.index):
            frame = dataset[idx]
            frame = resize_frame(frame, image_size=args.image_size, depth_size=args.depth_size).to(dtype).contiguous()
            out_name = f"{safe_stem(path)}_frame{0 if frame_index is None else frame_index:04d}.pt"
            out_path = output / out_name
            if out_path.exists() and not args.overwrite:
                continue
            torch.save(
                {
                    "frame": frame,
                    "source": str(path),
                    "frame_index": frame_index,
                    "shape": tuple(frame.shape),
                },
                out_path,
            )
            total_saved += 1
            if args.log_every > 0 and total_saved % args.log_every == 0:
                print(f"saved={total_saved} last={out_path.name}", flush=True)

    manifest["saved_frames"] = total_saved
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"saved_frames={total_saved}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
