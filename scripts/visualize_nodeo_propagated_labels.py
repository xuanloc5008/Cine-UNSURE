#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cardiac_nodeo_uq.label_propagation import propagate_label_probabilities
from cardiac_nodeo_uq.nodeo_ops import SpatialTransformer3D
from cardiac_nodeo_uq.preprocess import crop_or_pad_around_bbox, load_mask_bbox
from scripts.evaluate_nodeo_anatomy import load_masks_by_time


COLORS = {1: "#ff3b30", 2: "#34c759", 3: "#00c7ff"}


def resolve_source(stored_source: str, datasets_root: Path) -> Path:
    source = Path(stored_source)
    if source.exists():
        return source
    normalized = source.as_posix()
    marker = "/ACDC/"
    if marker not in normalized:
        raise FileNotFoundError(f"cannot remap ACDC source: {stored_source}")
    candidate = datasets_root / "ACDC" / normalized.split(marker, 1)[1]
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def add_contours(axis: object, mask: np.ndarray) -> None:
    for label, color in COLORS.items():
        binary = mask == label
        if binary.any():
            axis.contour(binary.astype(np.float32), levels=[0.5], colors=[color], linewidths=1.2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--slice-index", type=int)
    parser.add_argument("--save-labels")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    source_path = resolve_source(str(payload["source_path"]), Path(args.datasets_root))
    images = payload["images"].float()
    displacement = payload["displacement"].float().to(device)
    stored_warped = payload["warped"].float()
    predicted = torch.cat([images[0:1], stored_warped], dim=0)
    raw_times = [int(value) for value in payload["raw_time_indices"]]
    image_shape = tuple(int(value) for value in images.shape[-3:])

    masks_by_time = load_masks_by_time(source_path)
    reference_time = raw_times[0]
    if reference_time not in masks_by_time:
        raise ValueError(f"fixed frame has no label; labelled times={sorted(masks_by_time)}")
    bbox = load_mask_bbox(
        source_path, time_axis=-1, margin=(0, 16, 16), enabled=True, require_mask=True
    )
    reference_np = crop_or_pad_around_bbox(masks_by_time[reference_time], bbox, image_shape).astype(np.int16)
    reference = torch.from_numpy(reference_np).float()[None, None].to(device)

    nearest = SpatialTransformer3D(image_shape, mode="nearest").to(device)
    with torch.no_grad():
        raw_labels = nearest(reference.expand(displacement.shape[0], -1, -1, -1, -1), displacement)
        raw_labels = raw_labels.round().long().cpu()[:, 0]
        smooth_labels, probabilities = propagate_label_probabilities(
            reference,
            displacement,
            num_classes=4,
            sigma_inplane=float(args.sigma),
            kernel_size=int(args.kernel_size),
        )
        smooth_labels = smooth_labels.cpu()[:, 0]
        probabilities = probabilities.cpu()

    slice_index = (
        int(np.argmax((reference_np > 0).sum(axis=(1, 2))))
        if args.slice_index is None
        else int(args.slice_index)
    )
    frames: list[Image.Image] = []
    for frame_index in range(images.shape[0]):
        fixed = images[0, 0, slice_index].numpy()
        prediction = predicted[frame_index, 0, slice_index].numpy()
        target = images[frame_index, 0, slice_index].numpy()
        error = np.abs(prediction - target)
        raw_mask = raw_labels[frame_index, slice_index].numpy()
        smooth_mask = smooth_labels[frame_index, slice_index].numpy()

        figure, axes = plt.subplots(1, 5, figsize=(17, 4), dpi=100)
        axes[0].imshow(fixed, cmap="gray", vmin=0, vmax=1)
        add_contours(axes[0], reference_np[slice_index])
        axes[0].set_title("Fixed + label")
        axes[1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        add_contours(axes[1], raw_mask)
        axes[1].set_title("Nearest label")
        axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        add_contours(axes[2], smooth_mask)
        axes[2].set_title(f"Soft warp + smooth (sigma={args.sigma:g})")
        axes[3].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("Target")
        axes[4].imshow(error, cmap="magma", vmin=0)
        axes[4].set_title("|Predicted - Target|")
        for axis in axes:
            axis.axis("off")
        figure.suptitle(
            f"{payload['dataset']} | slice={slice_index} | time={raw_times[frame_index]} | "
            f"frame={frame_index + 1}/{images.shape[0]}"
        )
        figure.tight_layout()
        figure.canvas.draw()
        rgb = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(Image.fromarray(rgb))
        plt.close(figure)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / float(args.fps))),
        loop=0,
        optimize=False,
    )
    if args.save_labels:
        labels_output = Path(args.save_labels)
        labels_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "raw_labels": raw_labels,
                "smoothed_labels": smooth_labels,
                "smoothed_probabilities": probabilities.half(),
                "sigma_inplane": float(args.sigma),
                "kernel_size": int(args.kernel_size),
            },
            labels_output,
        )
    print(f"source: {source_path}")
    print(f"slice: {slice_index}")
    print(f"saved GIF: {output}")
    if args.save_labels:
        print(f"saved labels: {args.save_labels}")


if __name__ == "__main__":
    main()
