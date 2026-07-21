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
from scipy.ndimage import binary_erosion, distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nodeo_uq.config import project_root, resolve_path, select_device
from cardiac_nodeo_uq.nodeo_ops import SpatialTransformer3D
from cardiac_nodeo_uq.nodeo_roi_data import resolve_portable_source_path
from cardiac_nodeo_uq.preprocess import candidate_mask_paths, crop_or_pad_around_bbox, load_mask_bbox


LABELS = {1: "RV", 2: "MYO", 3: "LV"}


def nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def to_dhw(array: np.ndarray) -> np.ndarray:
    array = np.squeeze(np.asarray(array))
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"expected 3D array, got {array.shape}")
    return np.moveaxis(array, -1, 0)


def normalized_mse(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    left = (left - left.mean()) / max(float(left.std()), 1.0e-6)
    right = (right - right.mean()) / max(float(right.std()), 1.0e-6)
    return float(np.square(left - right).mean())


def companion_image(mask_path: Path) -> Path | None:
    stem = nifti_stem(mask_path)
    if not stem.endswith("_gt"):
        return None
    base = stem[:-3]
    for suffix in (".nii.gz", ".nii"):
        candidate = mask_path.parent / f"{base}{suffix}"
        if candidate.exists():
            return candidate
    return None


def match_3d_mask_time(mask_path: Path, source_4d: np.ndarray) -> int:
    frame_match = re.search(r"_frame(\d+)_gt", mask_path.name, flags=re.IGNORECASE)
    if frame_match:
        return int(frame_match.group(1)) - 1
    companion = companion_image(mask_path)
    if companion is None:
        raise ValueError(f"cannot determine cine time for mask {mask_path}")
    companion_array = np.asarray(nib.load(str(companion)).get_fdata(dtype=np.float32)).squeeze()
    if companion_array.ndim == 4 and companion.resolve() == mask_path.resolve():
        raise ValueError(f"mask companion is not a 3D ED/ES image: {companion}")
    scores = [normalized_mse(source_4d[..., index], companion_array) for index in range(source_4d.shape[-1])]
    return int(np.argmin(scores))


def load_masks_by_time(source_path: Path) -> dict[int, np.ndarray]:
    source = np.asarray(nib.load(str(source_path)).get_fdata(dtype=np.float32))
    if source.ndim != 4:
        raise ValueError(f"expected 4D cine source, got {source.shape} from {source_path}")
    masks: dict[int, np.ndarray] = {}
    for mask_path in candidate_mask_paths(source_path):
        array = np.asarray(nib.load(str(mask_path)).get_fdata(dtype=np.float32))
        if array.ndim == 4 and array.shape[-1] == source.shape[-1]:
            for time_index in range(array.shape[-1]):
                frame = to_dhw(array[..., time_index])
                if np.any(frame > 0):
                    masks[time_index] = frame
            continue
        time_index = match_3d_mask_time(mask_path, source)
        masks[time_index] = to_dhw(array)
    return masks


def surface_distances(left: np.ndarray, right: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    left_surface = left ^ binary_erosion(left)
    right_surface = right ^ binary_erosion(right)
    if not left_surface.any() or not right_surface.any():
        return np.asarray([], dtype=np.float64)
    distance_to_right = distance_transform_edt(~right_surface, sampling=spacing)
    distance_to_left = distance_transform_edt(~left_surface, sampling=spacing)
    return np.concatenate([distance_to_right[left_surface], distance_to_left[right_surface]])


def label_metrics(prediction: np.ndarray, target: np.ndarray, label: int, spacing: tuple[float, float, float]) -> dict:
    pred = prediction == label
    truth = target == label
    denominator = int(pred.sum() + truth.sum())
    dice = 1.0 if denominator == 0 else 2.0 * float(np.logical_and(pred, truth).sum()) / denominator
    distances = surface_distances(pred, truth, spacing)
    return {
        "dice": dice,
        "mcd_mm": None if distances.size == 0 else float(distances.mean()),
        "hd95_mm": None if distances.size == 0 else float(np.percentile(distances, 95.0)),
        "pred_voxels": int(pred.sum()),
        "target_voxels": int(truth.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Per-sequence NODEO .pt output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    args = parser.parse_args()

    root = project_root()
    payload = torch.load(resolve_path(args.input, root), map_location="cpu", weights_only=False)
    source_path = resolve_portable_source_path(
        str(payload["source_path"]),
        datasets_root=args.datasets_root,
        project_root=root,
    )
    masks_by_time = load_masks_by_time(source_path)
    raw_times = [int(v) for v in payload["raw_time_indices"]]
    reference_time = raw_times[0]
    if reference_time not in masks_by_time:
        raise ValueError(
            f"reference frame time={reference_time} has no segmentation; available={sorted(masks_by_time)}"
        )

    image_shape = tuple(int(v) for v in payload["images"].shape[-3:])
    bbox = load_mask_bbox(
        source_path,
        time_axis=-1,
        margin=(0, 16, 16),
        enabled=True,
        require_mask=True,
    )
    cropped_masks = {
        time_index: crop_or_pad_around_bbox(mask, bbox, image_shape).astype(np.int16)
        for time_index, mask in masks_by_time.items()
    }
    zooms = nib.load(str(source_path)).header.get_zooms()[:3]
    spacing_dhw = (float(zooms[2]), float(zooms[0]), float(zooms[1]))
    reference_mask = torch.from_numpy(cropped_masks[reference_time]).float()[None, None]
    displacement = payload["displacement"].float()
    device = select_device(args.device)
    transformer = SpatialTransformer3D(image_shape, mode="nearest").to(device)

    evaluations: list[dict] = []
    for sequence_index, raw_time in enumerate(raw_times):
        if raw_time == reference_time or raw_time not in cropped_masks:
            continue
        with torch.no_grad():
            warped = transformer(
                reference_mask.to(device),
                displacement[sequence_index : sequence_index + 1].to(device),
            ).cpu()[0, 0].round().numpy().astype(np.int16)
        target = cropped_masks[raw_time]
        evaluations.append(
            {
                "sequence_index": sequence_index,
                "raw_time_index": raw_time,
                "labels": {
                    name: label_metrics(warped, target, label, spacing_dhw)
                    for label, name in LABELS.items()
                },
            }
        )
    if not evaluations:
        raise ValueError(f"no labelled target frame found; available mask times={sorted(masks_by_time)}")

    report = {
        "input": str(args.input),
        "dataset": payload["dataset"],
        "source_path": str(source_path),
        "reference_time_index": reference_time,
        "spacing_dhw_mm": spacing_dhw,
        "evaluations": evaluations,
    }
    output = resolve_path(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
