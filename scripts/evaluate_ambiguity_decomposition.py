#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import rankdata, spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from cardiac_nodeo_uq.config import project_root, resolve_path, select_device
from cardiac_nodeo_uq.nodeo_ops import SpatialTransformer3D
from cardiac_nodeo_uq.preprocess import crop_or_pad_around_bbox, load_mask_bbox
from scripts.evaluate_nodeo_anatomy import LABELS, label_metrics, load_masks_by_time


def resolve_source(stored: str, datasets_root: Path, root: Path) -> Path:
    direct = Path(stored)
    if direct.exists():
        return direct
    normalized = stored.replace("\\", "/")
    for dataset in ("ACDC", "M&M1", "MnM2"):
        marker = f"/{dataset}/"
        if marker in normalized:
            candidate = datasets_root / dataset / normalized.split(marker, 1)[1]
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"cannot remap source path: {stored}")


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(bool)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(scores, method="average")
    value = (ranks[labels].sum() - positive * (positive + 1) / 2.0) / (
        positive * negative
    )
    return float(value)


def sparsification_error(errors: np.ndarray, scores: np.ndarray) -> float | None:
    if len(errors) < 10 or float(errors.mean()) <= 0.0:
        return None
    fractions = np.linspace(0.0, 0.9, 19)

    def curve(order: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for fraction in fractions:
            removed = int(round(len(errors) * fraction))
            kept = order[removed:]
            values.append(float(errors[kept].mean()))
        return np.asarray(values) / float(errors.mean())

    uncertainty_order = np.argsort(scores)[::-1]
    oracle_order = np.argsort(errors)[::-1]
    difference = curve(uncertainty_order) - curve(oracle_order)
    widths = fractions[1:] - fractions[:-1]
    return float(np.sum(0.5 * (difference[:-1] + difference[1:]) * widths))


def association(errors: np.ndarray, scores: np.ndarray, threshold_mm: float) -> dict[str, object]:
    if len(errors) == 0:
        return {"samples": 0, "spearman": None, "roc_auc": None, "ause": None}
    correlation = spearmanr(errors, scores).statistic
    high = errors >= float(threshold_mm)
    return {
        "samples": int(len(errors)),
        "spearman": None if not np.isfinite(correlation) else float(correlation),
        "roc_auc": binary_auc(high, scores),
        "ause": sparsification_error(errors, scores),
        "high_error_threshold_mm": float(threshold_mm),
        "high_error_fraction": float(high.mean()),
        "score_mean_low_error": None if high.all() else float(scores[~high].mean()),
        "score_mean_high_error": None if (~high).all() else float(scores[high].mean()),
    }


def surface_samples(
    prediction: np.ndarray,
    target: np.ndarray,
    score_maps: dict[str, np.ndarray],
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prediction_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    if not prediction_surface.any() or not target_surface.any():
        return np.empty(0), {name: np.empty(0) for name in score_maps}
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = distance_transform_edt(~prediction_surface, sampling=spacing)
    errors = np.concatenate(
        [distance_to_target[prediction_surface], distance_to_prediction[target_surface]]
    )
    sampled = {
        name: np.concatenate([values[prediction_surface], values[target_surface]])
        for name, values in score_maps.items()
    }
    return errors, sampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Post-hoc SDE sequence .pt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--high-error-mm", type=float, default=2.0)
    parser.add_argument("--roi-mask-margin", nargs=3, type=int, default=(0, 16, 16))
    args = parser.parse_args()

    root = project_root()
    input_path = resolve_path(args.input, root)
    output_path = resolve_path(args.output, root)
    datasets_root = Path(args.datasets_root)
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    source = resolve_source(str(payload["source_path"]), datasets_root, root)
    masks_by_time = load_masks_by_time(source)
    raw_times = [int(value) for value in payload["raw_time_indices"]]
    reference_time = raw_times[0]
    if reference_time not in masks_by_time:
        raise ValueError(f"reference time {reference_time} has no ED/ES label")

    image_shape = tuple(int(value) for value in payload["images"].shape[-3:])
    bbox = load_mask_bbox(
        source,
        time_axis=-1,
        margin=tuple(args.roi_mask_margin),
        enabled=True,
        require_mask=True,
    )
    cropped_masks = {
        time_index: crop_or_pad_around_bbox(mask, bbox, image_shape).astype(np.int16)
        for time_index, mask in masks_by_time.items()
    }
    zooms = nib.load(str(source)).header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[0]), float(zooms[1]))
    device = select_device(args.device)
    transformer = SpatialTransformer3D(image_shape, mode="nearest").to(device)
    reference = torch.from_numpy(cropped_masks[reference_time]).float()[None, None].to(device)
    displacement = payload["total_displacement"].float().to(device)
    total_sd = payload["deformation_variance_diag"].float().sum(dim=1).clamp_min(0).sqrt().numpy()
    ambiguity = payload["deformation_ambiguity_map"].float().numpy()
    image_quality = payload["image_quality_map"].float().numpy()

    rows: list[dict[str, object]] = []
    pooled_errors: list[np.ndarray] = []
    pooled_scores: dict[str, list[np.ndarray]] = {
        "deformation_ambiguity": [],
        "total_deformation_sd": [],
        "image_quality": [],
    }
    for sequence_index, raw_time in enumerate(raw_times):
        if raw_time == reference_time or raw_time not in cropped_masks:
            continue
        with torch.no_grad():
            prediction = transformer(
                reference,
                displacement[sequence_index : sequence_index + 1],
            )[0, 0].round().cpu().numpy().astype(np.int16)
        target = cropped_masks[raw_time]
        scores = {
            "deformation_ambiguity": ambiguity[sequence_index],
            "total_deformation_sd": total_sd[sequence_index],
            "image_quality": image_quality[sequence_index],
        }
        foreground_errors, foreground_scores = surface_samples(
            prediction > 0,
            target > 0,
            scores,
            spacing,
        )
        pooled_errors.append(foreground_errors)
        for name, values in foreground_scores.items():
            pooled_scores[name].append(values)
        rows.append(
            {
                "sequence_index": sequence_index,
                "raw_time_index": raw_time,
                "labels": {
                    name: label_metrics(prediction, target, label, spacing)
                    for label, name in LABELS.items()
                },
                "uncertainty_association": {
                    name: association(foreground_errors, values, args.high_error_mm)
                    for name, values in foreground_scores.items()
                },
            }
        )
    if not rows:
        raise ValueError(f"no labelled target among raw times; labels={sorted(masks_by_time)}")

    all_errors = np.concatenate(pooled_errors)
    pooled_association = {
        name: association(
            all_errors,
            np.concatenate(values),
            args.high_error_mm,
        )
        for name, values in pooled_scores.items()
    }
    deformation_metrics = pooled_association["deformation_ambiguity"]
    image_metrics = pooled_association["image_quality"]

    def difference(name: str, *, reverse: bool = False) -> float | None:
        deformation_value = deformation_metrics[name]
        image_value = image_metrics[name]
        if deformation_value is None or image_value is None:
            return None
        value = float(deformation_value) - float(image_value)
        return -value if reverse else value

    separation_values = {
        "spearman_gain_over_image_quality": difference("spearman"),
        "roc_auc_gain_over_image_quality": difference("roc_auc"),
        "ause_reduction_vs_image_quality": difference("ause", reverse=True),
    }
    finite_gains = [value for value in separation_values.values() if value is not None]
    report = {
        "input": str(input_path),
        "source_path": str(source),
        "reference_time_index": reference_time,
        "spacing_dhw_mm": spacing,
        "validation_definition": (
            "ED label is propagated to labelled ED/ES targets; surface-distance error "
            "is compared against deformation ambiguity, total deformation SD, and the "
            "separately retained image-quality score."
        ),
        "pooled_uncertainty_association": pooled_association,
        "separation_diagnostic": {
            **separation_values,
            "deformation_evidence_outperforms_image_quality": (
                None
                if not finite_gains
                else sum(value > 0.0 for value in finite_gains)
                >= (len(finite_gains) // 2 + 1)
            ),
        },
        "evaluations": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
