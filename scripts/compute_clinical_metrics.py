#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from statistics import NormalDist
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nodeo_uq.clinical_metrics import (
    delta_variance_diag,
    delta_variance_factor,
    ejection_fraction,
    mean_green_lagrange_strain,
    mean_wall_motion_mm,
    resize_mask,
    volume_from_deformation,
)
from cardiac_nodeo_uq.config import project_root, resolve_path
from cardiac_nodeo_uq.preprocess import (
    bbox_from_mask,
    center_crop_or_pad,
    crop_or_pad_around_bbox,
    extract_frame_array,
    mask_to_dhw,
)


def parse_labels(value: str | None) -> list[float] | None:
    if value is None or value == "":
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_spacing(value: str | None) -> tuple[float, float, float] | None:
    if value is None or value == "":
        return None
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("--spacing-mm must contain three comma-separated values in D,H,W order")
    return tuple(parts)  # type: ignore[return-value]


def nifti_spacing_dhw_mm(path: Path) -> tuple[float, float, float]:
    img = nib.load(str(path))
    zooms = tuple(float(v) for v in img.header.get_zooms())
    if len(zooms) >= 3:
        return (zooms[2], zooms[0], zooms[1])
    if len(zooms) == 2:
        return (1.0, zooms[0], zooms[1])
    raise ValueError(f"cannot infer spatial spacing from NIfTI header: {path}")


def effective_output_spacing_mm(
    input_spacing_dhw_mm: tuple[float, float, float],
    *,
    volume_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
) -> tuple[float, float, float]:
    return tuple(
        float(spacing) * float(input_dim) / float(output_dim)
        for spacing, input_dim, output_dim in zip(input_spacing_dhw_mm, volume_size, output_size, strict=True)
    )


def load_reference_mask(
    path: Path,
    *,
    time_index: int | None,
    time_axis: int,
    volume_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    labels: list[float] | None,
    roi_mask_crop: bool,
    roi_mask_margin: tuple[int, int, int],
) -> torch.Tensor:
    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    vol = extract_frame_array(arr, time_index, time_axis=time_axis, path=path)
    bbox = None
    if roi_mask_crop:
        union = None
        candidates = [
            candidate
            for candidate in sorted(path.parent.glob("*_gt*"))
            if candidate.name.endswith(".nii") or candidate.name.endswith(".nii.gz")
        ]
        for candidate in candidates:
            candidate_mask = mask_to_dhw(
                nib.load(str(candidate)).get_fdata(dtype=np.float32),
                time_axis=time_axis,
                path=candidate,
            )
            union = candidate_mask if union is None else np.logical_or(union, candidate_mask)
        if union is None:
            raise FileNotFoundError(f"no sibling ROI masks found next to {path}")
        bbox = bbox_from_mask(union, roi_mask_margin)
    vol = crop_or_pad_around_bbox(vol, bbox, volume_size) if roi_mask_crop else center_crop_or_pad(vol, volume_size)
    mask = torch.from_numpy(vol).float()
    if labels:
        binary = torch.zeros_like(mask)
        for label in labels:
            binary = torch.logical_or(binary.bool(), torch.isclose(mask, torch.tensor(label, dtype=mask.dtype))).float()
        mask = binary.float()
    else:
        mask = (mask > 0).float()
    return resize_mask(mask, output_size)


def variance_to_float(value: torch.Tensor | None) -> float | None:
    return None if value is None else float(value.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deformation", required=True)
    parser.add_argument("--reference-mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-time-index", type=int, default=None)
    parser.add_argument("--time-axis", type=int, default=-1)
    parser.add_argument("--volume-size", nargs=3, type=int, default=[16, 128, 128])
    parser.add_argument("--labels", default=None, help="comma-separated labels to include; default uses mask > 0")
    parser.add_argument("--spacing-mm", default=None, help="optional D,H,W spacing after crop/pad, before resize")
    parser.add_argument("--voxel-volume", type=float, default=None, help="optional output voxel volume in mm^3")
    parser.add_argument("--ed-index", type=int, default=0)
    parser.add_argument("--es-index", type=int, default=-1)
    parser.add_argument("--coverage", type=float, default=0.95)
    parser.add_argument("--roi-mask-crop", action="store_true")
    parser.add_argument("--roi-mask-margin", nargs=3, type=int, default=[0, 16, 16])
    parser.add_argument(
        "--displacement-key",
        default="auto",
        choices=("auto", "nodeo_displacement", "total_displacement", "residual_displacement", "displacement"),
    )
    args = parser.parse_args()

    root = project_root()
    deformation_path = resolve_path(args.deformation, root)
    output_path = resolve_path(args.output, root)
    mask_path = Path(args.reference_mask)
    if not mask_path.is_absolute():
        mask_path = root / mask_path

    item = torch.load(deformation_path, map_location="cpu", weights_only=False)
    if args.displacement_key == "auto":
        displacement_key = "displacement" if "displacement" in item else "total_displacement"
    else:
        displacement_key = args.displacement_key
    if displacement_key not in item:
        raise KeyError(f"deformation file has no displacement key: {displacement_key}")
    displacement_all = item[displacement_key].float()
    cov_diag_all = item.get("deformation_covariance_diag")
    if cov_diag_all is not None:
        cov_diag_all = cov_diag_all.float().clamp_min(0)
    cov_factor_all = item.get("deformation_covariance_factor")
    if cov_factor_all is not None:
        cov_factor_all = cov_factor_all.float()
    motion_basis = item.get("motion_basis")
    motion_factor_all = item.get("motion_covariance_factor")
    if motion_basis is not None and motion_factor_all is not None:
        motion_basis = motion_basis.float()
        motion_factor_all = motion_factor_all.float()
        expected_voxels = int(displacement_all[0].numel())
        if motion_basis.ndim != 2 or int(motion_basis.shape[0]) != expected_voxels:
            raise ValueError(
                "motion_basis must be [3*D*H*W, rank], got "
                f"{tuple(motion_basis.shape)} for {expected_voxels} displacement values"
            )
        if motion_factor_all.ndim != 3 or motion_factor_all.shape[1] != motion_basis.shape[1]:
            raise ValueError(
                "motion_covariance_factor must be [T, rank, hidden], got "
                f"{tuple(motion_factor_all.shape)}"
            )
    output_size = tuple(int(v) for v in displacement_all.shape[-3:])
    volume_size = tuple(int(v) for v in args.volume_size)
    spacing_before_resize = parse_spacing(args.spacing_mm) or nifti_spacing_dhw_mm(mask_path)
    spacing_mm = effective_output_spacing_mm(
        spacing_before_resize,
        volume_size=volume_size,
        output_size=output_size,
    )
    voxel_volume_mm3 = float(args.voxel_volume) if args.voxel_volume is not None else float(np.prod(spacing_mm))
    voxel_volume_ml = voxel_volume_mm3 / 1000.0
    mask = load_reference_mask(
        mask_path,
        time_index=args.mask_time_index,
        time_axis=int(args.time_axis),
        volume_size=volume_size,
        output_size=output_size,
        labels=parse_labels(args.labels),
        roi_mask_crop=bool(args.roi_mask_crop),
        roi_mask_margin=tuple(int(v) for v in args.roi_mask_margin),
    )

    volumes: list[float] = []
    volume_var: list[float | None] = []
    wall_motion: list[float] = []
    wall_motion_var: list[float | None] = []
    strain_rows: list[dict[str, float]] = []
    strain_var_rows: list[dict[str, float | None]] = []

    for idx in range(displacement_all.shape[0]):
        disp = displacement_all[idx].detach().clone().requires_grad_(True)
        cov = None if cov_diag_all is None else cov_diag_all[idx]
        factor = None if cov_factor_all is None else cov_factor_all[idx]
        if motion_basis is not None and motion_factor_all is not None:
            factor = motion_basis @ motion_factor_all[idx]
        vol = volume_from_deformation(mask, disp, voxel_volume=voxel_volume_ml)
        wm = mean_wall_motion_mm(mask, disp, spacing_mm=spacing_mm)
        strains = mean_green_lagrange_strain(mask, disp, spacing_mm=spacing_mm)
        volumes.append(float(vol.detach()))
        volume_var.append(variance_to_float(delta_variance_factor(vol, disp, factor) if factor is not None else delta_variance_diag(vol, disp, cov)))
        wall_motion.append(float(wm.detach()))
        wall_motion_var.append(variance_to_float(delta_variance_factor(wm, disp, factor) if factor is not None else delta_variance_diag(wm, disp, cov)))
        strain_rows.append({key: float(value.detach()) for key, value in strains.items()})
        strain_var_rows.append({
            key: variance_to_float(delta_variance_factor(value, disp, factor) if factor is not None else delta_variance_diag(value, disp, cov))
            for key, value in strains.items()
        })

    ed_index = int(args.ed_index)
    es_index = int(args.es_index)
    if es_index < 0:
        es_index = len(volumes) + es_index
    v_ed = torch.tensor(volumes[ed_index])
    v_es = torch.tensor(volumes[es_index])
    ef = ejection_fraction(v_ed, v_es)
    ef_var = None
    if volume_var[ed_index] is not None and volume_var[es_index] is not None:
        d_es = -1.0 / v_ed.clamp_min(1.0e-6)
        d_ed = v_es / v_ed.clamp_min(1.0e-6).pow(2)
        ef_var = float(d_ed.pow(2) * float(volume_var[ed_index]) + d_es.pow(2) * float(volume_var[es_index]))

    coverage = float(args.coverage)
    if not 0.0 < coverage < 1.0:
        raise ValueError("--coverage must be between 0 and 1")
    gaussian_multiplier = NormalDist().inv_cdf(0.5 + coverage / 2.0)

    def band(mean: float, variance: float | None, metric: str) -> dict[str, float | None]:
        if variance is None:
            return {"mean": mean, "variance": None, "standard_error": None, "lower": None, "upper": None}
        standard_error = float(np.sqrt(max(variance, 0.0)))
        multiplier = gaussian_multiplier
        if metric == "ef":
            bounded_mean = float(np.clip(mean, 1.0e-5, 1.0 - 1.0e-5))
            transformed_mean = float(np.log(bounded_mean / (1.0 - bounded_mean)))
            transformed_se = standard_error / (bounded_mean * (1.0 - bounded_mean))
            lower_logit = float(np.clip(transformed_mean - multiplier * transformed_se, -40.0, 40.0))
            upper_logit = float(np.clip(transformed_mean + multiplier * transformed_se, -40.0, 40.0))
            lower = 1.0 / (1.0 + np.exp(-lower_logit))
            upper = 1.0 / (1.0 + np.exp(-upper_logit))
            transform = "logit"
        elif metric == "volume_curve":
            positive_mean = max(mean, 1.0e-6)
            transformed_mean = float(np.log(positive_mean))
            transformed_se = standard_error / positive_mean
            lower = float(np.exp(np.clip(transformed_mean - multiplier * transformed_se, -40.0, 40.0)))
            upper = float(np.exp(np.clip(transformed_mean + multiplier * transformed_se, -40.0, 40.0)))
            transform = "log"
        else:
            lower = mean - multiplier * standard_error
            upper = mean + multiplier * standard_error
            transform = "identity"
        return {
            "mean": mean,
            "variance": variance,
            "standard_error": standard_error,
            "lower": lower,
            "upper": upper,
            "transform": transform,
        }

    result = {
        "deformation": str(deformation_path),
        "displacement_key": displacement_key,
        "reference_mask": str(mask_path),
        "dataset": item.get("dataset", ""),
        "source_path": item.get("source_path", ""),
        "units": {
            "spacing_order": "D,H,W",
            "spacing_before_resize_mm": list(spacing_before_resize),
            "output_spacing_mm": list(spacing_mm),
            "output_voxel_volume_mm3": voxel_volume_mm3,
            "volume": "ml",
            "wall_motion": "mm",
            "strain": "unitless",
        },
        "times": item.get("times", torch.arange(len(volumes))).tolist(),
        "volume_curve": volumes,
        "volume_variance": volume_var,
        "ed_index": ed_index,
        "es_index": es_index,
        "ef": float(ef),
        "ef_variance": ef_var,
        "wall_motion_mean": wall_motion,
        "wall_motion_variance": wall_motion_var,
        "strain_mean": strain_rows,
        "strain_variance": strain_var_rows,
        "prediction_bands": {
            "coverage": coverage,
            "method": "model_gaussian_delta",
            "calibration": "none",
            "ef_cross_time_covariance": "ED/ES independence approximation",
            "gaussian_multiplier": gaussian_multiplier,
            "ef": band(float(ef), ef_var, "ef"),
            "volume_curve": [band(mean, variance, "volume_curve") for mean, variance in zip(volumes, volume_var, strict=True)],
            "wall_motion": [band(mean, variance, "wall_motion") for mean, variance in zip(wall_motion, wall_motion_var, strict=True)],
            "strain": [
                {key: band(row[key], variance_row[key], "strain") for key in row}
                for row, variance_row in zip(strain_rows, strain_var_rows, strict=True)
            ],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "ef": result["ef"], "ef_variance": ef_var}, indent=2))


if __name__ == "__main__":
    main()
