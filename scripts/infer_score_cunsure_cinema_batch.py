#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import load_yaml, project_root, resolve_path, select_device
from cunsure_monai3d.foundation import build_foundation
from cunsure_monai3d.models import build_monai_unet3d
from cunsure_monai3d.preprocess import (
    FrameRef,
    crop_or_pad_around_bbox,
    extract_frame_array,
    load_mask_bbox,
    normalize_volume,
    scan_nifti_frames,
)
from cunsure_monai3d.score_cunsure import estimate_frame_noise, latent_covariance_from_noise_samples


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(text: str) -> str:
    text = text.replace(".nii.gz", "").replace(".nii", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def infer_dataset_name(path: Path) -> str:
    parts = set(path.parts)
    if "ACDC" in parts:
        return "ACDC"
    if "M&M1" in parts or "M_and_M1" in parts:
        return "M&M1"
    if "MnM2" in parts:
        return "MnM2"
    return "unknown"


def frame_output_name(root: Path, index: int, ref: FrameRef) -> str:
    try:
        relative = ref.path.relative_to(root)
    except ValueError:
        relative = ref.path
    time = "none" if ref.time_index is None else f"{ref.time_index:03d}"
    return f"{index:06d}_{safe_name(str(relative))}_t{time}.pt"


def resolve_output_path(path: str | Path, root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else root / value


def load_input_volume(
    ref: FrameRef,
    *,
    source_array: np.ndarray,
    roi_bbox: tuple[slice, slice, slice] | None,
    target_shape: tuple[int, int, int],
    time_axis: int,
    normalize: str,
    percentile_low: float,
    percentile_high: float,
) -> torch.Tensor:
    volume = extract_frame_array(source_array, ref.time_index, time_axis=time_axis, path=ref.path)
    volume = crop_or_pad_around_bbox(volume, roi_bbox, target_shape)
    volume = normalize_volume(
        volume,
        mode=normalize,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    return torch.from_numpy(volume[None, None]).float()


def covariance_metrics(covariance: torch.Tensor, *, storage: str) -> dict[str, float]:
    if storage == "diag":
        diagonal = covariance
        return {
            "diag_min": float(diagonal.min()),
            "diag_max": float(diagonal.max()),
            "trace": float(diagonal.sum()),
        }
    symmetric_error = float((covariance - covariance.T).abs().max())
    diagonal = covariance.diag()
    return {
        "symmetric_error": symmetric_error,
        "diag_min": float(diagonal.min()),
        "diag_max": float(diagonal.max()),
        "trace": float(diagonal.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/infer_cinema_score_cunsure_all_datasets.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    set_seed(int(cfg.get("seed", 2026)))
    device = select_device(cfg.get("device", "auto"))

    checkpoint_path = resolve_path(cfg["score"]["checkpoint"], root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "unsure_ardae_score":
        raise ValueError(f"{checkpoint_path} is not an UNSURE AR-DAE score checkpoint")
    train_cfg = checkpoint["config"]
    score_model = build_monai_unet3d(train_cfg["model"]).to(device)
    score_model.load_state_dict(checkpoint["model_state"])
    score_model.eval()
    for parameter in score_model.parameters():
        parameter.requires_grad_(False)
    volume_size = tuple(int(v) for v in train_cfg["data"].get("volume_size", [16, 96, 96]))

    input_cfg = cfg["input"]
    input_h5: h5py.File | None = None
    input_h5_path: Path | None = None
    h5_key = str(input_cfg.get("h5_key", "y"))
    if input_cfg.get("h5"):
        input_h5_path = resolve_path(input_cfg["h5"], root)
        input_h5 = h5py.File(input_h5_path, "r")
        if h5_key not in input_h5:
            raise KeyError(f"{input_h5_path} does not contain dataset {h5_key!r}")
        for metadata_key in ("source_path", "time_index"):
            if metadata_key not in input_h5:
                raise KeyError(f"{input_h5_path} does not contain metadata {metadata_key!r}")
        y_shape = tuple(int(v) for v in input_h5[h5_key].shape)
        expected_sample_shape = (int(train_cfg["model"]["in_channels"]), *volume_size)
        if y_shape[1:] != expected_sample_shape:
            raise ValueError(
                f"H5 sample shape {y_shape[1:]} does not match score checkpoint {expected_sample_shape}"
            )
        refs_all = []
        for source, time_index in zip(
            input_h5["source_path"][:],
            input_h5["time_index"][:],
            strict=True,
        ):
            source_text = source.decode("utf-8") if isinstance(source, bytes) else str(source)
            time_value = int(time_index)
            refs_all.append(FrameRef(Path(source_text), None if time_value < 0 else time_value))
        input_mode = "h5_roi"
    else:
        refs_all = scan_nifti_frames(
            root,
            list(input_cfg["globs"]),
            list(input_cfg.get("exclude_substrings", [])),
            split_4d=bool(input_cfg.get("split_4d", True)),
            time_axis=int(input_cfg.get("time_axis", -1)),
        )
        input_mode = "raw_nifti"
    start_index = int(input_cfg.get("start_index", 0))
    indexed_refs = list(enumerate(refs_all))[start_index:]
    if input_cfg.get("limit") is not None:
        indexed_refs = indexed_refs[: int(input_cfg["limit"])]
    if not indexed_refs:
        raise ValueError("no input frames found")

    foundation_cfg = dict(cfg["foundation"])
    foundation_cfg["repo_path"] = resolve_path(foundation_cfg["repo_path"], root)
    if foundation_cfg.get("cache_dir"):
        foundation_cfg["cache_dir"] = resolve_path(foundation_cfg["cache_dir"], root)
    encoder = build_foundation(foundation_cfg, device=device)

    output_dir = resolve_output_path(input_cfg["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"
    resume = bool(input_cfg.get("resume", True))
    if not resume and summary_path.exists():
        summary_path.unlink()

    noise_cfg = cfg["noise_estimation"]
    propagation_cfg = cfg["propagation"]
    storage = str(propagation_cfg.get("covariance_storage", "diag")).lower()
    if storage not in {"diag", "full"}:
        raise ValueError("covariance_storage must be 'diag' or 'full'")
    time_axis = int(input_cfg.get("time_axis", -1))
    roi_margin = tuple(int(v) for v in input_cfg.get("roi_mask_margin", [0, 16, 16]))

    cached_path: Path | None = None
    cached_array: np.ndarray | None = None
    cached_bbox: tuple[slice, slice, slice] | None = None
    for index, ref in tqdm(indexed_refs, desc="score C-UNSURE + CineMA"):
        output_name = frame_output_name(root, index, ref)
        output_path = output_dir / output_name
        if resume and output_path.exists():
            continue
        if input_h5 is not None:
            y = torch.from_numpy(np.asarray(input_h5[h5_key][index], dtype=np.float32)[None]).to(device)
        else:
            if cached_path != ref.path:
                cached_path = ref.path
                cached_array = np.asarray(nib.load(str(ref.path)).get_fdata(dtype=np.float32))
                cached_bbox = load_mask_bbox(
                    ref.path,
                    time_axis=time_axis,
                    margin=roi_margin,
                    enabled=bool(input_cfg.get("roi_mask_crop", False)),
                    require_mask=bool(input_cfg.get("require_roi_mask", False)),
                )
            if cached_array is None:
                raise RuntimeError(f"failed to load {ref.path}")
            y = load_input_volume(
                ref,
                source_array=cached_array,
                roi_bbox=cached_bbox,
                target_shape=volume_size,
                time_axis=time_axis,
                normalize=str(input_cfg.get("normalize", "percentile")),
                percentile_low=float(input_cfg.get("percentile_low", 1.0)),
                percentile_high=float(input_cfg.get("percentile_high", 99.0)),
            ).to(device)

        estimate = estimate_frame_noise(
            score_model,
            y,
            kernel_size=int(noise_cfg.get("kernel_size", 3)),
            spectral_floor=float(noise_cfg.get("spectral_floor", 1.0e-6)),
            relative_spectral_floor=float(noise_cfg.get("relative_spectral_floor", 0.01)),
            covariance_floor=float(noise_cfg.get("covariance_floor", 0.0)),
        )
        z, latent_covariance = latent_covariance_from_noise_samples(
            encoder,
            y,
            estimate.covariance_spectrum,
            num_samples=int(propagation_cfg.get("num_samples", 32)),
            sample_batch_size=int(propagation_cfg.get("sample_batch_size", 8)),
            tau=float(propagation_cfg.get("tau", 0.01)),
            difference_scheme=str(propagation_cfg.get("difference_scheme", "forward")),
            storage=storage,
        )
        tensors = {
            "eta": estimate.eta,
            "image_covariance_spectrum": estimate.covariance_spectrum,
            "z": z,
            "latent_covariance": latent_covariance,
        }
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"non-finite {name} for frame index {index}: {ref.path}")

        dataset_name = infer_dataset_name(ref.path)
        payload = {
            "method": "score_cunsure_per_frame",
            "z": z,
            "eta": estimate.eta.squeeze(0).detach().cpu(),
            "image_shape": tuple(y.shape),
            "source_path": str(ref.path),
            "time_index": -1 if ref.time_index is None else int(ref.time_index),
            "dataset": dataset_name,
            "input_mode": input_mode,
            "input_h5": None if input_h5_path is None else str(input_h5_path),
            "config": cfg,
        }
        covariance_key = "latent_covariance_diag" if storage == "diag" else "latent_covariance_psd"
        payload[covariance_key] = latent_covariance
        if bool(noise_cfg.get("save_diagnostics", False)):
            payload["score_autocorrelation"] = estimate.autocorrelation.squeeze(0).detach().cpu()
            payload["h_spectrum"] = estimate.h_spectrum.squeeze(0).detach().cpu()
            payload["image_covariance_spectrum"] = estimate.covariance_spectrum.squeeze(0).detach().cpu()
        torch.save(payload, output_path)

        metrics = {
            "index": index,
            "output": output_name,
            "dataset": dataset_name,
            "source_path": str(ref.path),
            "time_index": payload["time_index"],
            "method": payload["method"],
            "input_mode": input_mode,
            "kernel_size": int(estimate.eta.shape[-1]),
            "eta_mean": float(estimate.eta.mean()),
            "eta_norm": float(estimate.eta.norm()),
            "image_covariance_spectrum_min": float(estimate.covariance_spectrum.min()),
            "image_covariance_spectrum_max": float(estimate.covariance_spectrum.max()),
            "num_samples": int(propagation_cfg.get("num_samples", 32)),
            "covariance_storage": storage,
            "latent_covariance": covariance_metrics(latent_covariance, storage=storage),
        }
        with summary_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics, indent=2))

    if input_h5 is not None:
        input_h5.close()


if __name__ == "__main__":
    main()
