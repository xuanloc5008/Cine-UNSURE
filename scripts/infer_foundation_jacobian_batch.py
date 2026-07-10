#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import load_yaml, project_root, resolve_path, select_device
from cunsure_monai3d.foundation import (
    build_foundation,
    covariance_sanity_metrics,
    full_jacobian_rows,
    latent_covariance_from_full_jacobian,
    latent_covariance_mc_finite_difference,
    project_covariance_psd,
    symmetrize_covariance,
)
from cunsure_monai3d.preprocess import (
    FrameRef,
    center_crop_or_pad,
    load_frame,
    normalize_volume,
    scan_nifti_frames,
)


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
        rel = ref.path.relative_to(root)
    except ValueError:
        rel = ref.path
    time = "none" if ref.time_index is None else f"{ref.time_index:03d}"
    return f"{index:06d}_{safe_name(str(rel))}_t{time}.pt"


def load_input_volume(
    ref: FrameRef,
    *,
    target_shape: tuple[int, int, int],
    time_axis: int,
    normalize: str,
    percentile_low: float,
    percentile_high: float,
) -> torch.Tensor:
    vol = load_frame(ref, time_axis=time_axis)
    vol = normalize_volume(
        vol,
        mode=normalize,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    vol = center_crop_or_pad(vol, target_shape)
    return torch.from_numpy(vol[None, None]).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    device = select_device(cfg.get("device", "auto"))

    ckpt_path = resolve_path(cfg["cunsure"]["checkpoint"], root)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    eta = ckpt["eta"].to(device)
    train_cfg = ckpt["config"]
    volume_size = tuple(int(v) for v in train_cfg["data"].get("volume_size", [16, 128, 128]))

    input_cfg = cfg["input"]
    refs = scan_nifti_frames(
        root,
        list(input_cfg["globs"]),
        list(input_cfg.get("exclude_substrings", [])),
        split_4d=bool(input_cfg.get("split_4d", True)),
        time_axis=int(input_cfg.get("time_axis", -1)),
    )
    start_index = int(input_cfg.get("start_index", 0))
    if start_index > 0:
        refs = refs[start_index:]
    limit = input_cfg.get("limit")
    if limit is not None:
        refs = refs[: int(limit)]
    if not refs:
        raise ValueError("no input NIfTI frames found")

    foundation_cfg = dict(cfg["foundation"])
    foundation_cfg["repo_path"] = resolve_path(foundation_cfg["repo_path"], root)
    if foundation_cfg["name"] == "cinema" and foundation_cfg.get("cache_dir"):
        foundation_cfg["cache_dir"] = resolve_path(foundation_cfg["cache_dir"], root)
    if foundation_cfg["name"] == "medsam2":
        foundation_cfg["checkpoint"] = resolve_path(foundation_cfg["checkpoint"], root)
    encoder = build_foundation(foundation_cfg, device=device)

    out_dir = resolve_path(input_cfg["output_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    normalize = str(input_cfg.get("normalize", "percentile"))
    percentile_low = float(input_cfg.get("percentile_low", 1.0))
    percentile_high = float(input_cfg.get("percentile_high", 99.0))
    time_axis = int(input_cfg.get("time_axis", -1))
    chunk_size = int(cfg["jacobian"]["chunk_size"])
    method = str(cfg["jacobian"].get("method", "full")).lower()
    num_probes = int(cfg["jacobian"].get("num_probes", 64))
    fd_epsilon = float(cfg["jacobian"].get("fd_epsilon", 0.01))
    normalize_directions = bool(cfg["jacobian"].get("normalize_directions", True))
    save_jacobian = bool(cfg["jacobian"].get("save_jacobian", False))
    save_raw_covariance = bool(cfg["jacobian"].get("save_raw_covariance", False))

    for local_idx, ref in enumerate(tqdm(refs, desc="foundation batch")):
        idx = start_index + local_idx
        x = load_input_volume(
            ref,
            target_shape=volume_size,
            time_axis=time_axis,
            normalize=normalize,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        ).to(device)
        jac = None
        if method == "full":
            z, jac = full_jacobian_rows(encoder, x, chunk_size=chunk_size)
            sigma_z = latent_covariance_from_full_jacobian(
                jac,
                input_shape=tuple(x.shape[1:]),
                eta=eta,
                device=device,
            )
        elif method == "mc_fd":
            z, sigma_z = latent_covariance_mc_finite_difference(
                encoder,
                x,
                eta=eta,
                device=device,
                num_probes=num_probes,
                fd_epsilon=fd_epsilon,
                normalize_directions=normalize_directions,
            )
        else:
            raise ValueError(f"unsupported jacobian method: {method}")
        sigma_z_sym = symmetrize_covariance(sigma_z)
        sigma_z_psd = project_covariance_psd(sigma_z_sym)

        out_name = frame_output_name(root, idx, ref)
        dataset_name = infer_dataset_name(ref.path)
        payload = {
            "z": z,
            "eta": eta.detach().cpu(),
            "latent_covariance_psd": sigma_z_psd,
            "image_shape": tuple(x.shape),
            "source_path": str(ref.path),
            "time_index": -1 if ref.time_index is None else int(ref.time_index),
            "dataset": dataset_name,
            "config": cfg,
        }
        if save_raw_covariance:
            payload["latent_covariance"] = sigma_z
            payload["latent_covariance_sym"] = sigma_z_sym
        if save_jacobian and jac is not None:
            payload["jacobian"] = jac
        torch.save(payload, out_dir / out_name)

        metrics = {
            "index": idx,
            "output": out_name,
            "dataset": dataset_name,
            "source_path": str(ref.path),
            "time_index": -1 if ref.time_index is None else int(ref.time_index),
            "method": method,
            "num_probes": num_probes if method == "mc_fd" else None,
            "latent_dim": int(z.numel()),
            "jacobian_shape": None if jac is None else list(jac.shape),
            "covariance_psd": covariance_sanity_metrics(sigma_z_psd),
        }
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
