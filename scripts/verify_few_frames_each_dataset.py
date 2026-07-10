#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
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
from cunsure_monai3d.losses import MinimaxCUNSURE3DLoss
from cunsure_monai3d.models import build_monai_unet3d
from cunsure_monai3d.preprocess import FrameRef, center_crop_or_pad, load_frame, normalize_volume, scan_nifti_frames


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(text: str) -> str:
    text = text.replace(".nii.gz", "").replace(".nii", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def select_refs(refs: list[FrameRef], count: int, mode: str) -> list[FrameRef]:
    if count <= 0 or len(refs) <= count:
        return refs
    if mode == "first":
        return refs[:count]
    if mode == "even":
        indices = np.linspace(0, len(refs) - 1, count, dtype=int).tolist()
        return [refs[i] for i in indices]
    raise ValueError(f"unsupported sample mode: {mode}")


def load_checkpoint(checkpoint: Path, *, device: torch.device) -> tuple[torch.nn.Module, MinimaxCUNSURE3DLoss, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_monai_unet3d(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loss_cfg = cfg["loss"]
    loss_fn = MinimaxCUNSURE3DLoss(
        kernel_size=int(loss_cfg["kernel_size"]),
        eta_init=float(loss_cfg["eta_init"]),
        tau=float(loss_cfg["tau"]),
        eta_step_size=float(loss_cfg["eta_step_size"]),
        eta_momentum=float(loss_cfg["eta_momentum"]),
        eta_grad_clip=None if loss_cfg.get("eta_grad_clip") is None else float(loss_cfg["eta_grad_clip"]),
        eta_max_norm=None if loss_cfg.get("eta_max_norm") is None else float(loss_cfg["eta_max_norm"]),
        device=device,
    ).to(device)
    loss_fn.eta.data.copy_(ckpt["eta"].to(device))
    loss_fn.eval()
    return model, loss_fn, ckpt


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


def denoise_report(model: torch.nn.Module, y: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        x = model(y)
    delta = x - y
    return {
        "input_mean": float(y.mean().detach().cpu()),
        "input_std": float(y.std().detach().cpu()),
        "output_mean": float(x.mean().detach().cpu()),
        "output_std": float(x.std().detach().cpu()),
        "delta_abs_mean": float(delta.abs().mean().detach().cpu()),
        "delta_rmse": float(delta.pow(2).mean().sqrt().detach().cpu()),
        "output_min": float(x.min().detach().cpu()),
        "output_max": float(x.max().detach().cpu()),
    }


def cunsure_report(model: torch.nn.Module, loss_fn: MinimaxCUNSURE3DLoss, y: torch.Tensor) -> dict[str, float]:
    with torch.enable_grad():
        out = loss_fn(model, y)
    residual = float(out.residual.mean().detach().cpu())
    divergence = float(out.divergence.mean().detach().cpu())
    return {
        "loss": float(out.loss.mean().detach().cpu()),
        "residual": residual,
        "divergence": divergence,
        "abs_divergence_to_residual": abs(divergence) / max(residual, 1.0e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    set_seed(int(cfg.get("seed", 2026)))
    device = select_device(cfg.get("device", "auto"))

    model, loss_fn, ckpt = load_checkpoint(resolve_path(cfg["cunsure"]["checkpoint"], root), device=device)
    train_cfg = ckpt["config"]
    volume_size = tuple(int(v) for v in train_cfg["data"].get("volume_size", [16, 128, 128]))

    foundation_cfg = dict(cfg["foundation"])
    foundation_cfg["repo_path"] = resolve_path(foundation_cfg["repo_path"], root)
    if foundation_cfg["name"] == "cinema" and foundation_cfg.get("cache_dir"):
        foundation_cfg["cache_dir"] = resolve_path(foundation_cfg["cache_dir"], root)
    if foundation_cfg["name"] == "medsam2":
        foundation_cfg["checkpoint"] = resolve_path(foundation_cfg["checkpoint"], root)
    encoder = build_foundation(foundation_cfg, device=device)

    verify_cfg = cfg["verify"]
    out_dir = resolve_path(verify_cfg["output_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    sample_mode = str(verify_cfg.get("sample_mode", "even"))
    frames_per_dataset = int(verify_cfg.get("frames_per_dataset", 3))
    normalize = str(verify_cfg.get("normalize", "percentile"))
    percentile_low = float(verify_cfg.get("percentile_low", 1.0))
    percentile_high = float(verify_cfg.get("percentile_high", 99.0))
    split_4d = bool(verify_cfg.get("split_4d", True))
    time_axis = int(verify_cfg.get("time_axis", -1))
    chunk_size = int(cfg["jacobian"]["chunk_size"])
    method = str(cfg["jacobian"].get("method", "full")).lower()
    num_probes = int(cfg["jacobian"].get("num_probes", 64))
    fd_epsilon = float(cfg["jacobian"].get("fd_epsilon", 0.01))
    normalize_directions = bool(cfg["jacobian"].get("normalize_directions", True))
    save_outputs = bool(verify_cfg.get("save_outputs", True))

    for dataset in cfg["datasets"]:
        name = str(dataset["name"])
        refs = scan_nifti_frames(
            root,
            list(dataset["globs"]),
            list(verify_cfg.get("exclude_substrings", [])),
            split_4d=split_4d,
            time_axis=time_axis,
        )
        refs = select_refs(refs, frames_per_dataset, sample_mode)
        if not refs:
            row = {"dataset": name, "error": "no frames found"}
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row, indent=2))
            continue

        for local_idx, ref in enumerate(tqdm(refs, desc=f"verify {name}")):
            y = load_input_volume(
                ref,
                target_shape=volume_size,
                time_axis=time_axis,
                normalize=normalize,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
            ).to(device)
            cunsure = cunsure_report(model, loss_fn, y)
            denoise = denoise_report(model, y)
            jac = None
            if method == "full":
                z, jac = full_jacobian_rows(encoder, y, chunk_size=chunk_size)
                sigma_z = latent_covariance_from_full_jacobian(
                    jac,
                    input_shape=tuple(y.shape[1:]),
                    eta=loss_fn.eta,
                    device=device,
                )
            elif method == "mc_fd":
                z, sigma_z = latent_covariance_mc_finite_difference(
                    encoder,
                    y,
                    eta=loss_fn.eta,
                    device=device,
                    num_probes=num_probes,
                    fd_epsilon=fd_epsilon,
                    normalize_directions=normalize_directions,
                )
            else:
                raise ValueError(f"unsupported jacobian method: {method}")
            sigma_z_psd = project_covariance_psd(symmetrize_covariance(sigma_z))
            cov = covariance_sanity_metrics(sigma_z_psd)

            output_name = f"{safe_name(name)}_{local_idx:03d}_{safe_name(ref.path.name)}_t{ref.time_index if ref.time_index is not None else -1}.pt"
            row = {
                "dataset": name,
                "local_index": local_idx,
                "source_path": str(ref.path),
                "time_index": -1 if ref.time_index is None else int(ref.time_index),
                "checkpoint_epoch": int(ckpt["epoch"]),
                "eta_norm": float(loss_fn.eta.detach().cpu().norm()),
                "method": method,
                "num_probes": num_probes if method == "mc_fd" else None,
                "cunsure": cunsure,
                "denoise": denoise,
                "latent_dim": int(z.numel()),
                "jacobian_shape": None if jac is None else list(jac.shape),
                "covariance_psd": cov,
            }
            if save_outputs:
                payload = {
                    "z": z,
                    "eta": loss_fn.eta.detach().cpu(),
                    "latent_covariance_psd": sigma_z_psd,
                    "source_path": str(ref.path),
                    "time_index": row["time_index"],
                    "dataset": name,
                    "metrics": row,
                }
                torch.save(payload, out_dir / output_name)
                row["output"] = output_name

            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
