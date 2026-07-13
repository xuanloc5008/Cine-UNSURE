#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import load_yaml, project_root, resolve_path, select_device
from cunsure_monai3d.models import build_monai_unet3d
from cunsure_monai3d.score_cunsure import (
    estimate_noise_from_score,
    sample_correlated_noise,
    zed_denoise,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_h5_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def psnr_per_item(prediction: torch.Tensor, target: torch.Tensor, *, data_range: float) -> torch.Tensor:
    prediction = prediction.clamp(0.0, float(data_range))
    mse = (prediction - target).square().flatten(start_dim=1).mean(dim=1)
    return 10.0 * torch.log10(float(data_range) ** 2 / mse.clamp_min(1.0e-12))


def box_covariance_spectrum(
    spatial_shape: tuple[int, int, int],
    *,
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("blur_kernel_size must be a positive odd integer")
    if any(kernel_size > size for size in spatial_shape):
        raise ValueError(f"blur kernel {kernel_size} exceeds volume shape {spatial_shape}")
    radius = kernel_size // 2
    kernel = torch.zeros((1, 1, *spatial_shape), device=device, dtype=dtype)
    weight = 1.0 / float(kernel_size**3)
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                kernel[0, 0, dz % spatial_shape[0], dy % spatial_shape[1], dx % spatial_shape[2]] = weight
    kernel_energy = float(kernel_size**3) * weight**2
    transfer = torch.fft.fftn(kernel, dim=(-3, -2, -1))
    return (float(sigma) ** 2 / kernel_energy) * transfer.abs().square()


def known_noise(
    reference: torch.Tensor,
    *,
    noise_type: str,
    sigma: float,
    blur_kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = tuple(int(v) for v in reference.shape[-3:])
    if noise_type == "iid_gaussian":
        spectrum = reference.new_full((1, 1, *shape), float(sigma) ** 2)
    elif noise_type == "box_correlated_gaussian":
        spectrum = box_covariance_spectrum(
            shape,
            kernel_size=blur_kernel_size,
            sigma=sigma,
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        raise ValueError(f"unsupported noise type: {noise_type}")
    noise = sample_correlated_noise(spectrum, num_samples=reference.shape[0], dtype=reference.dtype)
    return reference + noise, spectrum.expand(reference.shape[0], -1, -1, -1, -1)


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def save_example(
    path: Path,
    *,
    reference: torch.Tensor,
    noisy: torch.Tensor,
    estimated: torch.Tensor,
    oracle: torch.Tensor,
    title: str,
) -> None:
    depth = int(reference.shape[-3] // 2)
    arrays = [
        reference[0, depth].detach().cpu().numpy(),
        noisy[0, depth].detach().cpu().numpy(),
        estimated[0, depth].detach().cpu().numpy(),
        oracle[0, depth].detach().cpu().numpy(),
        (estimated[0, depth] - reference[0, depth]).abs().detach().cpu().numpy(),
    ]
    labels = ["Held-out reference", "Known-noise input", "UNSURE via score", "Known-Sigma score", "Absolute error"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for axis, array, label in zip(axes, arrays, labels, strict=True):
        axis.imshow(array, cmap="magma" if label == "Absolute error" else "gray")
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["noise_model"], row["sigma"], row["estimator_kernel_size"])].append(row)
    summaries = []
    metric_names = (
        "noisy_psnr_db",
        "unsure_psnr_db",
        "known_covariance_psnr_db",
        "estimated_sigma",
        "estimated_variance",
        "variance_relative_error",
        "covariance_spectral_relative_error",
    )
    for (noise_model, sigma, kernel_size), group in sorted(groups.items()):
        summary = {
            "noise_model": noise_model,
            "sigma": sigma,
            "true_variance": sigma**2,
            "estimator_kernel_size": kernel_size,
            "num_measurements": len(group),
        }
        for metric in metric_names:
            summary[metric] = mean_std([float(row[metric]) for row in group])
        summary["psnr_gain_over_noisy_db"] = (
            summary["unsure_psnr_db"]["mean"] - summary["noisy_psnr_db"]["mean"]
        )
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-aligned controlled evaluation of an UNSURE-via-score checkpoint."
    )
    parser.add_argument("--config", default="configs/acdc/evaluate_cunsure_unsure_protocol.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    eval_cfg = cfg["evaluation"]
    set_seed(int(cfg.get("seed", 2026)))
    device = select_device(cfg.get("device", "auto"))

    checkpoint_path = resolve_path(eval_cfg["checkpoint"], root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "unsure_ardae_score":
        raise ValueError("checkpoint is not an UNSURE AR-DAE score model")
    model = build_monai_unet3d(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    h5_path = resolve_path(eval_cfg["test_h5"], root)
    output_dir = resolve_path(eval_cfg["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    kernel_sizes = [int(value) for value in eval_cfg["estimator_kernel_sizes"]]
    primary_kernel = int(eval_cfg.get("primary_kernel_size", kernel_sizes[0]))
    if primary_kernel not in kernel_sizes:
        raise ValueError("primary_kernel_size must be included in estimator_kernel_sizes")

    with h5py.File(h5_path, "r") as h5:
        total_frames = len(h5["y"])
        requested = min(int(eval_cfg.get("num_frames", total_frames)), total_frames)
        indices = np.linspace(0, total_frames - 1, requested, dtype=np.int64)
        sources = [decode_h5_string(h5["source_path"][index]) for index in indices]
        times = [int(h5["time_index"][index]) for index in indices]

    rows: list[dict] = []
    visual_count = 0
    max_visuals = int(eval_cfg.get("save_examples", 0))
    batch_size = int(eval_cfg.get("batch_size", 1))
    repeats = int(eval_cfg.get("noise_repeats", 1))
    data_range = float(eval_cfg.get("data_range", 1.0))
    spectral_floor = float(eval_cfg.get("spectral_floor", 1.0e-6))
    relative_floor = float(eval_cfg.get("relative_spectral_floor", 0.01))
    covariance_floor = float(eval_cfg.get("covariance_floor", 0.0))

    jobs = []
    for noise_cfg in eval_cfg["noise_models"]:
        for sigma in noise_cfg["sigmas"]:
            for repeat in range(repeats):
                jobs.append((noise_cfg, float(sigma), repeat))

    for noise_cfg, sigma, repeat in tqdm(jobs, desc="UNSURE protocol"):
        noise_name = str(noise_cfg["name"])
        noise_type = str(noise_cfg["type"])
        blur_size = int(noise_cfg.get("blur_kernel_size", 1))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            with h5py.File(h5_path, "r") as h5:
                reference_np = np.stack([h5["y"][int(index)] for index in batch_indices])
            reference = torch.from_numpy(reference_np).float().to(device)
            noisy, true_spectrum = known_noise(
                reference,
                noise_type=noise_type,
                sigma=sigma,
                blur_kernel_size=blur_size,
            )
            with torch.no_grad():
                score = model(noisy)
                known_covariance_reconstruction = noisy + zed_denoise(score, true_spectrum)
                noisy_psnr = psnr_per_item(noisy, reference, data_range=data_range)
                known_psnr = psnr_per_item(
                    known_covariance_reconstruction,
                    reference,
                    data_range=data_range,
                )

                for kernel_size in kernel_sizes:
                    estimate = estimate_noise_from_score(
                        score,
                        kernel_size=kernel_size,
                        spectral_floor=spectral_floor,
                        relative_spectral_floor=relative_floor,
                        covariance_floor=covariance_floor,
                    )
                    reconstruction = noisy + zed_denoise(score, estimate.covariance_spectrum)
                    unsure_psnr = psnr_per_item(reconstruction, reference, data_range=data_range)
                    estimated_variance = estimate.covariance_spectrum.mean(dim=(-4, -3, -2, -1))
                    covariance_error = (
                        (estimate.covariance_spectrum - true_spectrum)
                        .flatten(start_dim=1)
                        .norm(dim=1)
                        / true_spectrum.flatten(start_dim=1).norm(dim=1).clamp_min(1.0e-12)
                    )

                    for local_index, frame_index in enumerate(batch_indices):
                        variance = float(estimated_variance[local_index])
                        rows.append(
                            {
                                "frame_index": int(frame_index),
                                "source_path": sources[start + local_index],
                                "time_index": times[start + local_index],
                                "repeat": repeat,
                                "noise_model": noise_name,
                                "sigma": sigma,
                                "true_variance": sigma**2,
                                "estimator_kernel_size": kernel_size,
                                "noisy_psnr_db": float(noisy_psnr[local_index]),
                                "unsure_psnr_db": float(unsure_psnr[local_index]),
                                "known_covariance_psnr_db": float(known_psnr[local_index]),
                                "estimated_sigma": math.sqrt(max(variance, 0.0)),
                                "estimated_variance": variance,
                                "variance_relative_error": abs(variance - sigma**2) / max(sigma**2, 1.0e-12),
                                "covariance_spectral_relative_error": float(covariance_error[local_index]),
                            }
                        )

                    if kernel_size == primary_kernel and visual_count < max_visuals:
                        save_example(
                            output_dir / "examples" / f"example_{visual_count:03d}.png",
                            reference=reference[0].detach().cpu(),
                            noisy=noisy[0].detach().cpu(),
                            estimated=reconstruction[0].detach().cpu(),
                            oracle=known_covariance_reconstruction[0].detach().cpu(),
                            title=f"{noise_name}, sigma={sigma:g}, frame={int(batch_indices[0])}",
                        )
                        visual_count += 1

    summaries = aggregate_rows(rows)
    report = {
        "protocol": "UNSURE paper-style controlled denoising and noise-estimation evaluation",
        "reference_assumption": (
            "Held-out normalized cine-MRI frames are treated as references and receive additional known "
            "synthetic noise. They are not claimed to be physically noise-free MRI ground truth."
        ),
        "paper_alignment": {
            "reconstruction": "f(y) = y + Sigma_eta s_theta(y), Theorem 3 / UNSURE via score",
            "quality_metric": "test PSNR mean and standard deviation",
            "noise_estimation": "estimated eta variance versus known injected sigma squared",
            "kernel_ablation": kernel_sizes,
            "important_scope": (
                "This evaluates image-domain denoising. Reproducing the paper's FastMRI Table 4 additionally "
                "requires raw k-space, a 2x undersampling operator, and the EI reconstruction loss."
            ),
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_h5": str(h5_path),
        "num_reference_frames": len(indices),
        "noise_repeats": repeats,
        "num_rows": len(rows),
        "summary": summaries,
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"checkpoint epoch: {checkpoint['epoch']}")
    print(f"held-out references: {len(indices)}")
    print("\nPaper-style PSNR and eta evaluation:")
    for summary in summaries:
        print(
            f"{summary['noise_model']:24s} sigma={summary['sigma']:.3f} "
            f"K={summary['estimator_kernel_size']} | "
            f"noisy={summary['noisy_psnr_db']['mean']:.2f}+/-{summary['noisy_psnr_db']['std']:.2f} dB | "
            f"UNSURE={summary['unsure_psnr_db']['mean']:.2f}+/-{summary['unsure_psnr_db']['std']:.2f} dB | "
            f"sigma_hat={summary['estimated_sigma']['mean']:.4f}+/-{summary['estimated_sigma']['std']:.4f}"
        )
    print(f"\nreport: {report_path}")
    print(f"per-frame metrics: {output_dir / 'per_frame_metrics.csv'}")


if __name__ == "__main__":
    main()
