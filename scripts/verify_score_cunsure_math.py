#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.score_cunsure import (
    covariance_spectrum_from_eta,
    eta_from_score_autocorrelation,
    frame_score_autocorrelation,
    latent_covariance_from_noise_samples,
    linear_delta,
    sample_correlated_noise,
)


class LinearEncoder(nn.Module):
    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=1) @ self.matrix.T


def main() -> None:
    torch.manual_seed(2026)
    score = torch.full((1, 1, 4, 5, 6), 2.0)
    h = frame_score_autocorrelation(score, kernel_size=1)
    eta, _ = eta_from_score_autocorrelation(h, spectral_floor=1.0e-12)
    eta_error = float((eta - 0.25).abs().max())

    variance = 0.04
    eta_variance = torch.tensor([[[[[variance]]]]])
    spectrum = covariance_spectrum_from_eta(eta_variance, (4, 5, 6))
    samples = sample_correlated_noise(spectrum, num_samples=4096, dtype=torch.float32)
    sampled_variance = float(samples.var(unbiased=False))

    input_dim = 4 * 5 * 6
    matrix = torch.randn(3, input_dim) / input_dim**0.5
    encoder = LinearEncoder(matrix)
    y = torch.zeros((1, 1, 4, 5, 6))
    _, estimated = latent_covariance_from_noise_samples(
        encoder,
        y,
        spectrum,
        num_samples=8192,
        sample_batch_size=512,
        tau=0.01,
        difference_scheme="forward",
        storage="full",
    )
    exact = variance * (matrix @ matrix.T)
    relative_error = float((estimated - exact).norm() / exact.norm().clamp_min(1.0e-12))

    report = {
        "delta_start": linear_delta(0, 100, delta_min=0.001, delta_max=0.1),
        "delta_end": linear_delta(100, 100, delta_min=0.001, delta_max=0.1),
        "isotropic_eta_max_error": eta_error,
        "target_variance": variance,
        "sampled_variance": sampled_variance,
        "linear_covariance_relative_error": relative_error,
        "passed": eta_error < 1.0e-6 and abs(sampled_variance - variance) < 0.003 and relative_error < 0.08,
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
