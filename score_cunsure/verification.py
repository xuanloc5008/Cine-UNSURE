"""Verification utilities for score C-UNSURE covariance inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from score_cunsure.cunsure import CUNSUREConfig, covariance_from_deltas, estimate_latent_covariance


Tensor = torch.Tensor


@dataclass
class SyntheticAlignmentResult:
    """Synthetic noise verification against Monte Carlo encoder covariance."""

    cunsure_covariance: Tensor
    monte_carlo_covariance: Tensor
    cosine_similarity: Tensor
    relative_trace_error: Tensor
    cunsure_trace: Tensor
    monte_carlo_trace: Tensor
    passed: bool


@dataclass
class SensitivityTraceResult:
    """Trace monotonicity verification over increasing image noise levels."""

    noise_levels: list[float]
    traces: Tensor
    mean_traces: Tensor
    std_traces: Tensor
    monotonic_per_trial: Tensor
    pass_rate: float
    passed: bool


def covariance_cosine_similarity(a: Tensor, b: Tensor, eps: float = 1.0e-12) -> Tensor:
    """Cosine similarity between vectorized covariance matrices."""
    if a.ndim == 2:
        a = a.unsqueeze(0)
    if b.ndim == 2:
        b = b.unsqueeze(0)
    a_flat = a.flatten(start_dim=1)
    b_flat = b.flatten(start_dim=1)
    return (a_flat * b_flat).sum(dim=1) / (a_flat.norm(dim=1) * b_flat.norm(dim=1) + eps)


def covariance_trace(covariance: Tensor) -> Tensor:
    """Trace for `[B, d, d]` or `[d, d]` covariance."""
    if covariance.ndim == 2:
        covariance = covariance.unsqueeze(0)
    return covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def relative_trace_error(estimate: Tensor, target: Tensor, eps: float = 1.0e-12) -> Tensor:
    """Relative trace error between two covariance matrices."""
    target_trace = covariance_trace(target)
    return (covariance_trace(estimate) - target_trace).abs() / (target_trace.abs() + eps)


def add_white_noise(
    image: Tensor,
    sigma: float,
    *,
    generator: torch.Generator | None = None,
    base_noise: Tensor | None = None,
    clamp: bool = False,
) -> Tensor:
    """Add Gaussian white noise to a normalized image tensor."""
    if sigma == 0:
        noisy = image.clone()
    else:
        noise = base_noise
        if noise is None:
            noise = torch.randn(image.shape, dtype=image.dtype, device=image.device, generator=generator)
        noisy = image + sigma * noise
    return noisy.clamp(0.0, 1.0) if clamp else noisy


def monte_carlo_encoder_covariance(
    image: Tensor,
    encoder: torch.nn.Module,
    *,
    sigma: float,
    n_samples: int,
    batch_size: int = 8,
    generator: torch.Generator | None = None,
    clamp: bool = False,
) -> Tensor:
    """Estimate ground-truth latent covariance by Monte Carlo white-noise injection."""
    if image.shape[0] != 1:
        raise ValueError("Monte Carlo verification currently expects batch size 1")
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")

    samples: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            current = min(batch_size, n_samples - start)
            batch = image.expand(current, *image.shape[1:]).contiguous()
            noisy = add_white_noise(batch, sigma, generator=generator, clamp=clamp)
            samples.append(encoder(noisy).flatten(start_dim=1).unsqueeze(1))

    # [S, B=1, d]
    z_samples = torch.cat(samples, dim=0)
    return covariance_from_deltas(z_samples, unbiased=True)


def run_synthetic_alignment(
    image: Tensor,
    score_model: torch.nn.Module,
    encoder: torch.nn.Module,
    config: CUNSUREConfig,
    *,
    sigma: float = 0.1,
    n_mc_samples: int = 128,
    mc_batch_size: int = 8,
    cosine_threshold: float = 0.95,
    generator: torch.Generator | None = None,
    clamp: bool = False,
) -> SyntheticAlignmentResult:
    """Compare C-UNSURE covariance to Monte Carlo covariance under known noise."""
    noisy = add_white_noise(image, sigma, generator=generator, clamp=clamp)
    cunsure = estimate_latent_covariance(noisy, score_model, encoder, config, generator=generator).covariance
    mc = monte_carlo_encoder_covariance(
        image,
        encoder,
        sigma=sigma,
        n_samples=n_mc_samples,
        batch_size=mc_batch_size,
        generator=generator,
        clamp=clamp,
    )
    cosine = covariance_cosine_similarity(cunsure, mc)
    trace_error = relative_trace_error(cunsure, mc)
    return SyntheticAlignmentResult(
        cunsure_covariance=cunsure,
        monte_carlo_covariance=mc,
        cosine_similarity=cosine,
        relative_trace_error=trace_error,
        cunsure_trace=covariance_trace(cunsure),
        monte_carlo_trace=covariance_trace(mc),
        passed=bool((cosine >= cosine_threshold).all().item()),
    )


def _is_strictly_increasing(values: Tensor) -> Tensor:
    return (values[..., 1:] > values[..., :-1]).all(dim=-1)


def run_sensitivity_trace_test(
    image: Tensor,
    score_model: torch.nn.Module,
    encoder: torch.nn.Module,
    config: CUNSUREConfig,
    *,
    noise_levels: Iterable[float] = (0.0, 0.05, 0.20),
    trials: int = 1,
    generator: torch.Generator | None = None,
    clamp: bool = False,
    shared_noise_direction: bool = True,
) -> SensitivityTraceResult:
    """Verify Trace_raw < Trace_light < Trace_heavy under increasing noise."""
    levels = [float(x) for x in noise_levels]
    if image.shape[0] != 1:
        raise ValueError("Sensitivity verification currently expects batch size 1")
    if len(levels) < 2:
        raise ValueError("at least two noise levels are required")
    traces = []

    for _ in range(trials):
        base_noise = None
        if shared_noise_direction:
            base_noise = torch.randn(image.shape, dtype=image.dtype, device=image.device, generator=generator)
        trial_traces = []
        for sigma in levels:
            noisy = add_white_noise(
                image,
                sigma,
                generator=generator,
                base_noise=base_noise if sigma > 0 else None,
                clamp=clamp,
            )
            result = estimate_latent_covariance(noisy, score_model, encoder, config, generator=generator)
            trial_traces.append(covariance_trace(result.covariance))
        traces.append(torch.stack(trial_traces, dim=1))  # [B, L]

    trace_tensor = torch.stack(traces, dim=0).squeeze(1)  # [T, L] for B=1
    monotonic = _is_strictly_increasing(trace_tensor)
    pass_rate = float(monotonic.float().mean().item())
    return SensitivityTraceResult(
        noise_levels=levels,
        traces=trace_tensor,
        mean_traces=trace_tensor.mean(dim=0),
        std_traces=trace_tensor.std(dim=0, unbiased=False),
        monotonic_per_trial=monotonic,
        pass_rate=pass_rate,
        passed=bool(monotonic.all().item()),
    )
