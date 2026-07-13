from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import torch
from torch import nn


def linear_delta(step: int, total_steps: int, *, delta_min: float, delta_max: float) -> float:
    """Linear AR-DAE annealing used by the public UNSURE implementation."""
    weight = min(max(float(step) / max(int(total_steps), 1), 0.0), 1.0)
    return float(delta_max) * (1.0 - weight) + float(delta_min) * weight


def ardae_score_loss(
    model: nn.Module,
    y: torch.Tensor,
    *,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return ||epsilon + sigma S(y + sigma epsilon)||^2.

    This is the score objective implemented by ``ScoreModel`` in the public
    UNSURE repository. A signed scalar sigma is drawn independently per item.
    """
    sigma = torch.randn((y.shape[0],) + (1,) * (y.ndim - 1), device=y.device, dtype=y.dtype)
    sigma = sigma * float(delta)
    epsilon = torch.randn_like(y)
    score = model(y + sigma * epsilon)
    error = epsilon + sigma * score
    return error.square().flatten(start_dim=1).mean(dim=1), score


def frame_score_autocorrelation(score: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    """Estimate the local stationary score autocorrelation for each frame."""
    if score.ndim != 5:
        raise ValueError(f"score must be [B,C,D,H,W], got {tuple(score.shape)}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    radius = kernel_size // 2
    values = []
    for shift in product(range(-radius, radius + 1), repeat=3):
        shifted = torch.roll(score, shifts=shift, dims=(-3, -2, -1))
        values.append((score * shifted).mean(dim=(-3, -2, -1), keepdim=False))
    return torch.stack(values, dim=-1).reshape(score.shape[0], score.shape[1], kernel_size, kernel_size, kernel_size)


def _centrosymmetrize(kernel: torch.Tensor) -> torch.Tensor:
    return 0.5 * (kernel + torch.flip(kernel, dims=(-3, -2, -1)))


def eta_from_score_autocorrelation(
    autocorrelation: torch.Tensor,
    *,
    spectral_floor: float,
    relative_spectral_floor: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute eta = F^-1(1 / F h) with a positive spectral floor."""
    centered = _centrosymmetrize(autocorrelation)
    h_spectrum = torch.fft.fftn(torch.fft.ifftshift(centered, dim=(-3, -2, -1)), dim=(-3, -2, -1)).real
    relative_floor = h_spectrum.amax(dim=(-3, -2, -1), keepdim=True) * float(relative_spectral_floor)
    floor = torch.maximum(relative_floor, h_spectrum.new_tensor(float(spectral_floor)))
    h_spectrum = torch.maximum(h_spectrum, floor)
    eta_spectrum = h_spectrum.reciprocal()
    eta = torch.fft.fftshift(
        torch.fft.ifftn(eta_spectrum, dim=(-3, -2, -1)).real,
        dim=(-3, -2, -1),
    )
    return _centrosymmetrize(eta), h_spectrum


def covariance_spectrum_from_eta(
    eta: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    *,
    covariance_floor: float = 0.0,
) -> torch.Tensor:
    """Embed a compact centered covariance kernel and return its PSD spectrum."""
    if eta.ndim != 5:
        raise ValueError(f"eta must be [B,C,K,K,K], got {tuple(eta.shape)}")
    kernel_shape = eta.shape[-3:]
    if any(k > n for k, n in zip(kernel_shape, spatial_shape, strict=True)):
        raise ValueError(f"eta kernel {kernel_shape} is larger than spatial shape {spatial_shape}")
    radius = tuple(k // 2 for k in kernel_shape)
    embedded = eta.new_zeros((*eta.shape[:2], *spatial_shape))
    for kernel_index in product(*(range(k) for k in kernel_shape)):
        lag = tuple(i - r for i, r in zip(kernel_index, radius, strict=True))
        target = tuple(d % n for d, n in zip(lag, spatial_shape, strict=True))
        embedded[(slice(None), slice(None), *target)] += eta[(slice(None), slice(None), *kernel_index)]
    spectrum = torch.fft.fftn(embedded, dim=(-3, -2, -1)).real
    return spectrum.clamp_min(float(covariance_floor))


def sample_correlated_noise(
    covariance_spectrum: torch.Tensor,
    *,
    num_samples: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample N(0, Sigma) using the circulant square root of Sigma."""
    if covariance_spectrum.shape[0] != 1:
        raise ValueError("per-frame sampling expects one covariance spectrum")
    spectrum = covariance_spectrum.expand(num_samples, -1, -1, -1, -1)
    white = torch.randn(spectrum.shape, device=spectrum.device, dtype=dtype)
    shaped = torch.fft.ifftn(
        torch.fft.fftn(white, dim=(-3, -2, -1)) * spectrum.sqrt(),
        dim=(-3, -2, -1),
    ).real
    return shaped.to(dtype=dtype)


@dataclass(frozen=True)
class FrameNoiseEstimate:
    score: torch.Tensor
    autocorrelation: torch.Tensor
    eta: torch.Tensor
    h_spectrum: torch.Tensor
    covariance_spectrum: torch.Tensor


@torch.no_grad()
def estimate_noise_from_score(
    score: torch.Tensor,
    *,
    kernel_size: int,
    spectral_floor: float,
    relative_spectral_floor: float,
    covariance_floor: float,
) -> FrameNoiseEstimate:
    """Apply the score-based UNSURE covariance estimator to a known score."""
    autocorrelation = frame_score_autocorrelation(score, kernel_size=kernel_size)
    eta, h_spectrum = eta_from_score_autocorrelation(
        autocorrelation,
        spectral_floor=spectral_floor,
        relative_spectral_floor=relative_spectral_floor,
    )
    covariance_spectrum = covariance_spectrum_from_eta(
        eta,
        tuple(int(v) for v in score.shape[-3:]),
        covariance_floor=covariance_floor,
    )
    return FrameNoiseEstimate(score, autocorrelation, eta, h_spectrum, covariance_spectrum)


@torch.no_grad()
def estimate_frame_noise(
    score_model: nn.Module,
    y: torch.Tensor,
    *,
    kernel_size: int,
    spectral_floor: float,
    relative_spectral_floor: float,
    covariance_floor: float,
) -> FrameNoiseEstimate:
    score = score_model(y)
    return estimate_noise_from_score(
        score,
        kernel_size=kernel_size,
        spectral_floor=spectral_floor,
        relative_spectral_floor=relative_spectral_floor,
        covariance_floor=covariance_floor,
    )


@torch.no_grad()
def zed_denoise(score: torch.Tensor, covariance_spectrum: torch.Tensor) -> torch.Tensor:
    r"""Compute the UNSURE-via-score estimator y + Sigma_eta s(y) correction.

    The returned tensor is the correction ``Sigma_eta s(y)``. Keeping the
    addition to ``y`` at the call site makes it explicit which measurement is
    reconstructed and permits fair clipping only during metric computation.
    """
    if score.shape != covariance_spectrum.shape:
        raise ValueError(
            "score and covariance spectrum must have identical shapes, got "
            f"{tuple(score.shape)} and {tuple(covariance_spectrum.shape)}"
        )
    return torch.fft.ifftn(
        torch.fft.fftn(score, dim=(-3, -2, -1)) * covariance_spectrum,
        dim=(-3, -2, -1),
    ).real


@torch.no_grad()
def latent_covariance_from_noise_samples(
    encoder: nn.Module,
    y: torch.Tensor,
    covariance_spectrum: torch.Tensor,
    *,
    num_samples: int,
    sample_batch_size: int,
    tau: float,
    difference_scheme: Literal["forward", "central"] = "forward",
    storage: Literal["full", "diag"] = "diag",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Estimate J Sigma J^T from direct samples n ~ N(0, Sigma)."""
    if num_samples < 1 or sample_batch_size < 1:
        raise ValueError("num_samples and sample_batch_size must be positive")
    z0 = encoder(y)
    latent_dim = int(z0.shape[-1])
    accumulator = torch.zeros(
        (latent_dim, latent_dim) if storage == "full" else (latent_dim,),
        device=y.device,
        dtype=torch.float32,
    )
    completed = 0
    while completed < num_samples:
        count = min(sample_batch_size, num_samples - completed)
        noise = sample_correlated_noise(covariance_spectrum, num_samples=count, dtype=y.dtype)
        if difference_scheme == "forward":
            delta_z = (encoder(y + float(tau) * noise) - z0) / float(tau)
        elif difference_scheme == "central":
            delta_z = (encoder(y + float(tau) * noise) - encoder(y - float(tau) * noise)) / (2.0 * float(tau))
        else:
            raise ValueError(f"unsupported difference_scheme: {difference_scheme}")
        delta_z = delta_z.float()
        if storage == "full":
            accumulator.add_(delta_z.T @ delta_z)
        else:
            accumulator.add_(delta_z.square().sum(dim=0))
        completed += count
    return z0.detach().cpu(), (accumulator / float(num_samples)).detach().cpu()
