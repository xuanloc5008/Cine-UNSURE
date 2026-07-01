"""Core score-based C-UNSURE and encoder covariance push-forward.

The functions in this file implement the draft-paper observation model

    Sigma_z = J_E Sigma_img J_E^T

without materializing the encoder Jacobian. The image covariance is estimated
from the C-UNSURE score-autocorrelation formula and pushed through a frozen
encoder with finite-difference probes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class CUNSUREConfig:
    """Configuration for score-based C-UNSURE covariance estimation."""

    radius: int = 5
    eps: float = 1.0e-6
    spectral_floor: float = 1.0e-8
    n_probes: int = 32
    finite_difference_tau: float = 1.0e-2
    unbiased_covariance: bool = True
    circular_autocorrelation: bool = True


@dataclass
class LatentCovarianceResult:
    """Outputs for one score C-UNSURE observation-covariance estimate."""

    z: Tensor
    covariance: Tensor
    deltas: Tensor
    score: Tensor
    autocorrelation: Tensor
    eta_hat: Tensor
    sqrt_spectrum: Tensor


def _spatial_dim_tuple(x: Tensor) -> tuple[int, ...]:
    if x.ndim < 4:
        raise ValueError(f"expected [B, C, *spatial] tensor, got shape {tuple(x.shape)}")
    return tuple(range(2, x.ndim))


def _kernel_dim_tuple(x: Tensor) -> tuple[int, ...]:
    if x.ndim < 2:
        raise ValueError(f"expected [B, *kernel] or [*kernel] tensor, got shape {tuple(x.shape)}")
    return tuple(range(1, x.ndim))


def _valid_slices(size: int, shift: int) -> tuple[slice, slice]:
    if shift >= 0:
        return slice(0, size - shift), slice(shift, size)
    return slice(-shift, size), slice(0, size + shift)


def score_autocorrelation(
    score: Tensor,
    radius: int,
    *,
    circular: bool = True,
    reduce_batch: bool = False,
) -> Tensor:
    """Estimate local score autocorrelation over offsets in [-radius, radius].

    Args:
        score: score map `s(y) ~= grad_y log p_y(y)`, shape [B, C, *spatial].
        radius: maximum offset per spatial axis.
        circular: use circular shifts, matching the circulant covariance model.
        reduce_batch: average the returned autocorrelation over batch.

    Returns:
        Tensor with shape [B, 2r+1, ...] or [2r+1, ...] if `reduce_batch`.
        The zero-lag entry is centered at index `radius` along each kernel axis.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    spatial_dims = _spatial_dim_tuple(score)
    spatial_shape = score.shape[2:]
    offsets = list(product(range(-radius, radius + 1), repeat=len(spatial_shape)))
    values: list[Tensor] = []

    for offset in offsets:
        if circular:
            shifted = torch.roll(score, shifts=offset, dims=spatial_dims)
            prod_value = (score * shifted).flatten(start_dim=1).mean(dim=1)
        else:
            base_index: list[slice] = [slice(None), slice(None)]
            shifted_index: list[slice] = [slice(None), slice(None)]
            for size, shift in zip(spatial_shape, offset, strict=True):
                base_slice, shifted_slice = _valid_slices(size, shift)
                base_index.append(base_slice)
                shifted_index.append(shifted_slice)
            prod_value = (score[tuple(base_index)] * score[tuple(shifted_index)]).flatten(start_dim=1).mean(dim=1)
        values.append(prod_value)

    kernel_shape = (2 * radius + 1,) * len(spatial_shape)
    h = torch.stack(values, dim=1).reshape(score.shape[0], *kernel_shape)
    if reduce_batch:
        h = h.mean(dim=0)
    return h


def eta_from_score_autocorrelation(h: Tensor, eps: float = 1.0e-6, *, batched: bool | None = None) -> Tensor:
    """Closed-form C-UNSURE covariance kernel from score autocorrelation.

    Paper formula, generalized to 2D/3D by separable FFT axes:

        eta_hat = F^{-1}(1 / F h)

    The input `h` is expected to have zero lag centered. The returned `eta_hat`
    keeps the same centered-kernel convention.
    """
    if batched is None:
        # Most calls in this project are batched 2D kernels [B, K, K].
        # Pass batched=False for a single unbatched 3D kernel [K, K, K].
        batched = h.ndim >= 3

    squeeze_batch = not batched
    if squeeze_batch:
        h = h.unsqueeze(0)
    kernel_dims = _kernel_dim_tuple(h)

    h_origin = torch.fft.ifftshift(h, dim=kernel_dims)
    h_spectrum = torch.fft.fftn(h_origin, dim=kernel_dims)
    inv_spectrum_real = torch.reciprocal(torch.clamp(h_spectrum.real, min=eps))
    inv_spectrum = torch.complex(inv_spectrum_real, torch.zeros_like(inv_spectrum_real))
    eta_origin = torch.fft.ifftn(inv_spectrum, dim=kernel_dims).real
    eta_centered = torch.fft.fftshift(eta_origin, dim=kernel_dims)
    return eta_centered.squeeze(0) if squeeze_batch else eta_centered


def _centered_kernel_to_full_spectrum(
    kernel: Tensor,
    spatial_shape: tuple[int, ...],
    *,
    eps: float,
) -> Tensor:
    """Embed a centered local kernel into a full circulant image spectrum."""
    squeeze_batch = kernel.ndim == len(spatial_shape)
    if squeeze_batch:
        kernel = kernel.unsqueeze(0)
    if kernel.ndim != len(spatial_shape) + 1:
        raise ValueError(
            f"kernel must have shape [B, *kernel] or [*kernel], got {tuple(kernel.shape)} "
            f"for spatial shape {spatial_shape}"
        )

    kernel_dims = _kernel_dim_tuple(kernel)
    kernel_origin = torch.fft.ifftshift(kernel, dim=kernel_dims)
    full = torch.zeros((kernel.shape[0], *spatial_shape), dtype=kernel.dtype, device=kernel.device)
    insert = (slice(None), *[slice(0, min(k, s)) for k, s in zip(kernel_origin.shape[1:], spatial_shape, strict=True)])
    source = (slice(None), *[slice(0, min(k, s)) for k, s in zip(kernel_origin.shape[1:], spatial_shape, strict=True)])
    full[insert] = kernel_origin[source]
    full_spectrum = torch.fft.fftn(full, dim=tuple(range(1, full.ndim))).real
    full_spectrum = torch.clamp(full_spectrum, min=eps)
    return full_spectrum.squeeze(0) if squeeze_batch else full_spectrum


def sqrt_spectrum_from_eta(
    eta_hat: Tensor,
    spatial_shape: tuple[int, ...],
    *,
    eps: float = 1.0e-8,
) -> Tensor:
    """Return the full-image square-root covariance spectrum for sampling."""
    spectrum = _centered_kernel_to_full_spectrum(eta_hat, spatial_shape, eps=eps)
    return torch.sqrt(spectrum)


def _broadcast_spectrum(spectrum: Tensor, reference: Tensor) -> Tensor:
    spatial_ndim = reference.ndim - 2
    if spectrum.ndim == spatial_ndim:
        return spectrum.reshape((1, 1, *spectrum.shape))
    if spectrum.ndim == spatial_ndim + 1:
        return spectrum.unsqueeze(1)
    raise ValueError(
        f"spectrum shape {tuple(spectrum.shape)} cannot broadcast to reference shape {tuple(reference.shape)}"
    )


def sample_correlated_noise_like(
    reference: Tensor,
    sqrt_spectrum: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample `n = kappa * xi` with covariance defined by `sqrt_spectrum`."""
    spatial_dims = _spatial_dim_tuple(reference)
    xi = torch.randn(reference.shape, dtype=reference.dtype, device=reference.device, generator=generator)
    xi_fft = torch.fft.fftn(xi, dim=spatial_dims)
    spectrum = _broadcast_spectrum(sqrt_spectrum.to(device=reference.device, dtype=reference.dtype), reference)
    return torch.fft.ifftn(xi_fft * spectrum, dim=spatial_dims).real


def covariance_from_deltas(deltas: Tensor, *, unbiased: bool = True) -> Tensor:
    """Compute per-batch covariance from finite-difference latent deltas.

    Args:
        deltas: shape [S, B, d].

    Returns:
        covariance: shape [B, d, d].
    """
    if deltas.ndim != 3:
        raise ValueError(f"expected deltas [S, B, d], got {tuple(deltas.shape)}")
    samples = deltas.transpose(0, 1)  # [B, S, d]
    samples = samples - samples.mean(dim=1, keepdim=True)
    denom = max(samples.shape[1] - 1, 1) if unbiased else samples.shape[1]
    cov = samples.transpose(1, 2) @ samples / denom
    return 0.5 * (cov + cov.transpose(-1, -2))


def estimate_latent_covariance(
    image: Tensor,
    score_model: torch.nn.Module | Callable[[Tensor], Tensor],
    encoder: torch.nn.Module | Callable[[Tensor], Tensor],
    config: CUNSUREConfig,
    *,
    generator: torch.Generator | None = None,
) -> LatentCovarianceResult:
    """Estimate latent observation covariance for a frozen image encoder.

    Args:
        image: normalized image tensor, shape [B, C, *spatial].
        score_model: returns score map with the same shape as `image`.
        encoder: frozen encoder returning [B, d] latent observations.
        config: C-UNSURE and finite-difference settings.
        generator: optional torch random generator.
    """
    if image.ndim < 4:
        raise ValueError("image must be [B, C, *spatial]")

    with torch.no_grad():
        score = score_model(image)
        if score.shape != image.shape:
            raise ValueError(f"score shape {tuple(score.shape)} must match image shape {tuple(image.shape)}")
        h = score_autocorrelation(
            score,
            radius=config.radius,
            circular=config.circular_autocorrelation,
            reduce_batch=False,
        )
        eta_hat = eta_from_score_autocorrelation(h, eps=config.eps)
        sqrt_spectrum = sqrt_spectrum_from_eta(eta_hat, image.shape[2:], eps=config.spectral_floor)
        z0 = encoder(image)
        z0 = z0.flatten(start_dim=1)

        deltas = []
        for _ in range(config.n_probes):
            noise = sample_correlated_noise_like(image, sqrt_spectrum, generator=generator)
            z1 = encoder(image + config.finite_difference_tau * noise).flatten(start_dim=1)
            deltas.append((z1 - z0) / config.finite_difference_tau)
        delta_tensor = torch.stack(deltas, dim=0)
        covariance = covariance_from_deltas(delta_tensor, unbiased=config.unbiased_covariance)

    return LatentCovarianceResult(
        z=z0,
        covariance=covariance,
        deltas=delta_tensor,
        score=score,
        autocorrelation=h,
        eta_hat=eta_hat,
        sqrt_spectrum=sqrt_spectrum,
    )
