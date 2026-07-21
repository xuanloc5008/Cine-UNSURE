from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class PhotometricEvidence:
    corrected: Tensor
    raw_residual: Tensor
    corrected_residual: Tensor
    intensity_change: Tensor
    intensity_explained_fraction: Tensor
    structural_residual: Tensor
    gradient_residual: Tensor
    edge_confidence: Tensor
    gain: Tensor
    bias: Tensor


@dataclass
class ResidualDecomposition:
    image_quality_map: Tensor
    deformation_ambiguity_map: Tensor
    intensity_change_map: Tensor
    artifact_residual_map: Tensor
    structural_residual_map: Tensor
    gradient_residual_map: Tensor
    inverse_consistency_map: Tensor
    jacobian_violation_map: Tensor
    component_scales: dict[str, Tensor]


def _validate_window(window: int) -> int:
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("local window must be a positive odd integer")
    return window


def local_mean(values: Tensor, window: int) -> Tensor:
    window = _validate_window(window)
    padding = window // 2
    padded = F.pad(
        values,
        (padding, padding, padding, padding, padding, padding),
        mode="replicate",
    )
    return F.avg_pool3d(padded, kernel_size=window, stride=1)


def local_affine_intensity_correction(
    target: Tensor,
    warped: Tensor,
    *,
    window: int,
    gain_min: float,
    gain_max: float,
    bias_min: float,
    bias_max: float,
    epsilon: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit a slowly varying ``target ~= gain * warped + bias`` model."""

    if target.shape != warped.shape or target.ndim != 5:
        raise ValueError(
            f"target and warped must be matching [T,C,D,H,W], got {target.shape} and {warped.shape}"
        )
    target = target.float()
    warped = warped.float()
    target_mean = local_mean(target, window)
    warped_mean = local_mean(warped, window)
    covariance = local_mean(target * warped, window) - target_mean * warped_mean
    warped_variance = local_mean(warped.square(), window) - warped_mean.square()
    gain = covariance / warped_variance.clamp_min(float(epsilon))
    gain = gain.clamp(float(gain_min), float(gain_max))
    bias = (target_mean - gain * warped_mean).clamp(float(bias_min), float(bias_max))
    corrected = (gain * warped + bias).clamp(0.0, 1.0)
    return corrected, gain, bias


def _gradient(values: Tensor) -> Tensor:
    gradients = torch.gradient(values.float(), dim=(2, 3, 4), edge_order=1)
    return torch.cat(gradients, dim=1)


def photometric_evidence(
    target: Tensor,
    warped: Tensor,
    *,
    intensity_window: int,
    structural_window: int,
    gain_min: float,
    gain_max: float,
    bias_min: float,
    bias_max: float,
    epsilon: float,
) -> PhotometricEvidence:
    corrected, gain, bias = local_affine_intensity_correction(
        target,
        warped,
        window=intensity_window,
        gain_min=gain_min,
        gain_max=gain_max,
        bias_min=bias_min,
        bias_max=bias_max,
        epsilon=epsilon,
    )
    raw_residual = (target.float() - warped.float()).square().mean(dim=1)
    corrected_residual = (target.float() - corrected).square().mean(dim=1)
    intensity_change = (corrected - warped.float()).square().mean(dim=1)
    explained = (
        (raw_residual - corrected_residual).clamp_min(0.0)
        / raw_residual.clamp_min(float(epsilon))
    ).clamp(0.0, 1.0)

    target_mean = local_mean(target.float(), structural_window)
    corrected_mean = local_mean(corrected, structural_window)
    target_centered = target.float() - target_mean
    corrected_centered = corrected - corrected_mean
    covariance = local_mean(target_centered * corrected_centered, structural_window)
    target_variance = local_mean(target_centered.square(), structural_window)
    corrected_variance = local_mean(corrected_centered.square(), structural_window)
    local_correlation = covariance / (
        target_variance * corrected_variance
    ).clamp_min(float(epsilon)).sqrt()
    structural_residual = (0.5 * (1.0 - local_correlation.clamp(-1.0, 1.0))).mean(dim=1)

    target_gradient = _gradient(target)
    corrected_gradient = _gradient(corrected)
    target_norm = target_gradient.square().sum(dim=1).sqrt()
    corrected_norm = corrected_gradient.square().sum(dim=1).sqrt()
    cosine = (target_gradient * corrected_gradient).sum(dim=1) / (
        target_norm * corrected_norm
    ).clamp_min(float(epsilon))
    gradient_residual = 0.5 * (1.0 - cosine.clamp(-1.0, 1.0))
    edge_values = target_norm[1:].reshape(-1) if len(target_norm) > 1 else target_norm.reshape(-1)
    edge_scale = torch.quantile(edge_values, 0.95).clamp_min(float(epsilon))
    edge_confidence = (target_norm / edge_scale).clamp(0.0, 1.0)
    gradient_residual = gradient_residual * edge_confidence

    for value in (
        raw_residual,
        corrected_residual,
        intensity_change,
        explained,
        structural_residual,
        gradient_residual,
        edge_confidence,
    ):
        value[0].zero_()
    return PhotometricEvidence(
        corrected=corrected,
        raw_residual=raw_residual,
        corrected_residual=corrected_residual,
        intensity_change=intensity_change,
        intensity_explained_fraction=explained,
        structural_residual=structural_residual,
        gradient_residual=gradient_residual,
        edge_confidence=edge_confidence,
        gain=gain,
        bias=bias,
    )


def robust_sequence_normalize(
    values: Tensor,
    *,
    quantile: float,
    clip: float,
    epsilon: float,
    positive_only: bool = False,
) -> tuple[Tensor, Tensor]:
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("normalization quantile must be in (0, 1]")
    candidates = values[1:].reshape(-1) if len(values) > 1 else values.reshape(-1)
    if positive_only:
        positive = candidates[candidates > float(epsilon)]
        if positive.numel() > 0:
            candidates = positive
    scale = torch.quantile(candidates.float(), float(quantile)).clamp_min(float(epsilon))
    normalized = (values.float() / scale).clamp(0.0, float(clip))
    normalized[0].zero_()
    return normalized, scale


def decompose_residuals(
    *,
    forward: PhotometricEvidence,
    backward_structural_target_space: Tensor,
    backward_gradient_target_space: Tensor,
    backward_image_quality_target_space: Tensor,
    inverse_consistency: Tensor,
    jacobian_violation: Tensor,
    quantile: float,
    clip: float,
    epsilon: float,
    weights: dict[str, float],
) -> ResidualDecomposition:
    structural = 0.5 * (
        forward.structural_residual + backward_structural_target_space
    )
    gradient = 0.5 * (
        forward.gradient_residual + backward_gradient_target_space
    )
    intensity_change = forward.intensity_change.sqrt()
    artifact = (
        forward.corrected_residual.sqrt() * (1.0 - forward.edge_confidence)
        + backward_image_quality_target_space
    )

    components: dict[str, Tensor] = {
        "intensity": intensity_change,
        "artifact": artifact,
        "structural": structural,
        "gradient": gradient,
        "inverse": inverse_consistency,
        "jacobian": jacobian_violation,
    }
    normalized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    for name, values in components.items():
        normalized[name], scales[name] = robust_sequence_normalize(
            values,
            quantile=quantile,
            clip=clip,
            epsilon=epsilon,
            positive_only=name in {"inverse", "jacobian"},
        )

    image_names = ("intensity", "artifact")
    image_denominator = sum(
        max(float(weights.get(f"image_{name}", 0.0)), 0.0)
        for name in image_names
    )
    if image_denominator <= 0.0:
        raise ValueError("at least one image-quality weight must be positive")
    image_quality = sum(
        max(float(weights.get(f"image_{name}", 0.0)), 0.0)
        * normalized[name]
        for name in image_names
    ) / image_denominator
    deformation_names = ("structural", "gradient", "inverse", "jacobian")
    denominator = sum(max(float(weights.get(name, 0.0)), 0.0) for name in deformation_names)
    if denominator <= 0.0:
        raise ValueError("at least one deformation ambiguity weight must be positive")
    deformation = sum(
        max(float(weights.get(name, 0.0)), 0.0) * normalized[name]
        for name in deformation_names
    ) / denominator
    image_quality[0].zero_()
    deformation[0].zero_()
    return ResidualDecomposition(
        image_quality_map=image_quality,
        deformation_ambiguity_map=deformation,
        intensity_change_map=normalized["intensity"],
        artifact_residual_map=normalized["artifact"],
        structural_residual_map=normalized["structural"],
        gradient_residual_map=normalized["gradient"],
        inverse_consistency_map=normalized["inverse"],
        jacobian_violation_map=normalized["jacobian"],
        component_scales=scales,
    )
