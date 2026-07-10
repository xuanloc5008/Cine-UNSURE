from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cunsure_monai3d.nodeo_ops import SpatialTransformer3D, identity_grid_voxel


def as_spacing_tensor(spacing_mm: tuple[float, float, float], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    spacing = torch.tensor(spacing_mm, device=device, dtype=dtype)
    if torch.any(spacing <= 0):
        raise ValueError(f"spacing values must be positive, got {spacing_mm}")
    return spacing


def resize_mask(mask: Tensor, size: tuple[int, int, int]) -> Tensor:
    if mask.ndim == 3:
        mask = mask[None, None]
    elif mask.ndim == 4:
        mask = mask[None]
    if tuple(mask.shape[-3:]) != tuple(size):
        mask = F.interpolate(mask.float(), size=size, mode="nearest")
    return mask.float()


def warp_mask(mask: Tensor, displacement: Tensor, *, mode: str = "bilinear") -> Tensor:
    size = tuple(int(v) for v in displacement.shape[-3:])
    transformer = SpatialTransformer3D(size, mode=mode).to(displacement.device)
    return transformer(mask.to(displacement), displacement[None]).squeeze(0)


def volume_from_mask(mask: Tensor, *, voxel_volume: float = 1.0) -> Tensor:
    return mask.sum() * float(voxel_volume)


def volume_from_deformation(reference_mask: Tensor, displacement: Tensor, *, voxel_volume: float = 1.0) -> Tensor:
    warped = warp_mask(reference_mask, displacement, mode="bilinear")
    return volume_from_mask(warped, voxel_volume=voxel_volume)


def ejection_fraction(v_ed: Tensor, v_es: Tensor) -> Tensor:
    return (v_ed - v_es) / v_ed.clamp_min(1.0e-6)


def mean_wall_motion(reference_mask: Tensor, displacement: Tensor) -> Tensor:
    return mean_wall_motion_mm(reference_mask, displacement, spacing_mm=(1.0, 1.0, 1.0))


def mean_wall_motion_mm(
    reference_mask: Tensor,
    displacement: Tensor,
    *,
    spacing_mm: tuple[float, float, float],
) -> Tensor:
    mask = reference_mask.to(displacement).clamp(0, 1)
    while mask.ndim < displacement.ndim:
        mask = mask.squeeze(0)
    if mask.ndim == 3:
        mask = mask[None]
    spacing = as_spacing_tensor(spacing_mm, device=displacement.device, dtype=displacement.dtype)
    displacement_mm = displacement * spacing[:, None, None, None]
    magnitude = displacement_mm.pow(2).sum(dim=0, keepdim=True).sqrt()
    return (magnitude * mask).sum() / mask.sum().clamp_min(1.0)


def deformation_gradient(
    displacement: Tensor,
    *,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    """Return physical finite-difference deformation gradient [3,3,D-1,H-1,W-1]."""

    size = tuple(int(v) for v in displacement.shape[-3:])
    phi = identity_grid_voxel(size, device=displacement.device, dtype=displacement.dtype).squeeze(0) + displacement
    base = phi[:, :-1, :-1, :-1]
    d0 = phi[:, 1:, :-1, :-1] - base
    d1 = phi[:, :-1, 1:, :-1] - base
    d2 = phi[:, :-1, :-1, 1:] - base
    grad_voxel = torch.stack([d0, d1, d2], dim=1)
    spacing = as_spacing_tensor(spacing_mm, device=displacement.device, dtype=displacement.dtype)
    return grad_voxel * spacing[:, None, None, None, None] / spacing[None, :, None, None, None]


def mean_green_lagrange_strain(
    reference_mask: Tensor,
    displacement: Tensor,
    *,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Tensor]:
    grad = deformation_gradient(displacement, spacing_mm=spacing_mm)
    fmat = grad.permute(2, 3, 4, 0, 1)
    eye = torch.eye(3, device=displacement.device, dtype=displacement.dtype)
    strain = 0.5 * (torch.matmul(fmat.transpose(-1, -2), fmat) - eye)
    mask = reference_mask.to(displacement).squeeze()
    mask = mask[:-1, :-1, :-1].clamp(0, 1)
    denom = mask.sum().clamp_min(1.0)
    return {
        "strain_xx": (strain[..., 0, 0] * mask).sum() / denom,
        "strain_yy": (strain[..., 1, 1] * mask).sum() / denom,
        "strain_zz": (strain[..., 2, 2] * mask).sum() / denom,
    }


def delta_variance_diag(metric: Tensor, displacement: Tensor, covariance_diag: Tensor | None) -> Tensor | None:
    if covariance_diag is None:
        return None
    grad = torch.autograd.grad(metric, displacement, retain_graph=True, allow_unused=False)[0]
    return (grad.pow(2) * covariance_diag.to(grad)).sum()


def delta_variance_block(metric: Tensor, displacement: Tensor, covariance_blocks: Tensor | None) -> Tensor | None:
    if covariance_blocks is None:
        return None
    grad = torch.autograd.grad(metric, displacement, retain_graph=True, allow_unused=False)[0]
    g = grad.permute(1, 2, 3, 0)[..., None]
    blocks = covariance_blocks.to(grad)
    return torch.matmul(torch.matmul(g.transpose(-1, -2), blocks), g).sum()
