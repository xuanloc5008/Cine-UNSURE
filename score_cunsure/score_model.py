"""Score network and AR-DAE loss adapted from the UNSURE score objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ScoreLossConfig:
    """Annealed denoising score matching settings."""

    delta_min: float = 1.0e-3
    delta_max: float = 1.0e-1
    total_steps: int = 10000


def _group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _conv_nd(spatial_dims: int):
    return {2: nn.Conv2d, 3: nn.Conv3d}[spatial_dims]


def _avg_pool_nd(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return F.avg_pool2d(x, kernel_size=2, stride=2)
    if x.ndim == 5:
        return F.avg_pool3d(x, kernel_size=2, stride=2)
    raise ValueError(f"expected 2D or 3D batch tensor, got {tuple(x.shape)}")


def _interp_mode(spatial_dims: int) -> str:
    return {2: "bilinear", 3: "trilinear"}[spatial_dims]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, spatial_dims: int = 2) -> None:
        super().__init__()
        conv = _conv_nd(spatial_dims)
        self.block = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.GroupNorm(num_groups=_group_count(out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScoreUNet(nn.Module):
    """Small U-Net returning a score map with the same shape as the input.

    This is intentionally lightweight. The original UNSURE repo uses deepinv
    models in experiments; this implementation keeps the score training
    dependency-free and suitable for cine-MRI frame-level covariance estimation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        spatial_dims: int = 2,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if spatial_dims not in {2, 3}:
            raise ValueError("spatial_dims must be 2 or 3")
        self.in_channels = in_channels
        self.depth = depth
        self.spatial_dims = spatial_dims

        downs = []
        channels = base_channels
        downs.append(ConvBlock(in_channels, channels, spatial_dims=spatial_dims))
        for _ in range(1, depth):
            downs.append(ConvBlock(channels, channels * 2, spatial_dims=spatial_dims))
            channels *= 2
        self.downs = nn.ModuleList(downs)
        self.mid = ConvBlock(channels, channels, spatial_dims=spatial_dims)

        ups = []
        for _ in range(depth - 1):
            ups.append(ConvBlock(channels + channels // 2, channels // 2, spatial_dims=spatial_dims))
            channels //= 2
        self.ups = nn.ModuleList(ups)
        self.out = _conv_nd(spatial_dims)(channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_ndim = self.spatial_dims + 2
        if x.ndim != expected_ndim:
            shape = "[B, C, H, W]" if self.spatial_dims == 2 else "[B, C, H, W, D]"
            raise ValueError(f"ScoreUNet expects {shape}, got {tuple(x.shape)}")
        skips = []
        h = x
        for idx, block in enumerate(self.downs):
            h = block(h)
            if idx < len(self.downs) - 1:
                skips.append(h)
                h = _avg_pool_nd(h)
        h = self.mid(h)
        for block in self.ups:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-self.spatial_dims :], mode=_interp_mode(self.spatial_dims), align_corners=False)
            h = block(torch.cat([h, skip], dim=1))
        return self.out(h)


def annealed_tau(
    step: int,
    total_steps: int,
    delta_min: float,
    delta_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Linear annealing from delta_max to delta_min."""
    weight = min(max(step / max(total_steps, 1), 0.0), 1.0)
    value = delta_max * (1.0 - weight) + delta_min * weight
    return torch.tensor(value, device=device, dtype=dtype)


def ardae_score_loss(
    score_model: nn.Module,
    image: torch.Tensor,
    *,
    step: int,
    config: ScoreLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Annealed residual denoising autoencoder score loss.

    Formula used in the UNSURE paper:

        min_psi E || xi + tau s_psi(y + tau xi) ||_2^2

    where xi ~ N(0, I), tau is annealed, and s_psi approximates
    grad_y log p_y(y).
    """
    tau = annealed_tau(
        step,
        config.total_steps,
        config.delta_min,
        config.delta_max,
        device=image.device,
        dtype=image.dtype,
    )
    xi = torch.randn_like(image)
    score = score_model(image + tau * xi)
    residual = xi + tau * score
    loss = residual.pow(2).mean()
    metrics = {
        "loss": loss.detach(),
        "tau": tau.detach(),
        "score_norm": score.pow(2).mean().sqrt().detach(),
    }
    return loss, metrics
