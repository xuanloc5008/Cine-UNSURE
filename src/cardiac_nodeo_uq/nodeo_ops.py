from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LocalNCC3D(nn.Module):
    """NODEO-style local normalized cross-correlation loss for 3D images."""

    def __init__(self, win: int = 21, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.win = int(win)
        self.eps = float(eps)

    def _window_sum(self, x: Tensor) -> Tensor:
        half = self.win // 2
        pad = [half + 1, half] * 3
        x_pad = F.pad(x, pad=pad, mode="constant", value=0)
        cs = torch.cumsum(torch.cumsum(torch.cumsum(x_pad, dim=2), dim=3), dim=4)
        sx, sy, sz = x.shape[2:]
        w = self.win
        return (
            cs[:, :, w:, w:, w:]
            - cs[:, :, w:, w:, :sz]
            - cs[:, :, w:, :sy, w:]
            - cs[:, :, :sx, w:, w:]
            + cs[:, :, w:, :sy, :sz]
            + cs[:, :, :sx, w:, :sz]
            + cs[:, :, :sx, :sy, w:]
            - cs[:, :, :sx, :sy, :sz]
        )

    def forward(self, fixed: Tensor, warped: Tensor) -> Tensor:
        fixed = fixed.float()
        warped = warped.float()
        fixed2 = fixed * fixed
        warped2 = warped * warped
        product = fixed * warped

        fixed_sum = self._window_sum(fixed)
        warped_sum = self._window_sum(warped)
        fixed2_sum = self._window_sum(fixed2)
        warped2_sum = self._window_sum(warped2)
        product_sum = self._window_sum(product)

        win_volume = float(self.win**3)
        fixed_mean = fixed_sum / win_volume
        warped_mean = warped_sum / win_volume
        cross = product_sum - warped_mean * fixed_sum - fixed_mean * warped_sum + fixed_mean * warped_mean * win_volume
        fixed_var = fixed2_sum - 2.0 * fixed_mean * fixed_sum + fixed_mean.pow(2) * win_volume
        warped_var = warped2_sum - 2.0 * warped_mean * warped_sum + warped_mean.pow(2) * win_volume
        cc = cross.pow(2) / (fixed_var * warped_var + self.eps)
        return (1.0 - cc.mean()).float()


class SpatialTransformer3D(nn.Module):
    """Voxel-flow spatial transformer following NODEO-DIR conventions."""

    def __init__(self, size: tuple[int, int, int], mode: str = "bilinear") -> None:
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, int(s)) for s in size]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij")).float()[None]
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, src: Tensor, flow: Tensor, *, return_phi: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        new_locs = self.grid.to(flow) + flow
        shape = flow.shape[2:]
        for dim in range(len(shape)):
            new_locs[:, dim] = 2.0 * (new_locs[:, dim] / (shape[dim] - 1.0) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        warped = F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        if return_phi:
            return warped, new_locs
        return warped


def compose_displacements(
    outer: Tensor,
    inner: Tensor,
    *,
    transformer: SpatialTransformer3D | None = None,
) -> Tensor:
    """Compose voxel displacements as ``outer(inner(x))``.

    A displacement maps ``x`` to ``x + displacement(x)``. The returned field
    is therefore ``inner + warp(outer, inner)``.
    """

    if outer.shape != inner.shape or outer.ndim != 5 or outer.shape[1] != 3:
        raise ValueError(
            f"expected matching [B,3,D,H,W] fields, got {tuple(outer.shape)} and {tuple(inner.shape)}"
        )
    spatial_shape = tuple(int(v) for v in outer.shape[-3:])
    transformer = transformer or SpatialTransformer3D(spatial_shape)
    return inner + transformer(outer, inner)


def generate_grid3d_normalized(shape: tuple[int, int, int], *, device: torch.device | None = None) -> Tensor:
    """Return NODEO normalized grid [1, 3, D, H, W] in [-1, 1]."""

    d, h, w = (int(v) for v in shape)
    z = torch.linspace(-1.0, 1.0, d, device=device)
    y = torch.linspace(-1.0, 1.0, h, device=device)
    x = torch.linspace(-1.0, 1.0, w, device=device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([xx, yy, zz], dim=0)[None]


def identity_grid_voxel(shape: tuple[int, int, int], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    vectors = [torch.arange(0, int(s), device=device, dtype=dtype) for s in shape]
    return torch.stack(torch.meshgrid(vectors, indexing="ij"))[None]


def jacobian_det_3d(phi: Tensor) -> Tensor:
    """Finite-difference Jacobian determinant for phi [B,3,D,H,W] or [B,D,H,W,3]."""

    if phi.size(-1) == 3:
        phi = phi.permute(0, 4, 1, 2, 3)
    base = phi[:, :, :-1, :-1, :-1]
    d0 = (phi[:, :, 1:, :-1, :-1] - base).permute(0, 2, 3, 4, 1)
    d1 = (phi[:, :, :-1, 1:, :-1] - base).permute(0, 2, 3, 4, 1)
    d2 = (phi[:, :, :-1, :-1, 1:] - base).permute(0, 2, 3, 4, 1)
    jac = torch.stack([d0, d1, d2], dim=-1)
    return torch.linalg.det(jac)


def negative_jacobian_loss(phi: Tensor, *, margin: float = 0.5) -> Tensor:
    return F.relu(-(jacobian_det_3d(phi) - float(margin))).pow(2).mean()


def nodeo_jacobian_metrics(
    phi: Tensor,
    *,
    minimum: float = 0.5,
    maximum: float = 4.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Two-sided NODEO Jacobian penalty and regularity statistics."""

    determinant = jacobian_det_3d(phi)
    lower_penalty = F.relu(float(minimum) - determinant).pow(2).mean()
    upper_penalty = F.relu(determinant - float(maximum)).pow(2).mean()
    penalty = lower_penalty + upper_penalty
    fold_fraction = (determinant <= 0.0).float().mean()
    volume_deviation = (determinant - 1.0).abs().mean()
    return (
        penalty,
        lower_penalty,
        upper_penalty,
        fold_fraction,
        volume_deviation,
        determinant.min(),
        determinant.max(),
    )


def smoothness_loss(flow: Tensor) -> Tensor:
    return (
        (flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]).pow(2).mean()
        + (flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]).pow(2).mean()
        + (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).pow(2).mean()
    )


def velocity_magnitude_loss(all_velocity: Tensor) -> Tensor:
    """NODEO velocity magnitude loss for [T,B,3,D,H,W] or [B,3,D,H,W]."""

    return all_velocity.pow(2).sum(dim=-4 if all_velocity.ndim == 6 else 1).mean()


def euler_integrate(z0: Tensor, n_steps: int, func, step_size: float) -> Tensor:
    z = z0
    for _ in range(int(n_steps)):
        z = z + float(step_size) * func(z)
    return z


def rk4_integrate(z0: Tensor, n_steps: int, func, step_size: float) -> Tensor:
    z = z0
    h = float(step_size)
    for _ in range(int(n_steps)):
        k1 = h * func(z)
        k2 = h * func(z + 0.5 * k1)
        k3 = h * func(z + 0.5 * k2)
        k4 = h * func(z + k3)
        z = z + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return z
