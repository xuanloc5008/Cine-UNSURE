from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchdiffeq import odeint_adjoint


class AveragingKernel3D(nn.Module):
    def __init__(self, window: int) -> None:
        super().__init__()
        self.window = int(window)

    def forward(self, velocity: Tensor) -> Tensor:
        return F.avg_pool3d(
            velocity,
            kernel_size=self.window,
            stride=1,
            padding=self.window // 2,
            count_include_pad=False,
        )


class GaussianKernel3D(nn.Module):
    def __init__(self, window: int, sigma: float) -> None:
        super().__init__()
        radius = (int(window) - 1) / 2.0
        coordinates = torch.linspace(-radius, radius, int(window))
        kernel = torch.exp(-0.5 * (coordinates / float(sigma)).pow(2))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel_d", kernel.view(1, 1, -1, 1, 1), persistent=False)
        self.register_buffer("kernel_h", kernel.view(1, 1, 1, -1, 1), persistent=False)
        self.register_buffer("kernel_w", kernel.view(1, 1, 1, 1, -1), persistent=False)
        self.padding = int(window) // 2

    def forward(self, velocity: Tensor) -> Tensor:
        channels = velocity.shape[1]
        kd = self.kernel_d.to(velocity).expand(channels, 1, -1, -1, -1)
        kh = self.kernel_h.to(velocity).expand(channels, 1, -1, -1, -1)
        kw = self.kernel_w.to(velocity).expand(channels, 1, -1, -1, -1)
        velocity = F.conv3d(velocity, kd, padding=(self.padding, 0, 0), groups=channels)
        velocity = F.conv3d(velocity, kh, padding=(0, self.padding, 0), groups=channels)
        return F.conv3d(velocity, kw, padding=(0, 0, self.padding), groups=channels)


class NODEODIRVelocityNet(nn.Module):
    """BrainNet-style velocity parameterization from the NODEO-DIR repository.

    The sequential extension concatenates time to the transformed coordinate grid,
    as described in Eq. (8) and Section 3.4 of the sequential NODEO paper.
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int],
        encoder_channels: int = 32,
        encoder_depth: int = 5,
        output_downsamples: int = 2,
        bottleneck_dim: int = 16,
        smoothing_kernel: str = "gaussian",
        smoothing_window: int = 15,
        smoothing_sigma: float = 3.0,
        smoothing_passes: int = 1,
    ) -> None:
        super().__init__()
        self.image_shape = tuple(int(v) for v in image_shape)
        self.output_downsamples = int(output_downsamples)
        self.smoothing_passes = int(smoothing_passes)
        layers: list[nn.Module] = []
        in_channels = 4
        for _ in range(int(encoder_depth)):
            layers.append(
                nn.Conv3d(
                    in_channels,
                    int(encoder_channels),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    padding_mode="replicate",
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_channels = int(encoder_channels)
        self.encoder = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 4, *self.image_shape)
            dummy = F.interpolate(dummy, scale_factor=0.5, mode="trilinear", align_corners=True)
            encoded_features = int(self.encoder(dummy).numel())
        self.fc1 = nn.Linear(encoded_features, int(bottleneck_dim))
        coarse_shape = tuple(int(math.ceil(size / (2**self.output_downsamples))) for size in self.image_shape)
        self.coarse_shape = coarse_shape
        self.fc2 = nn.Linear(int(bottleneck_dim), 3 * math.prod(coarse_shape))
        if smoothing_kernel.lower() == "average":
            self.smoother: nn.Module = AveragingKernel3D(smoothing_window)
        elif smoothing_kernel.lower() == "gaussian":
            self.smoother = GaussianKernel3D(smoothing_window, smoothing_sigma)
        else:
            raise ValueError("smoothing_kernel must be 'gaussian' or 'average'")

    def forward(self, phi_normalized: Tensor, t: Tensor) -> Tensor:
        batch = phi_normalized.shape[0]
        time_channel = t.reshape(1, 1, 1, 1, 1).to(phi_normalized).expand(batch, 1, *self.image_shape)
        features = torch.cat([phi_normalized, time_channel], dim=1)
        features = F.interpolate(features, scale_factor=0.5, mode="trilinear", align_corners=True)
        features = self.encoder(features).flatten(1)
        features = F.relu(self.fc1(features))
        velocity = self.fc2(features).view(batch, 3, *self.coarse_shape)
        velocity = F.interpolate(velocity, size=self.image_shape, mode="trilinear", align_corners=True)
        for _ in range(self.smoothing_passes):
            velocity = self.smoother(velocity)
        return velocity


@dataclass
class NODEODIROutput:
    phi_normalized: Tensor
    phi_voxel: Tensor
    displacement_voxel: Tensor
    velocity_normalized: Tensor


class NODEODIRModel(nn.Module):
    """Per-sequence NODEO deformation optimizer with a full coordinate-grid state."""

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int],
        solver: str = "rk4",
        step_size: float = 0.05,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-8,
        **velocity_kwargs: object,
    ) -> None:
        super().__init__()
        self.image_shape = tuple(int(v) for v in image_shape)
        self.solver = str(solver).lower()
        if self.solver not in {"euler", "rk4", "dopri5"}:
            raise ValueError("solver must be 'euler', 'rk4', or 'dopri5'")
        self.step_size = float(step_size)
        self.rtol = float(rtol)
        self.atol = float(atol)
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError("rtol and atol must be positive")
        self.velocity_net = NODEODIRVelocityNet(image_shape=self.image_shape, **velocity_kwargs)
        self.register_buffer("identity_normalized", self._identity_grid(self.image_shape), persistent=False)

    @staticmethod
    def _identity_grid(shape: tuple[int, int, int]) -> Tensor:
        d, h, w = shape
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, d),
            torch.linspace(-1.0, 1.0, h),
            torch.linspace(-1.0, 1.0, w),
            indexing="ij",
        )
        return torch.stack([zz, yy, xx], dim=0)[None]

    def velocity(self, phi: Tensor, t: Tensor) -> Tensor:
        return self.velocity_net(phi, t)

    def forward(self, t: Tensor, phi: Tensor) -> Tensor:
        return self.velocity(phi, t)

    def _to_voxel(self, phi: Tensor) -> Tensor:
        scale = phi.new_tensor(self.image_shape).view(1, 3, 1, 1, 1) - 1.0
        return (phi + 1.0) * 0.5 * scale

    def integrate_sequence(self, times: Tensor) -> NODEODIROutput:
        if times.ndim != 1 or times.numel() < 1:
            raise ValueError(f"expected non-empty times [T], got {tuple(times.shape)}")
        phi0 = self.identity_normalized.to(times).clone()
        if times.numel() == 1:
            phi_normalized = phi0[None]
        else:
            # Fixed-step methods consume step_size. Dopri5 instead adapts its
            # internal evaluations to rtol/atol, and every evaluation passes
            # through velocity_net where the Gaussian K smoothing is applied.
            options = {"step_size": self.step_size} if self.solver in {"euler", "rk4"} else None
            phi_normalized = odeint_adjoint(
                self,
                phi0,
                times,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver,
                options=options,
                adjoint_rtol=self.rtol,
                adjoint_atol=self.atol,
                adjoint_method=self.solver,
                adjoint_options=options,
            ).squeeze(1)
        phi_voxel = self._to_voxel(phi_normalized)
        identity_voxel = self._to_voxel(self.identity_normalized.to(times)).squeeze(0)
        displacement = phi_voxel - identity_voxel[None]
        if times.numel() > 1:
            dt = (times[1:] - times[:-1]).clamp_min(1.0e-6).view(-1, 1, 1, 1, 1)
            interval_velocities = (phi_normalized[1:] - phi_normalized[:-1]) / dt
            velocities = torch.cat([torch.zeros_like(interval_velocities[0:1]), interval_velocities], dim=0)
        else:
            velocities = torch.zeros_like(phi_normalized)
        return NODEODIROutput(
            phi_normalized=phi_normalized,
            phi_voxel=phi_voxel,
            displacement_voxel=displacement,
            velocity_normalized=velocities,
        )
