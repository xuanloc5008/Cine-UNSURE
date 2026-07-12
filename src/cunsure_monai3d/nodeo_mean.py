from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _same_padding(kernel_size: int) -> int:
    return int(kernel_size) // 2


class NODEOVelocityNet(nn.Module):
    """ConvNet velocity field V_theta(phi, t) for NODEO mean deformation."""

    def __init__(
        self,
        *,
        in_channels: int = 4,
        channels: tuple[int, ...] = (32, 32, 32),
        kernel_size: int = 3,
        velocity_scale: float = 4.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must contain at least one value")
        layers: list[nn.Module] = []
        prev = int(in_channels)
        pad = _same_padding(kernel_size)
        for width in channels:
            layers.append(nn.Conv3d(prev, int(width), kernel_size=int(kernel_size), padding=pad))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = int(width)
        layers.append(nn.Conv3d(prev, 3, kernel_size=int(kernel_size), padding=pad))
        self.net = nn.Sequential(*layers)
        self.velocity_scale = float(velocity_scale)
        if zero_init:
            last = self.net[-1]
            if isinstance(last, nn.Conv3d):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.net(x)) * self.velocity_scale


@dataclass
class NODEOMeanOutput:
    phi: Tensor
    displacement: Tensor
    velocity: Tensor


class NODEOMeanDeformationModel(nn.Module):
    """NODEO mean trajectory model.

    The state is the full voxel coordinate grid phi(t). A neural velocity field
    integrates phi over observed frame times, matching the NODEO formulation.
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, int, int],
        channels: tuple[int, ...] = (32, 32, 32),
        kernel_size: int = 3,
        velocity_scale: float = 4.0,
        smoothing_kernel: int = 3,
        ode_steps_per_interval: int = 4,
        zero_init_velocity: bool = True,
    ) -> None:
        super().__init__()
        self.image_shape = tuple(int(v) for v in image_shape)
        self.ode_steps_per_interval = int(ode_steps_per_interval)
        self.smoothing_kernel = int(smoothing_kernel)
        self.velocity_net = NODEOVelocityNet(
            in_channels=4,
            channels=tuple(int(v) for v in channels),
            kernel_size=int(kernel_size),
            velocity_scale=float(velocity_scale),
            zero_init=bool(zero_init_velocity),
        )
        grid = self._identity_grid(self.image_shape)
        self.register_buffer("identity_grid", grid, persistent=False)

    @staticmethod
    def _identity_grid(shape: tuple[int, int, int]) -> Tensor:
        vectors = [torch.arange(0, int(s)).float() for s in shape]
        return torch.stack(torch.meshgrid(vectors, indexing="ij"))[None]

    def _normalize_phi(self, phi: Tensor) -> Tensor:
        shape = phi.shape[-3:]
        out = phi.clone()
        for dim, size in enumerate(shape):
            out[:, dim] = 2.0 * (out[:, dim] / max(float(size - 1), 1.0) - 0.5)
        return out

    def velocity(self, phi: Tensor, t: Tensor) -> Tensor:
        batch = phi.shape[0]
        t_channel = t.reshape(1, 1, 1, 1, 1).to(phi).expand(batch, 1, *phi.shape[-3:])
        inp = torch.cat([self._normalize_phi(phi), t_channel], dim=1)
        velocity = self.velocity_net(inp)
        if self.smoothing_kernel > 1:
            pad = self.smoothing_kernel // 2
            velocity = F.avg_pool3d(
                velocity,
                kernel_size=self.smoothing_kernel,
                stride=1,
                padding=pad,
            )
        return velocity

    def integrate_sequence(self, times: Tensor) -> NODEOMeanOutput:
        if times.ndim != 1:
            raise ValueError(f"expected times [T], got {tuple(times.shape)}")
        if times.numel() < 1:
            raise ValueError("times must contain at least one frame")

        identity = self.identity_grid.to(device=times.device, dtype=times.dtype)
        phi = identity.clone()
        phis = [phi.squeeze(0)]
        velocities: list[Tensor] = []

        for idx in range(1, int(times.numel())):
            t0 = times[idx - 1]
            t1 = times[idx]
            dt_total = (t1 - t0).clamp_min(1.0e-6)
            n_steps = max(1, self.ode_steps_per_interval)
            dt = dt_total / float(n_steps)
            interval_velocity: list[Tensor] = []
            for step in range(n_steps):
                t = t0 + dt * float(step)
                v = self.velocity(phi, t)
                phi = phi + dt * v
                interval_velocity.append(v.squeeze(0))
            velocities.append(torch.stack(interval_velocity).mean(dim=0))
            phis.append(phi.squeeze(0))

        phi_seq = torch.stack(phis, dim=0)
        displacement = phi_seq - identity.squeeze(0)[None]
        if velocities:
            velocity_seq = torch.stack([torch.zeros_like(velocities[0]), *velocities], dim=0)
        else:
            velocity_seq = torch.zeros_like(displacement)
        return NODEOMeanOutput(phi=phi_seq, displacement=displacement, velocity=velocity_seq)

