from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class SDERNNOutput:
    hidden_mean: Tensor
    hidden_covariance: Tensor
    residual_displacement: Tensor
    total_displacement: Tensor
    phi: Tensor


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int, activation: nn.Module | None = None) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        modules: list[nn.Module] = []
        prev = int(in_dim)
        act = activation if activation is not None else nn.Tanh()
        for _ in range(int(layers) - 1):
            modules.append(nn.Linear(prev, int(hidden_dim)))
            modules.append(act.__class__())
            prev = int(hidden_dim)
        modules.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SDEDrift(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.net = MLP(hidden_dim + 1, mlp_hidden_dim, hidden_dim, layers)

    def forward(self, h: Tensor, t: Tensor) -> Tensor:
        t = t.reshape(1).to(h).expand(1)
        return self.net(torch.cat([h, t], dim=0))


class SDEDiffusion(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden_dim: int, layers: int, scale: float) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.net = MLP(hidden_dim + 1, mlp_hidden_dim, hidden_dim * hidden_dim, layers)
        self.scale = float(scale)

    def forward(self, h: Tensor, t: Tensor) -> Tensor:
        t = t.reshape(1).to(h).expand(1)
        raw = self.net(torch.cat([h, t], dim=0)).reshape(self.hidden_dim, self.hidden_dim)
        return torch.tanh(raw) * self.scale


class ResidualDeformationDecoder(nn.Module):
    """Coordinate-conditioned output network c(h, phi_bar, t) -> delta phi.

    The network is shared over voxels, which keeps the output transform faithful to
    the SDE-RNN definition while avoiding a dense per-voxel parameter table.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        image_shape: tuple[int, int, int],
        mlp_hidden_dim: int,
        layers: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.residual_scale = float(residual_scale)
        self.net = MLP(hidden_dim + 4, mlp_hidden_dim, 3, layers)
        coords = self._normalized_coords(self.image_shape)
        self.register_buffer("coords", coords, persistent=False)

    @staticmethod
    def _normalized_coords(shape: tuple[int, int, int]) -> Tensor:
        d, h, w = (int(v) for v in shape)
        z = torch.linspace(-1.0, 1.0, d)
        y = torch.linspace(-1.0, 1.0, h)
        x = torch.linspace(-1.0, 1.0, w)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)

    def forward(self, h: Tensor, t: Tensor) -> Tensor:
        n_vox = self.coords.shape[0]
        coords = self.coords.to(device=h.device, dtype=h.dtype)
        h_repeat = h.reshape(1, -1).expand(n_vox, -1)
        t_repeat = t.reshape(1, 1).to(h).expand(n_vox, 1)
        field = torch.tanh(self.net(torch.cat([coords, h_repeat, t_repeat], dim=1))) * self.residual_scale
        return field.T.reshape(3, *self.image_shape)

    def covariance_factor(self, h: Tensor, p: Tensor, t: Tensor, *, jitter: float = 1.0e-6) -> Tensor:
        """Return exact low-rank factor L_phi such that R_phi = L_phi L_phi^T."""

        h = h.detach().requires_grad_(True)
        columns: list[Tensor] = []
        basis = torch.eye(h.numel(), device=h.device, dtype=h.dtype)

        def decode(inp: Tensor) -> Tensor:
            return self.forward(inp, t).reshape(-1)

        for col in range(h.numel()):
            _, jvp = torch.autograd.functional.jvp(decode, (h,), (basis[col],), create_graph=False, strict=False)
            columns.append(jvp.detach())
        jac = torch.stack(columns, dim=1)
        eye = torch.eye(p.shape[0], device=p.device, dtype=p.dtype)
        chol = torch.linalg.cholesky(p.detach() + float(jitter) * eye)
        return jac @ chol


class NeuralSDERNNUncertainty(nn.Module):
    """Neural SDE-RNN mean/covariance propagation following Dahale et al.

    Hidden mean propagates with an SDE drift. Hidden covariance propagates with
    the linearized SDE covariance equation, and observed frames update the hidden
    covariance through the CVRNN Jacobians.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        image_shape: tuple[int, int, int],
        mlp_hidden_dim: int = 128,
        mlp_layers: int = 2,
        diffusion_scale: float = 0.05,
        init_covariance: float = 1.0e-4,
        residual_scale: float = 2.0,
        sde_steps_per_interval: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.init_covariance = float(init_covariance)
        self.sde_steps_per_interval = int(sde_steps_per_interval)
        self.drift = SDEDrift(hidden_dim, mlp_hidden_dim, mlp_layers)
        self.diffusion = SDEDiffusion(hidden_dim, mlp_hidden_dim, mlp_layers, scale=diffusion_scale)
        self.rnn = nn.GRUCell(latent_dim, hidden_dim)
        self.output = ResidualDeformationDecoder(
            hidden_dim=hidden_dim,
            image_shape=image_shape,
            mlp_hidden_dim=mlp_hidden_dim,
            layers=mlp_layers,
            residual_scale=residual_scale,
        )
        self.h0 = nn.Parameter(torch.zeros(hidden_dim))

    def _drift_jacobian(self, h: Tensor, t: Tensor) -> Tensor:
        h_detached = h.detach().requires_grad_(True)

        def func(inp: Tensor) -> Tensor:
            return self.drift(inp, t.detach())

        return torch.autograd.functional.jacobian(func, h_detached, create_graph=False, strict=False).detach()

    def _rnn_jacobians(self, h: Tensor, z: Tensor) -> tuple[Tensor, Tensor]:
        h_detached = h.detach().requires_grad_(True)
        z_detached = z.detach().requires_grad_(True)

        def with_h(inp: Tensor) -> Tensor:
            return self.rnn(z_detached[None], inp[None]).squeeze(0)

        def with_z(inp: Tensor) -> Tensor:
            return self.rnn(inp[None], h_detached[None]).squeeze(0)

        dh = torch.autograd.functional.jacobian(with_h, h_detached, create_graph=False, strict=False).detach()
        dz = torch.autograd.functional.jacobian(with_z, z_detached, create_graph=False, strict=False).detach()
        return dh, dz

    def _sde_step(self, h: Tensor, p: Tensor, t0: Tensor, t1: Tensor) -> tuple[Tensor, Tensor]:
        dt_total = (t1 - t0).clamp_min(1.0e-6)
        n_steps = max(1, self.sde_steps_per_interval)
        dt = dt_total / float(n_steps)
        eye = torch.eye(self.hidden_dim, device=h.device, dtype=h.dtype)
        for step in range(n_steps):
            t = t0 + dt * float(step)
            f = self.drift(h, t)
            f_h = self._drift_jacobian(h, t).to(p)
            g = self.diffusion(h.detach(), t.detach()).detach().to(p)
            dp = p @ f_h.T + f_h @ p + g @ eye @ g.T
            p_next = p + dt.detach() * dp
            p = 0.5 * (p_next + p_next.T)
            h = h + dt * f
        return h, p

    @staticmethod
    def _obs_covariance_matrix(r: Tensor) -> Tensor:
        if r.ndim == 1:
            return torch.diag(r)
        if r.ndim == 2:
            return r
        raise ValueError(f"expected covariance diag [Z] or full [Z,Z], got {tuple(r.shape)}")

    def _cv_rnn_update(self, h_prior: Tensor, p_prior: Tensor, z: Tensor, r: Tensor) -> tuple[Tensor, Tensor]:
        h = self.rnn(z[None], h_prior[None]).squeeze(0)
        dh, dz = self._rnn_jacobians(h_prior, z)
        sigma = self._obs_covariance_matrix(r).to(p_prior)
        p = dh.to(p_prior) @ p_prior @ dh.to(p_prior).T + dz.to(p_prior) @ sigma @ dz.to(p_prior).T
        p = 0.5 * (p + p.T)
        return h, p

    def forward(self, *, times: Tensor, z: Tensor, r: Tensor, phi_bar: Tensor) -> SDERNNOutput:
        if times.ndim != 1 or z.ndim != 2:
            raise ValueError("times must be [T] and z must be [T,Z]")
        if z.shape[1] != self.latent_dim:
            raise ValueError(f"expected latent_dim={self.latent_dim}, got {z.shape[1]}")
        if phi_bar.shape[0] != z.shape[0]:
            raise ValueError("phi_bar and z must have matching sequence length")

        h = self.h0.to(z)
        p = torch.eye(self.hidden_dim, device=z.device, dtype=z.dtype) * self.init_covariance
        h_list: list[Tensor] = []
        p_list: list[Tensor] = []
        residuals: list[Tensor] = []

        for idx in range(z.shape[0]):
            if idx > 0:
                h, p = self._sde_step(h, p, times[idx - 1], times[idx])
            h, p = self._cv_rnn_update(h, p, z[idx], r[idx])
            residual = self.output(h, times[idx])
            h_list.append(h)
            p_list.append(p)
            residuals.append(residual)

        hidden_mean = torch.stack(h_list)
        hidden_covariance = torch.stack(p_list)
        residual_displacement = torch.stack(residuals)
        total_displacement = (phi_bar - phi_bar[0:1]) + residual_displacement
        phi = phi_bar + residual_displacement
        return SDERNNOutput(
            hidden_mean=hidden_mean,
            hidden_covariance=hidden_covariance,
            residual_displacement=residual_displacement,
            total_displacement=total_displacement,
            phi=phi,
        )
