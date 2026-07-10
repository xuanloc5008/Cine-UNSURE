from __future__ import annotations

import torch
from torch import nn


class LatentEulerDrift(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_invariant: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.time_invariant = bool(time_invariant)
        input_dim = self.latent_dim if self.time_invariant else self.latent_dim + 1
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(int(num_layers)):
            layers.append(nn.Linear(dim, int(hidden_dim)))
            layers.append(nn.SiLU())
            dim = int(hidden_dim)
        layers.append(nn.Linear(dim, self.latent_dim))
        self.net = nn.Sequential(*layers)

    def drift(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.time_invariant:
            x = z
        else:
            x = torch.cat([z, t[..., None]], dim=-1)
        return self.net(x)

    def forward(self, z: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        z0 = z[:-1]
        t0 = times[:-1]
        dt = (times[1:] - times[:-1]).clamp_min(1.0e-6)
        return z0 + dt[:, None] * self.drift(z0, t0)


def gaussian_nll_diag(
    pred: torch.Tensor,
    target: torch.Tensor,
    variance: torch.Tensor,
    *,
    jitter: float,
) -> torch.Tensor:
    var = variance.clamp_min(float(jitter))
    err = target - pred
    return 0.5 * ((err.pow(2) / var) + var.log()).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean()
