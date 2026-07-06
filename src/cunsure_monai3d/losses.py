from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def circular_conv3d_depthwise(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"expected [B,C,D,H,W], got {tuple(x.shape)}")
    if kernel.ndim != 5:
        raise ValueError(f"expected kernel [1,1,kD,kH,kW], got {tuple(kernel.shape)}")
    k_d, k_h, k_w = kernel.shape[-3:]
    pad = (k_w // 2, k_w // 2, k_h // 2, k_h // 2, k_d // 2, k_d // 2)
    weight = kernel.repeat(x.shape[1], 1, 1, 1, 1)
    return F.conv3d(F.pad(x, pad, mode="circular"), weight, groups=x.shape[1])


@dataclass
class CUNSURELossOutput:
    loss: torch.Tensor
    residual: torch.Tensor
    divergence: torch.Tensor
    eta: torch.Tensor


class MinimaxCUNSURE3DLoss(nn.Module):
    r"""Minimax C-UNSURE loss for 3D denoising.

    Implements the correlated Gaussian UNSURE objective

        min_theta max_eta E ||f_theta(y)-y||^2 + 2 tr(Sigma_eta J_f(y)).

    The trace is approximated with the Monte-Carlo finite-difference estimator

        2 <Sigma_eta b, (f(y + tau b) - f(y)) / tau>,

    where b ~ N(0, I). The eta tensor is updated by gradient ascent outside the
    model optimizer, matching the Lagrange-multiplier update pattern in the
    reference UNSURE repository.
    """

    def __init__(
        self,
        *,
        kernel_size: int,
        eta_init: float,
        tau: float,
        eta_step_size: float,
        eta_momentum: float,
        eta_grad_clip: float | None,
        eta_max_norm: float | None,
        device: torch.device,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        eta = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), device=device)
        eta = eta / eta.sum() * float(eta_init)
        self.eta = nn.Parameter(eta, requires_grad=True)
        self.tau = float(tau)
        self.eta_step_size = float(eta_step_size)
        self.eta_momentum = float(eta_momentum)
        self.eta_grad_clip = None if eta_grad_clip is None else float(eta_grad_clip)
        self.eta_max_norm = None if eta_max_norm is None else float(eta_max_norm)
        self.register_buffer("eta_grad_momentum", torch.zeros_like(eta), persistent=True)
        self._has_momentum = False

    def forward(self, model: nn.Module, y: torch.Tensor) -> CUNSURELossOutput:
        x_net = model(y)
        residual = (x_net - y).pow(2).flatten(start_dim=1).mean(dim=1)

        probe = torch.randn_like(y)
        sigma_probe = circular_conv3d_depthwise(probe, self.eta)
        x_pert = model(y + self.tau * probe)
        fd_jvp = (x_pert - x_net) / self.tau
        divergence = 2.0 * (sigma_probe * fd_jvp).flatten(start_dim=1).mean(dim=1)
        loss = residual + divergence
        return CUNSURELossOutput(
            loss=loss,
            residual=residual.detach(),
            divergence=divergence.detach(),
            eta=self.eta.detach(),
        )

    @torch.no_grad()
    def ascend_eta(self, grad: torch.Tensor) -> None:
        if self.eta_grad_clip is not None:
            grad = grad.clamp(min=-self.eta_grad_clip, max=self.eta_grad_clip)
        if not self._has_momentum:
            self.eta_grad_momentum.copy_(grad)
            self._has_momentum = True
        else:
            self.eta_grad_momentum.mul_(self.eta_momentum).add_(grad, alpha=1.0 - self.eta_momentum)
        self.eta.add_(self.eta_grad_momentum, alpha=self.eta_step_size)
        if self.eta_max_norm is not None:
            eta_norm = self.eta.norm()
            if eta_norm > self.eta_max_norm:
                self.eta.mul_(self.eta_max_norm / eta_norm.clamp_min(1.0e-12))

    def apply_sigma_img(self, volume: torch.Tensor) -> torch.Tensor:
        return circular_conv3d_depthwise(volume, self.eta)
