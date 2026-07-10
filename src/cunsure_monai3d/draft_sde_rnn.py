from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = int(input_dim)
    for _ in range(int(num_layers)):
        layers.append(nn.Linear(dim, int(hidden_dim)))
        layers.append(nn.SiLU())
        dim = int(hidden_dim)
    layers.append(nn.Linear(dim, int(output_dim)))
    return nn.Sequential(*layers)


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.T)


def matrix_from_covariance(covariance: torch.Tensor) -> torch.Tensor:
    if covariance.ndim == 1:
        return torch.diag(covariance)
    if covariance.ndim == 2:
        return covariance
    raise ValueError(f"expected covariance [d] or [d,d], got {tuple(covariance.shape)}")


def covariance_diag(covariance: torch.Tensor) -> torch.Tensor:
    if covariance.ndim == 1:
        return covariance
    if covariance.ndim == 2:
        return covariance.diag()
    raise ValueError(f"expected covariance [d] or [d,d], got {tuple(covariance.shape)}")


def jacobian_matrix(fn, x: torch.Tensor, *, create_graph: bool, vectorize: bool) -> torch.Tensor:
    x = x.detach().requires_grad_(True) if not create_graph else x.requires_grad_(True)
    y = fn(x)
    jac = torch.autograd.functional.jacobian(fn, x, create_graph=create_graph, vectorize=vectorize)
    return jac.reshape(y.numel(), x.numel())


def gaussian_nll_from_diag(error: torch.Tensor, variance: torch.Tensor, *, jitter: float) -> torch.Tensor:
    variance = variance.clamp_min(float(jitter))
    return 0.5 * ((error.pow(2) / variance) + variance.log()).mean()


@dataclass
class DraftSDERNNOutput:
    loss: torch.Tensor
    mse: torch.Tensor
    nll: torch.Tensor
    mean_innovation_ratio: torch.Tensor
    final_calibration: torch.Tensor


@dataclass
class DeformationPrediction:
    displacement: torch.Tensor
    covariance_diag: torch.Tensor | None
    covariance_blocks: torch.Tensor | None


class DraftNeuralSDERNN(nn.Module):
    r"""Neural SDE-RNN core following the draft formulation.

    It implements the latent-space version of:
      - SDE mean/covariance propagation, Eqs. (13)-(14)
      - CV-GRU covariance update, Eq. (16)
      - innovation covariance and adaptive process scaling, Eqs. (37)-(41)

    For tractability, the observation likelihood can use the diagonal of the
    innovation covariance. The covariance propagation itself still uses dense
    hidden-state covariance P and the dense/diagonal observation covariance.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
        mlp_layers: int,
        process_noise_floor: float,
        init_covariance: float,
        calibration_lambda: float,
        covariance_grad: bool = False,
        jacobian_vectorize: bool = False,
        innovation_mode: Literal["diag", "full"] = "diag",
        deformation_shape: tuple[int, int, int] | None = None,
        deformation_covariance: Literal["none", "diag", "block"] = "none",
        deformation_jacobian_chunk: int = 512,
        deformation_scale: float = 4.0,
        zero_init_deformation: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.process_noise_floor = float(process_noise_floor)
        self.init_covariance = float(init_covariance)
        self.calibration_lambda = float(calibration_lambda)
        self.covariance_grad = bool(covariance_grad)
        self.jacobian_vectorize = bool(jacobian_vectorize)
        self.innovation_mode = innovation_mode
        self.deformation_shape = deformation_shape
        self.deformation_covariance = deformation_covariance
        self.deformation_jacobian_chunk = int(deformation_jacobian_chunk)
        self.deformation_scale = float(deformation_scale)

        self.init_encoder = nn.Linear(self.latent_dim, self.hidden_dim)
        self.drift_net = make_mlp(self.hidden_dim + 1, mlp_hidden_dim, self.hidden_dim, mlp_layers)
        self.diffusion_net = make_mlp(self.hidden_dim + 1, mlp_hidden_dim, self.hidden_dim, mlp_layers)
        self.gru_cell = nn.GRUCell(self.latent_dim, self.hidden_dim)
        self.obs_decoder = make_mlp(self.hidden_dim + 1, mlp_hidden_dim, self.latent_dim, mlp_layers)
        self.deformation_decoder: nn.Module | None = None
        if deformation_shape is not None:
            deformation_dim = 3
            for value in deformation_shape:
                deformation_dim *= int(value)
            self.deformation_decoder = make_mlp(self.hidden_dim + 1, mlp_hidden_dim, deformation_dim, mlp_layers)
            if zero_init_deformation:
                last = self.deformation_decoder[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    nn.init.zeros_(last.bias)

    def _time_cat(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, t.reshape(1).to(h).expand(1)], dim=0)

    def drift(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.drift_net(self._time_cat(h, t))

    def diffusion_diag(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.diffusion_net(self._time_cat(h, t))) + self.process_noise_floor

    def decode_observation(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.obs_decoder(self._time_cat(h, t))

    def decode_deformation_flat(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.deformation_decoder is None:
            raise RuntimeError("deformation decoder is disabled; set model.deformation_shape in config")
        return torch.tanh(self.deformation_decoder(self._time_cat(h, t))) * self.deformation_scale

    def decode_deformation(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat = self.decode_deformation_flat(h, t)
        if self.deformation_shape is None:
            raise RuntimeError("deformation decoder is disabled; set model.deformation_shape in config")
        return flat.reshape(3, *self.deformation_shape)

    def initialize(self, z0: torch.Tensor, sigma0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.tanh(self.init_encoder(z0))
        a = self.init_encoder.weight
        sigma = matrix_from_covariance(sigma0)
        p0 = a @ sigma @ a.T
        eye = torch.eye(self.hidden_dim, device=z0.device, dtype=z0.dtype)
        return h0, symmetrize(p0 + self.init_covariance * eye)

    def propagate_sde(
        self,
        h: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        calibration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.drift(h, t)
        h_pred = h + dt * f
        f_jac = jacobian_matrix(
            lambda hh: self.drift(hh, t),
            h,
            create_graph=self.covariance_grad,
            vectorize=self.jacobian_vectorize,
        ).to(h)
        g = self.diffusion_diag(h, t)
        process = torch.diag(calibration.to(h) * g.pow(2) * dt.clamp_min(1.0e-6))
        p_pred = p + dt * (p @ f_jac.T + f_jac @ p) + process
        return h_pred, symmetrize(p_pred)

    def innovation_stats(
        self,
        h_pred: torch.Tensor,
        p_pred: torch.Tensor,
        z: torch.Tensor,
        sigma_z: torch.Tensor,
        t: torch.Tensor,
        *,
        jitter: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_pred = self.decode_observation(h_pred, t)
        err = z - z_pred
        c_jac = jacobian_matrix(
            lambda hh: self.decode_observation(hh, t),
            h_pred,
            create_graph=self.covariance_grad,
            vectorize=self.jacobian_vectorize,
        ).to(h_pred)
        if self.innovation_mode == "full":
            sigma = matrix_from_covariance(sigma_z).to(h_pred)
            s = symmetrize(c_jac @ p_pred @ c_jac.T + sigma)
            eye = torch.eye(s.shape[0], device=s.device, dtype=s.dtype)
            s = s + float(jitter) * eye
            solve = torch.linalg.solve(s, err[:, None]).squeeze(1)
            ratio = (err @ solve) / err.numel()
            sign, logabsdet = torch.linalg.slogdet(s)
            nll = 0.5 * ((err @ solve) + logabsdet.clamp_min(-50.0)) / err.numel()
            diag = s.diag()
        else:
            sigma_diag = covariance_diag(sigma_z).to(h_pred)
            s_diag = (c_jac @ p_pred @ c_jac.T).diag() + sigma_diag
            s_diag = s_diag.clamp_min(float(jitter))
            ratio = (err.pow(2) / s_diag).mean()
            nll = gaussian_nll_from_diag(err, s_diag, jitter=jitter)
            diag = s_diag
        return z_pred, err, nll, ratio.detach()

    def cvgru_update(
        self,
        h_pred: torch.Tensor,
        p_pred: torch.Tensor,
        z: torch.Tensor,
        sigma_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_updated = self.gru_cell(z[None], h_pred[None]).squeeze(0)
        a = jacobian_matrix(
            lambda hh: self.gru_cell(z[None], hh[None]).squeeze(0),
            h_pred,
            create_graph=self.covariance_grad,
            vectorize=self.jacobian_vectorize,
        ).to(h_pred)
        b = jacobian_matrix(
            lambda zz: self.gru_cell(zz[None], h_pred[None]).squeeze(0),
            z,
            create_graph=self.covariance_grad,
            vectorize=self.jacobian_vectorize,
        ).to(h_pred)
        sigma = matrix_from_covariance(sigma_z).to(h_pred)
        p_updated = a @ p_pred @ a.T + b @ sigma @ b.T
        return h_updated, symmetrize(p_updated)

    def deformation_output(
        self,
        h: torch.Tensor,
        p: torch.Tensor,
        t: torch.Tensor,
        *,
        covariance_mode: Literal["none", "diag", "block"] | None = None,
    ) -> DeformationPrediction:
        mode = covariance_mode or self.deformation_covariance
        displacement = self.decode_deformation(h, t)
        if mode == "none":
            return DeformationPrediction(displacement=displacement, covariance_diag=None, covariance_blocks=None)

        rows: list[torch.Tensor] = []
        h_req = h.detach().requires_grad_(True)
        flat_req = self.decode_deformation_flat(h_req, t)
        for start in range(0, flat_req.numel(), self.deformation_jacobian_chunk):
            end = min(start + self.deformation_jacobian_chunk, flat_req.numel())
            eye = torch.zeros((end - start, flat_req.numel()), device=h.device, dtype=h.dtype)
            eye[:, start:end] = torch.eye(end - start, device=h.device, dtype=h.dtype)
            grad = torch.autograd.grad(
                flat_req,
                h_req,
                grad_outputs=eye,
                retain_graph=end < flat_req.numel(),
                is_grads_batched=True,
            )[0]
            rows.append(grad.detach())
        c_jac = torch.cat(rows, dim=0).to(p)
        covariance_diag = None
        covariance_blocks = None
        if mode == "diag":
            if self.deformation_shape is None:
                raise RuntimeError("deformation_shape is required for diagonal covariance")
            covariance_diag = ((c_jac @ p) * c_jac).sum(dim=1).reshape(3, *self.deformation_shape)
        elif mode == "block":
            if self.deformation_shape is None:
                raise RuntimeError("deformation_shape is required for block covariance")
            d, h_size, w_size = self.deformation_shape
            c_voxel = c_jac.reshape(3, d * h_size * w_size, self.hidden_dim).permute(1, 0, 2)
            blocks = torch.matmul(torch.matmul(c_voxel, p), c_voxel.transpose(1, 2))
            covariance_blocks = blocks.reshape(d, h_size, w_size, 3, 3)
        else:
            raise ValueError(f"unsupported deformation covariance mode: {mode}")
        return DeformationPrediction(
            displacement=displacement,
            covariance_diag=covariance_diag,
            covariance_blocks=covariance_blocks,
        )

    def forward_sequence(
        self,
        z: torch.Tensor,
        covariance: torch.Tensor,
        times: torch.Tensor,
        *,
        jitter: float,
        loss_name: Literal["nll", "mse"] = "nll",
    ) -> DraftSDERNNOutput:
        if z.ndim != 2:
            raise ValueError(f"expected z [T,d], got {tuple(z.shape)}")
        if z.shape[0] < 2:
            zero = z.new_tensor(0.0)
            return DraftSDERNNOutput(zero, zero, zero, zero, z.new_tensor(1.0))

        h, p = self.initialize(z[0], covariance[0])
        calibration = z.new_tensor(1.0)
        losses: list[torch.Tensor] = []
        mses: list[torch.Tensor] = []
        nlls: list[torch.Tensor] = []
        ratios: list[torch.Tensor] = []

        for idx in range(1, z.shape[0]):
            t0 = times[idx - 1]
            t1 = times[idx]
            dt = (t1 - t0).clamp_min(1.0e-6)
            h_pred, p_pred = self.propagate_sde(h, p, t0, dt, calibration)
            z_pred, err, nll, ratio = self.innovation_stats(
                h_pred,
                p_pred,
                z[idx],
                covariance[idx],
                t1,
                jitter=jitter,
            )
            mse = err.pow(2).mean()
            loss = nll if loss_name == "nll" else mse
            losses.append(loss)
            mses.append(mse)
            nlls.append(nll)
            ratios.append(ratio)

            calibration = (1.0 - self.calibration_lambda) * calibration + self.calibration_lambda * ratio
            h, p = self.cvgru_update(h_pred, p_pred, z[idx], covariance[idx])

        return DraftSDERNNOutput(
            loss=torch.stack(losses).mean(),
            mse=torch.stack(mses).mean(),
            nll=torch.stack(nlls).mean(),
            mean_innovation_ratio=torch.stack(ratios).mean(),
            final_calibration=calibration.detach(),
        )

    def infer_sequence_states(
        self,
        z: torch.Tensor,
        covariance: torch.Tensor,
        times: torch.Tensor,
        *,
        jitter: float,
    ) -> list[dict[str, torch.Tensor]]:
        if z.shape[0] < 1:
            return []
        h, p = self.initialize(z[0], covariance[0])
        calibration = z.new_tensor(1.0)
        states: list[dict[str, torch.Tensor]] = [
            {"h": h.detach(), "P": p.detach(), "time": times[0].detach(), "calibration": calibration.detach()}
        ]
        for idx in range(1, z.shape[0]):
            t0 = times[idx - 1]
            t1 = times[idx]
            dt = (t1 - t0).clamp_min(1.0e-6)
            h_pred, p_pred = self.propagate_sde(h, p, t0, dt, calibration)
            _, _, _, ratio = self.innovation_stats(
                h_pred,
                p_pred,
                z[idx],
                covariance[idx],
                t1,
                jitter=jitter,
            )
            calibration = (1.0 - self.calibration_lambda) * calibration + self.calibration_lambda * ratio
            h, p = self.cvgru_update(h_pred, p_pred, z[idx], covariance[idx])
            states.append({"h": h.detach(), "P": p.detach(), "time": t1.detach(), "calibration": calibration.detach()})
        return states

    def rollout_sequence(
        self,
        z: torch.Tensor,
        covariance: torch.Tensor,
        times: torch.Tensor,
        *,
        jitter: float,
    ) -> list[dict[str, torch.Tensor]]:
        """Differentiable SDE/CVGRU rollout used for NODEO-style deformation training."""

        if z.shape[0] < 1:
            return []
        h, p = self.initialize(z[0], covariance[0])
        calibration = z.new_tensor(1.0)
        states: list[dict[str, torch.Tensor]] = [{"h": h, "P": p, "time": times[0], "calibration": calibration}]
        for idx in range(1, z.shape[0]):
            t0 = times[idx - 1]
            t1 = times[idx]
            dt = (t1 - t0).clamp_min(1.0e-6)
            h_pred, p_pred = self.propagate_sde(h, p, t0, dt, calibration)
            _, _, _, ratio = self.innovation_stats(
                h_pred,
                p_pred,
                z[idx],
                covariance[idx],
                t1,
                jitter=jitter,
            )
            calibration = (1.0 - self.calibration_lambda) * calibration + self.calibration_lambda * ratio
            h, p = self.cvgru_update(h_pred, p_pred, z[idx], covariance[idx])
            states.append({"h": h, "P": p, "time": t1, "calibration": calibration})
        return states
