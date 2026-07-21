from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be at least 1")
        modules: list[nn.Module] = []
        current = int(in_dim)
        for _ in range(int(layers) - 1):
            modules.extend((nn.Linear(current, int(hidden_dim)), nn.Tanh()))
            current = int(hidden_dim)
        modules.append(nn.Linear(current, int(out_dim)))
        self.net = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass
class MeanSequenceOutput:
    hidden_prior: Tensor
    hidden_mean: Tensor
    motion_code: Tensor


@dataclass
class AnalyticalSequenceOutput:
    hidden_mean: Tensor
    hidden_covariance: Tensor
    hidden_ambiguity_covariance: Tensor
    hidden_process_covariance: Tensor
    motion_code: Tensor
    motion_covariance_factor: Tensor
    ambiguity_motion_covariance_factor: Tensor
    process_motion_covariance_factor: Tensor


def fit_linear_basis(values: Tensor, rank: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return mean, orthonormal basis, and coefficients for [T,N] values."""
    if values.ndim != 2:
        raise ValueError(f"values must be [T,N], got {tuple(values.shape)}")
    mean = values.mean(dim=0)
    centered = values - mean
    maximum_rank = min(int(centered.shape[0]), int(centered.shape[1]))
    selected_rank = min(max(int(rank), 1), maximum_rank)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:selected_rank].T.contiguous()
    coefficients = centered @ basis
    return mean, basis, coefficients


def project_observation_covariance(covariance: Tensor, basis: Tensor) -> Tensor:
    """Project diagonal or full voxel covariance into a low-rank basis."""
    if covariance.ndim == 2:
        return torch.einsum("zr,tz,zs->trs", basis, covariance, basis)
    if covariance.ndim == 3:
        return torch.einsum("zr,tzu,us->trs", basis, covariance, basis)
    raise ValueError(f"covariance must be [T,Z] or [T,Z,Z], got {tuple(covariance.shape)}")


def stabilize_covariance(covariance: Tensor, *, eigenvalue_floor: float) -> Tensor:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    return (eigenvectors * eigenvalues.clamp_min(float(eigenvalue_floor))) @ eigenvectors.T


def covariance_cholesky(covariance: Tensor, *, eigenvalue_floor: float) -> tuple[Tensor, Tensor]:
    """Return a numerically PSD covariance and its Cholesky factor.

    Covariance propagation runs in the model dtype, normally float32. A matrix
    reconstructed from a float32 eigendecomposition can therefore acquire tiny
    negative eigenvalues even after eigenvalue clipping. Perform the final PSD
    projection and factorization in float64, then reconstruct the covariance
    from the returned factor so both saved quantities remain consistent.
    """
    original_device = covariance.device
    original_dtype = covariance.dtype
    work_device = torch.device("cpu") if covariance.device.type == "mps" else covariance.device
    work = covariance.to(device=work_device, dtype=torch.float64)
    work = 0.5 * (work + work.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(work)
    scale = eigenvalues.abs().max().clamp_min(1.0)
    numerical_floor = torch.finfo(work.dtype).eps * scale * 10.0
    floor = torch.maximum(
        numerical_floor,
        torch.as_tensor(float(eigenvalue_floor), device=work.device, dtype=work.dtype),
    )
    work = (eigenvectors * eigenvalues.clamp_min(floor)) @ eigenvectors.T
    work = 0.5 * (work + work.T)
    chol = torch.linalg.cholesky(work)
    chol = chol.to(device=original_device, dtype=original_dtype)
    stabilized = chol @ chol.T
    stabilized = 0.5 * (stabilized + stabilized.T)
    return stabilized, chol


class PerSequencePostHocSDERNN(nn.Module):
    """Per-sequence mean SDE-CVGRU with analytical post-hoc covariance.

    Only the mean path is optimized. Hidden and output covariance are computed
    after freezing the network, using the Jacobian recursions in Eqs. 16, 20,
    and 22 of the SDE-RNN paper. The observation stream is driven directly by
    NODEO residual ambiguity, while Q is estimated from sequence innovations.
    The two covariance streams remain separate throughout propagation.
    """

    def __init__(
        self,
        *,
        observation_dim: int,
        motion_dim: int,
        hidden_dim: int,
        mlp_hidden_dim: int,
        mlp_layers: int,
        integration_steps: int,
        periodic_time: bool = True,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.motion_dim = int(motion_dim)
        self.hidden_dim = int(hidden_dim)
        self.integration_steps = max(int(integration_steps), 1)
        self.periodic_time = bool(periodic_time)
        time_dim = 2 if self.periodic_time else 1
        self.drift = MLP(hidden_dim + time_dim, mlp_hidden_dim, hidden_dim, mlp_layers)
        self.cvgru = nn.GRUCell(observation_dim, hidden_dim)
        self.decoder = MLP(hidden_dim, mlp_hidden_dim, motion_dim, mlp_layers)
        self.h0 = nn.Parameter(torch.zeros(hidden_dim))

    def drift_value(self, hidden: Tensor, time: Tensor) -> Tensor:
        time = time.reshape(1).to(hidden)
        if self.periodic_time:
            time_features = torch.cat(
                (torch.sin(2.0 * torch.pi * time), torch.cos(2.0 * torch.pi * time))
            )
        else:
            time_features = time
        return self.drift(torch.cat((hidden, time_features)))

    def propagate_mean(self, hidden: Tensor, time0: Tensor, time1: Tensor) -> Tensor:
        total_dt = (time1 - time0).clamp_min(1.0e-6)
        dt = total_dt / float(self.integration_steps)
        for step in range(self.integration_steps):
            time = time0 + dt * float(step)
            hidden = hidden + dt * self.drift_value(hidden, time)
        return hidden

    def update_mean(self, hidden_prior: Tensor, observation: Tensor) -> Tensor:
        return self.cvgru(observation[None], hidden_prior[None]).squeeze(0)

    def forward_mean(self, *, times: Tensor, observation: Tensor, mask: Tensor) -> MeanSequenceOutput:
        if times.ndim != 1 or observation.ndim != 2 or mask.ndim != 1:
            raise ValueError("times, observation, and mask must be [T], [T,Z], and [T]")
        if not (len(times) == len(observation) == len(mask)):
            raise ValueError("times, observation, and mask must have the same length")
        hidden = self.h0.to(observation)
        priors: list[Tensor] = []
        means: list[Tensor] = []
        motion: list[Tensor] = []
        for index in range(len(times)):
            if index > 0:
                hidden = self.propagate_mean(hidden, times[index - 1], times[index])
            prior = hidden
            if bool(mask[index]):
                hidden = self.update_mean(prior, observation[index])
            priors.append(prior)
            means.append(hidden)
            motion.append(self.decoder(hidden))
        return MeanSequenceOutput(torch.stack(priors), torch.stack(means), torch.stack(motion))

    def estimate_process_covariance(
        self,
        *,
        times: Tensor,
        observation: Tensor,
        observation_covariance: Tensor,
        covariance_floor: float,
        shrinkage: float,
    ) -> Tensor:
        """Estimate dynamic Q from hidden innovations after removing observation noise."""
        full_mask = torch.ones(len(times), device=times.device, dtype=torch.bool)
        output = self.forward_mean(times=times, observation=observation, mask=full_mask)
        innovations = output.hidden_mean[1:] - output.hidden_prior[1:]
        if len(innovations) < 2:
            return torch.eye(self.hidden_dim, device=times.device, dtype=times.dtype) * float(covariance_floor)
        centered = innovations - innovations.mean(dim=0, keepdim=True)
        residual_covariances: list[Tensor] = []
        for index, innovation in enumerate(centered, start=1):
            _, jacobian_z = self._update_jacobians(output.hidden_prior[index], observation[index])
            explained = (
                jacobian_z.to(innovation)
                @ observation_covariance[index].to(innovation)
                @ jacobian_z.to(innovation).T
            )
            residual_covariances.append(torch.outer(innovation, innovation) - explained)
        covariance = torch.stack(residual_covariances).mean(dim=0)
        diagonal = torch.diag(torch.diag(covariance))
        covariance = (1.0 - float(shrinkage)) * covariance + float(shrinkage) * diagonal
        return stabilize_covariance(covariance, eigenvalue_floor=covariance_floor)

    def _drift_jacobian(self, hidden: Tensor, time: Tensor) -> Tensor:
        value = hidden.detach().requires_grad_(True)
        return torch.autograd.functional.jacobian(
            lambda item: self.drift_value(item, time.detach()),
            value,
            create_graph=False,
            strict=False,
        ).detach()

    def _update_jacobians(self, hidden: Tensor, observation: Tensor) -> tuple[Tensor, Tensor]:
        h = hidden.detach().requires_grad_(True)
        z = observation.detach().requires_grad_(True)
        jacobian_h = torch.autograd.functional.jacobian(
            lambda item: self.update_mean(item, z.detach()),
            h,
            create_graph=False,
            strict=False,
        ).detach()
        jacobian_z = torch.autograd.functional.jacobian(
            lambda item: self.update_mean(h.detach(), item),
            z,
            create_graph=False,
            strict=False,
        ).detach()
        return jacobian_h, jacobian_z

    def _decoder_jacobian(self, hidden: Tensor) -> Tensor:
        value = hidden.detach().requires_grad_(True)
        return torch.autograd.functional.jacobian(
            self.decoder,
            value,
            create_graph=False,
            strict=False,
        ).detach()

    def propagate_analytical(
        self,
        *,
        times: Tensor,
        observation: Tensor,
        observation_covariance: Tensor,
        process_covariance: Tensor,
        init_covariance: float,
        covariance_floor: float,
    ) -> AnalyticalSequenceOutput:
        """Freeze weights and propagate mean/covariance at every observed frame."""
        hidden = self.h0.detach().to(observation)
        eye = torch.eye(self.hidden_dim, device=hidden.device, dtype=hidden.dtype)
        ambiguity_covariance = torch.zeros_like(eye)
        dynamics_covariance = eye * float(init_covariance)
        process_covariance = process_covariance.to(dynamics_covariance)
        means: list[Tensor] = []
        covariances: list[Tensor] = []
        ambiguity_covariances: list[Tensor] = []
        process_covariances: list[Tensor] = []
        motion: list[Tensor] = []
        factors: list[Tensor] = []
        ambiguity_factors: list[Tensor] = []
        process_factors: list[Tensor] = []

        for index in range(len(times)):
            if index > 0:
                total_dt = (times[index] - times[index - 1]).clamp_min(1.0e-6)
                dt = total_dt / float(self.integration_steps)
                for step in range(self.integration_steps):
                    time = times[index - 1] + dt * float(step)
                    drift_jacobian = self._drift_jacobian(hidden, time).to(dynamics_covariance)
                    ambiguity_covariance = ambiguity_covariance + dt * (
                        drift_jacobian @ ambiguity_covariance
                        + ambiguity_covariance @ drift_jacobian.T
                    )
                    dynamics_covariance = dynamics_covariance + dt * (
                        drift_jacobian @ dynamics_covariance
                        + dynamics_covariance @ drift_jacobian.T
                        + process_covariance
                    )
                    ambiguity_covariance = stabilize_covariance(
                        ambiguity_covariance, eigenvalue_floor=covariance_floor
                    )
                    dynamics_covariance = stabilize_covariance(
                        dynamics_covariance, eigenvalue_floor=covariance_floor
                    )
                    with torch.no_grad():
                        hidden = hidden + dt * self.drift_value(hidden, time)

            jacobian_h, jacobian_z = self._update_jacobians(hidden, observation[index])
            with torch.no_grad():
                hidden = self.update_mean(hidden, observation[index])
            sigma = observation_covariance[index].to(ambiguity_covariance)
            update_h = jacobian_h.to(ambiguity_covariance)
            update_z = jacobian_z.to(ambiguity_covariance)
            ambiguity_covariance = (
                update_h @ ambiguity_covariance @ update_h.T
                + update_z @ sigma @ update_z.T
            )
            dynamics_covariance = update_h @ dynamics_covariance @ update_h.T
            ambiguity_covariance = stabilize_covariance(
                ambiguity_covariance, eigenvalue_floor=covariance_floor
            )
            dynamics_covariance = stabilize_covariance(
                dynamics_covariance, eigenvalue_floor=covariance_floor
            )
            covariance = stabilize_covariance(
                ambiguity_covariance + dynamics_covariance,
                eigenvalue_floor=covariance_floor,
            )

            decoder_jacobian = self._decoder_jacobian(hidden).to(covariance)
            covariance, chol = covariance_cholesky(
                covariance,
                eigenvalue_floor=covariance_floor,
            )
            ambiguity_covariance, ambiguity_chol = covariance_cholesky(
                ambiguity_covariance,
                eigenvalue_floor=covariance_floor,
            )
            dynamics_covariance, process_chol = covariance_cholesky(
                dynamics_covariance,
                eigenvalue_floor=covariance_floor,
            )
            means.append(hidden)
            covariances.append(covariance)
            ambiguity_covariances.append(ambiguity_covariance)
            process_covariances.append(dynamics_covariance)
            with torch.no_grad():
                motion.append(self.decoder(hidden))
            factors.append(decoder_jacobian @ chol)
            ambiguity_factors.append(decoder_jacobian @ ambiguity_chol)
            process_factors.append(decoder_jacobian @ process_chol)

        return AnalyticalSequenceOutput(
            hidden_mean=torch.stack(means),
            hidden_covariance=torch.stack(covariances),
            hidden_ambiguity_covariance=torch.stack(ambiguity_covariances),
            hidden_process_covariance=torch.stack(process_covariances),
            motion_code=torch.stack(motion),
            motion_covariance_factor=torch.stack(factors),
            ambiguity_motion_covariance_factor=torch.stack(ambiguity_factors),
            process_motion_covariance_factor=torch.stack(process_factors),
        )
