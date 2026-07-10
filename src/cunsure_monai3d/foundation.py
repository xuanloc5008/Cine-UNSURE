from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .losses import circular_conv3d_depthwise


def prepend_repo(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


class CineMAFoundation(nn.Module):
    def __init__(
        self,
        *,
        repo_path: Path,
        view: str,
        pool: str,
        device: torch.device,
        cache_dir: Path | None,
    ) -> None:
        super().__init__()
        prepend_repo(repo_path)
        from cinema import CineMA  # type: ignore

        kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        self.model = freeze(CineMA.from_pretrained(**kwargs)).to(device)
        self.view = view
        self.pool = pool

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if self.view == "sax":
            if x.shape[1] != 1:
                x = x[:, :1]
            if x.ndim == 5:
                x = x.permute(0, 1, 3, 4, 2)  # [B,C,D,H,W] -> [B,C,H,W,D]
            return F.interpolate(x, size=(192, 192, 16), mode="trilinear", align_corners=False)
        x2d = x[..., x.shape[-3] // 2, :, :] if x.ndim == 5 else x
        if x2d.shape[1] != 1:
            x2d = x2d[:, :1]
        return F.interpolate(x2d, size=(256, 256), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare(x)
        features = self.model.feature_forward({self.view: x})
        if self.pool == "cls":
            return features["cls"][:, 0]
        if self.pool == "mean_patch":
            return features[self.view].mean(dim=1)
        if self.pool == "flatten_patch":
            return features[self.view].flatten(start_dim=1)
        raise ValueError(f"unsupported CineMA pool: {self.pool}")


class MedSAM2Foundation(nn.Module):
    def __init__(
        self,
        *,
        repo_path: Path,
        config_name: str,
        checkpoint: Path,
        pool: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        prepend_repo(repo_path)
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        config_dir = repo_path / "sam2"
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state_dict, strict=False)
        self.model = freeze(model).to(device)
        self.pool = pool
        image_size = int(getattr(self.model, "image_size", 512))
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _prepare_slices(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"MedSAM2 expects [B,C,D,H,W], got {tuple(x.shape)}")
        b, c, d, h, w = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        if slices.shape[1] == 1:
            slices = slices.repeat(1, 3, 1, 1)
        elif slices.shape[1] == 2:
            slices = torch.cat([slices, slices[:, :1]], dim=1)
        elif slices.shape[1] > 3:
            slices = slices[:, :3]
        slices = torch.clamp(slices, 0.0, 1.0)
        slices = F.interpolate(slices, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return (slices - self.mean.to(slices)) / self.std.to(slices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, _, _ = x.shape
        slices = self._prepare_slices(x)
        out = self.model.forward_image(slices)
        if self.pool == "vision_mean":
            z = out["vision_features"].mean(dim=(-2, -1))
        elif self.pool == "fpn_last_mean":
            z = out["backbone_fpn"][-1].mean(dim=(-2, -1))
        elif self.pool == "fpn_all_mean":
            z = torch.cat([feat.mean(dim=(-2, -1)) for feat in out["backbone_fpn"]], dim=1)
        else:
            raise ValueError(f"unsupported MedSAM2 pool: {self.pool}")
        return z.reshape(b, d, -1).mean(dim=1)


def build_foundation(config: dict, *, device: torch.device) -> nn.Module:
    name = str(config["name"]).lower()
    repo_path = Path(config["repo_path"])
    if name == "cinema":
        cache_dir = config.get("cache_dir")
        return CineMAFoundation(
            repo_path=repo_path,
            view=str(config["view"]),
            pool=str(config["pool"]),
            device=device,
            cache_dir=Path(cache_dir) if cache_dir else None,
        )
    if name == "medsam2":
        return MedSAM2Foundation(
            repo_path=repo_path,
            config_name=str(config["config_name"]),
            checkpoint=Path(config["checkpoint"]),
            pool=str(config["pool"]),
            device=device,
        )
    raise ValueError(f"unsupported foundation model: {name}")


def full_jacobian_rows(
    encoder: nn.Module,
    x: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.detach().requires_grad_(True)
    z = encoder(x)
    if z.ndim != 2 or z.shape[0] != 1:
        raise ValueError(f"encoder must return [1,d], got {tuple(z.shape)}")
    z_flat = z.reshape(-1)
    rows: list[torch.Tensor] = []
    for start in range(0, z_flat.numel(), chunk_size):
        end = min(start + chunk_size, z_flat.numel())
        eye = torch.zeros((end - start, z_flat.numel()), device=x.device, dtype=x.dtype)
        eye[:, start:end] = torch.eye(end - start, device=x.device, dtype=x.dtype)
        grad = torch.autograd.grad(
            z_flat,
            x,
            grad_outputs=eye,
            retain_graph=end < z_flat.numel(),
            is_grads_batched=True,
        )[0]
        rows.append(grad.detach().reshape(end - start, -1).cpu())
    return z.detach().cpu(), torch.cat(rows, dim=0)


def latent_covariance_from_full_jacobian(
    jacobian: torch.Tensor,
    *,
    input_shape: tuple[int, int, int, int],
    eta: torch.Tensor,
    device: torch.device,
    channel_chunk: int = 8,
) -> torch.Tensor:
    c, d, h, w = input_shape
    j = jacobian.to(device)
    eta = eta.to(device)
    sigma_j_rows: list[torch.Tensor] = []
    for start in range(0, j.shape[0], channel_chunk):
        block = j[start : start + channel_chunk].reshape(-1, c, d, h, w)
        sigma_block = circular_conv3d_depthwise(block, eta).reshape(block.shape[0], -1)
        sigma_j_rows.append(sigma_block)
    sigma_j = torch.cat(sigma_j_rows, dim=0)
    return (j @ sigma_j.T).detach().cpu()


@torch.no_grad()
def finite_difference_jvp(
    encoder: nn.Module,
    x: torch.Tensor,
    z0: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilon: float,
    normalize_direction: bool,
) -> torch.Tensor:
    if normalize_direction:
        norm = direction.flatten(start_dim=1).norm(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1.0e-12)
        step = direction / norm
        scale = norm.flatten()[0]
    else:
        step = direction
        scale = direction.new_tensor(1.0)
    z_step = encoder(x + float(epsilon) * step)
    return ((z_step - z0) / float(epsilon)).reshape(-1).detach().cpu() * scale.detach().cpu()


@torch.no_grad()
def latent_covariance_mc_finite_difference(
    encoder: nn.Module,
    x: torch.Tensor,
    *,
    eta: torch.Tensor,
    device: torch.device,
    num_probes: int,
    fd_epsilon: float,
    normalize_directions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Estimate J Sigma J^T without materializing J.

    For b ~ N(0,I), E[(J Sigma b)(J b)^T] = J Sigma J^T.
    The two JVPs are estimated by finite differences, matching the Monte-Carlo
    finite-difference style used by UNSURE/C-UNSURE.
    """

    z0 = encoder(x)
    latent_dim = int(z0.numel())
    cov = torch.zeros((latent_dim, latent_dim), dtype=torch.float32)
    eta = eta.to(device)
    for _ in range(int(num_probes)):
        probe = torch.randn_like(x)
        sigma_probe = circular_conv3d_depthwise(probe, eta)
        j_probe = finite_difference_jvp(
            encoder,
            x,
            z0,
            probe,
            epsilon=fd_epsilon,
            normalize_direction=normalize_directions,
        )
        j_sigma_probe = finite_difference_jvp(
            encoder,
            x,
            z0,
            sigma_probe,
            epsilon=fd_epsilon,
            normalize_direction=normalize_directions,
        )
        cov.add_(torch.outer(j_sigma_probe, j_probe))
    cov.div_(max(int(num_probes), 1))
    return z0.detach().cpu(), symmetrize_covariance(cov)


def symmetrize_covariance(covariance: torch.Tensor) -> torch.Tensor:
    return 0.5 * (covariance + covariance.T)


def project_covariance_psd(covariance: torch.Tensor, *, eigenvalue_floor: float = 0.0) -> torch.Tensor:
    covariance = symmetrize_covariance(covariance)
    evals, evecs = torch.linalg.eigh(covariance)
    evals = evals.clamp_min(float(eigenvalue_floor))
    return (evecs * evals) @ evecs.T


def covariance_sanity_metrics(covariance: torch.Tensor) -> dict[str, float]:
    covariance = covariance.detach().cpu()
    evals = torch.linalg.eigvalsh(symmetrize_covariance(covariance))
    diag = covariance.diag()
    return {
        "symmetric_error": float((covariance - covariance.T).abs().max()),
        "diag_min": float(diag.min()),
        "diag_max": float(diag.max()),
        "trace": float(torch.trace(covariance)),
        "eig_min": float(evals.min()),
        "eig_max": float(evals.max()),
    }
