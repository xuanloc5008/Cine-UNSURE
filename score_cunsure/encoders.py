"""Foundation encoder adapters for CineMA and MedSAM2."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _prepend_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


class IdentityPoolEncoder(nn.Module):
    """Tiny deterministic encoder for smoke tests and pipeline debugging."""

    def __init__(self, out_dim: int = 16) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4:
            pooled = F.adaptive_avg_pool2d(image, output_size=(4, 4)).flatten(start_dim=1)
        elif image.ndim == 5:
            pooled = F.adaptive_avg_pool3d(image, output_size=(4, 4, 4)).flatten(start_dim=1)
        else:
            raise ValueError(f"expected [B,C,H,W] or [B,C,H,W,D], got {tuple(image.shape)}")
        if pooled.shape[1] >= self.out_dim:
            return pooled[:, : self.out_dim]
        pad = torch.zeros((pooled.shape[0], self.out_dim - pooled.shape[1]), device=pooled.device, dtype=pooled.dtype)
        return torch.cat([pooled, pad], dim=1)


class CineMAEncoderAdapter(nn.Module):
    """Adapter over mathpluscode/CineMA feature extraction.

    For the pretrained MAE model, CineMA expects a dict keyed by view. SAX is
    3D `[B, 1, 192, 192, 16]`; LAX views are 2D `[B, 1, 256, 256]`.
    """

    def __init__(
        self,
        external_root: str | Path,
        *,
        view: str = "lax_4c",
        pool: str = "cls",
        device: str | torch.device = "cpu",
        cache_dir: str | None = None,
        auto_preprocess: bool = True,
        sax_depth: int = 16,
    ) -> None:
        super().__init__()
        self.view = view
        self.pool = pool
        self.auto_preprocess = auto_preprocess
        self.sax_depth = sax_depth
        repo = Path(external_root) / "CineMA"
        _prepend_path(repo)
        from cinema import CineMA  # type: ignore

        kwargs: dict[str, Any] = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        self.model = freeze_module(CineMA.from_pretrained(**kwargs)).to(device)

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        if not self.auto_preprocess:
            return image
        if image.ndim not in {4, 5}:
            return image
        if self.view == "sax":
            if image.shape[1] != 1:
                image = image[:, :1]
            if image.ndim == 5:
                return F.interpolate(
                    image,
                    size=(192, 192, self.sax_depth),
                    mode="trilinear",
                    align_corners=False,
                )
            image = F.interpolate(image, size=(192, 192), mode="bilinear", align_corners=False)
            return image.unsqueeze(-1).repeat(1, 1, 1, 1, self.sax_depth)
        if image.ndim == 5:
            # LAX models are 2D. Use the central SAX slice as a pragmatic
            # fallback when a volume is passed to a LAX view.
            image = image[..., image.shape[-1] // 2]
        if image.shape[1] != 1:
            image = image[:, :1]
        image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
        return image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self._prepare(image)
        image_dict = {self.view: image}
        features = self.model.feature_forward(image_dict)
        if self.pool == "cls":
            return features["cls"][:, 0]
        if self.pool in {"patch", "mean_patch"}:
            return features[self.view].mean(dim=1)
        if self.pool == "all":
            return torch.cat([features["cls"], features[self.view]], dim=1).mean(dim=1)
        if self.pool == "flatten_patch":
            return features[self.view].flatten(start_dim=1)
        raise ValueError(f"unsupported CineMA pool mode: {self.pool}")


class MedSAM2EncoderAdapter(nn.Module):
    """Adapter over bowang-lab/MedSAM2 image backbone features.

    This adapter expects model-ready tensors `[B, 3, H, W]`, usually resized to
    the config resolution, e.g. 512x512. It pools the FPN/vision feature maps to
    a single latent vector so the SDE observation model can compare it with
    CineMA under the same covariance push-forward code.
    """

    def __init__(
        self,
        external_root: str | Path,
        *,
        config_name: str = "configs/sam2.1_hiera_t512.yaml",
        checkpoint: str | Path | None = None,
        pool: str = "vision_mean",
        device: str | torch.device = "cpu",
        auto_preprocess: bool = True,
        image_size: int | None = None,
        imagenet_normalize: bool = True,
        volume_pool: str = "mean",
    ) -> None:
        super().__init__()
        self.pool = pool
        self.auto_preprocess = auto_preprocess
        self.imagenet_normalize = imagenet_normalize
        self.volume_pool = volume_pool
        repo = Path(external_root) / "MedSAM2"
        _prepend_path(repo)

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        config_dir = repo / "sam2"
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
            cfg = compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[MedSAM2EncoderAdapter] missing checkpoint keys: {len(missing)}")
            if unexpected:
                print(f"[MedSAM2EncoderAdapter] unexpected checkpoint keys: {len(unexpected)}")
        self.model = freeze_module(model).to(device)
        self.image_size = int(image_size or getattr(self.model, "image_size", 512))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _prepare_2d(self, image: torch.Tensor) -> torch.Tensor:
        if not self.auto_preprocess:
            return image
        if image.ndim != 4:
            raise ValueError(f"MedSAM2 adapter expects [B, C, H, W], got {tuple(image.shape)}")
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] > 3:
            image = image[:, :3]
        elif image.shape[1] == 2:
            image = torch.cat([image, image[:, :1]], dim=1)
        image = torch.clamp(image, 0.0, 1.0)
        image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if self.imagenet_normalize:
            image = (image - self.mean.to(image)) / self.std.to(image)
        return image

    def _forward_2d(self, image: torch.Tensor) -> torch.Tensor:
        image = self._prepare_2d(image)
        out = self.model.forward_image(image)
        if self.pool == "vision_mean":
            return out["vision_features"].mean(dim=(-2, -1))
        if self.pool == "fpn_last_mean":
            return out["backbone_fpn"][-1].mean(dim=(-2, -1))
        if self.pool == "fpn_all_mean":
            return torch.cat([feat.mean(dim=(-2, -1)) for feat in out["backbone_fpn"]], dim=1)
        if self.pool == "flatten_vision":
            return out["vision_features"].flatten(start_dim=1)
        raise ValueError(f"unsupported MedSAM2 pool mode: {self.pool}")

    def _forward_volume(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(f"MedSAM2 volume path expects [B,C,H,W,D], got {tuple(image.shape)}")
        batch, channels, height, width, depth = image.shape
        slices = image.permute(0, 4, 1, 2, 3).reshape(batch * depth, channels, height, width)
        slice_features = self._forward_2d(slices).reshape(batch, depth, -1)
        if self.volume_pool == "mean":
            return slice_features.mean(dim=1)
        if self.volume_pool == "max":
            return slice_features.max(dim=1).values
        if self.volume_pool == "mean_std":
            return torch.cat([slice_features.mean(dim=1), slice_features.std(dim=1, unbiased=False)], dim=1)
        if self.volume_pool == "flatten":
            return slice_features.flatten(start_dim=1)
        raise ValueError(f"unsupported MedSAM2 volume_pool mode: {self.volume_pool}")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 5:
            return self._forward_volume(image)
        return self._forward_2d(image)


def build_encoder(name: str, **kwargs: Any) -> nn.Module:
    """Build an encoder adapter by name."""
    name = name.lower()
    if name in {"identity", "debug"}:
        return IdentityPoolEncoder(out_dim=int(kwargs.pop("out_dim", 16)))
    if name == "cinema":
        return CineMAEncoderAdapter(**kwargs)
    if name in {"medsam2", "medsam"}:
        return MedSAM2EncoderAdapter(**kwargs)
    raise ValueError(f"unknown encoder name: {name}")


@contextlib.contextmanager
def inference_mode(module: nn.Module):
    was_training = module.training
    module.eval()
    with torch.no_grad():
        yield module
    module.train(was_training)
