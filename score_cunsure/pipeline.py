"""High-level score C-UNSURE observation model wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from score_cunsure.cunsure import CUNSUREConfig, LatentCovarianceResult, estimate_latent_covariance


@dataclass(frozen=True)
class ObservationModelConfig:
    cunsure: CUNSUREConfig = CUNSUREConfig()
    device: str = "cpu"


class ScoreCUNSUREObservationModel(nn.Module):
    """Frozen encoder plus score-based observation covariance estimator."""

    def __init__(
        self,
        score_model: nn.Module,
        encoder: nn.Module,
        config: ObservationModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.score_model = score_model
        self.encoder = encoder
        self.config = config or ObservationModelConfig()

    def estimate(self, image: torch.Tensor, *, generator: torch.Generator | None = None) -> LatentCovarianceResult:
        image = image.to(self.config.device)
        self.score_model.to(self.config.device).eval()
        self.encoder.to(self.config.device).eval()
        return estimate_latent_covariance(
            image=image,
            score_model=self.score_model,
            encoder=self.encoder,
            config=self.config.cunsure,
            generator=generator,
        )

