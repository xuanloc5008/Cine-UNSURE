"""Score-based C-UNSURE observation covariance for foundation encoders."""

from score_cunsure.cunsure import (
    CUNSUREConfig,
    LatentCovarianceResult,
    eta_from_score_autocorrelation,
    estimate_latent_covariance,
    sample_correlated_noise_like,
    score_autocorrelation,
    sqrt_spectrum_from_eta,
)
from score_cunsure.encoders import build_encoder
from score_cunsure.score_model import ScoreUNet, ardae_score_loss
from score_cunsure.verification import (
    covariance_cosine_similarity,
    monte_carlo_encoder_covariance,
    relative_trace_error,
    run_sensitivity_trace_test,
    run_synthetic_alignment,
)

__all__ = [
    "CUNSUREConfig",
    "LatentCovarianceResult",
    "ScoreUNet",
    "ardae_score_loss",
    "build_encoder",
    "eta_from_score_autocorrelation",
    "estimate_latent_covariance",
    "sample_correlated_noise_like",
    "score_autocorrelation",
    "sqrt_spectrum_from_eta",
    "covariance_cosine_similarity",
    "monte_carlo_encoder_covariance",
    "relative_trace_error",
    "run_sensitivity_trace_test",
    "run_synthetic_alignment",
]
