from __future__ import annotations

import torch

from score_cunsure.cunsure import (
    CUNSUREConfig,
    estimate_latent_covariance,
    eta_from_score_autocorrelation,
    sample_correlated_noise_like,
    score_autocorrelation,
    sqrt_spectrum_from_eta,
)
from score_cunsure.encoders import IdentityPoolEncoder
from score_cunsure.verification import (
    covariance_cosine_similarity,
    run_sensitivity_trace_test,
    run_synthetic_alignment,
)


class LinearScore(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -x


def test_score_autocorrelation_shape_and_center() -> None:
    score = torch.randn(2, 1, 16, 16)
    h = score_autocorrelation(score, radius=2)
    assert h.shape == (2, 5, 5)
    center = h[:, 2, 2]
    energy = score.pow(2).flatten(start_dim=1).mean(dim=1)
    torch.testing.assert_close(center, energy)


def test_eta_and_sampling_shapes() -> None:
    score = torch.randn(3, 1, 16, 16)
    h = score_autocorrelation(score, radius=1)
    eta = eta_from_score_autocorrelation(h)
    sqrt_spec = sqrt_spectrum_from_eta(eta, spatial_shape=(16, 16))
    noise = sample_correlated_noise_like(score, sqrt_spec)
    assert eta.shape == (3, 3, 3)
    assert sqrt_spec.shape == (3, 16, 16)
    assert noise.shape == score.shape
    assert torch.isfinite(noise).all()


def test_latent_covariance_is_symmetric() -> None:
    image = torch.rand(1, 1, 16, 16)
    score = LinearScore()
    encoder = IdentityPoolEncoder(out_dim=8)
    result = estimate_latent_covariance(
        image,
        score,
        encoder,
        CUNSUREConfig(radius=1, n_probes=5, finite_difference_tau=1.0e-2),
        generator=torch.Generator().manual_seed(7),
    )
    assert result.z.shape == (1, 8)
    assert result.covariance.shape == (1, 8, 8)
    torch.testing.assert_close(result.covariance, result.covariance.transpose(-1, -2))
    assert torch.isfinite(result.covariance).all()


def test_3d_latent_covariance_is_symmetric() -> None:
    image = torch.rand(1, 1, 12, 10, 8)
    score = LinearScore()
    encoder = IdentityPoolEncoder(out_dim=10)
    result = estimate_latent_covariance(
        image,
        score,
        encoder,
        CUNSUREConfig(radius=1, n_probes=5, finite_difference_tau=1.0e-2),
        generator=torch.Generator().manual_seed(11),
    )
    assert result.autocorrelation.shape == (1, 3, 3, 3)
    assert result.eta_hat.shape == (1, 3, 3, 3)
    assert result.sqrt_spectrum.shape == (1, 12, 10, 8)
    assert result.z.shape == (1, 10)
    assert result.covariance.shape == (1, 10, 10)
    torch.testing.assert_close(result.covariance, result.covariance.transpose(-1, -2))
    assert torch.isfinite(result.covariance).all()


def test_covariance_cosine_similarity_identity() -> None:
    cov = torch.eye(4).unsqueeze(0)
    sim = covariance_cosine_similarity(cov, cov)
    torch.testing.assert_close(sim, torch.ones(1))


def test_verification_outputs_shapes() -> None:
    image = torch.rand(1, 1, 10, 8, 6)
    score = LinearScore()
    encoder = IdentityPoolEncoder(out_dim=6)
    config = CUNSUREConfig(radius=1, n_probes=3, finite_difference_tau=1.0e-2)
    synthetic = run_synthetic_alignment(
        image,
        score,
        encoder,
        config,
        sigma=0.1,
        n_mc_samples=4,
        mc_batch_size=2,
        cosine_threshold=-1.0,
        generator=torch.Generator().manual_seed(13),
    )
    assert synthetic.cunsure_covariance.shape == (1, 6, 6)
    assert synthetic.monte_carlo_covariance.shape == (1, 6, 6)
    assert synthetic.cosine_similarity.shape == (1,)

    sensitivity = run_sensitivity_trace_test(
        image,
        score,
        encoder,
        config,
        noise_levels=(0.0, 0.05, 0.2),
        trials=2,
        generator=torch.Generator().manual_seed(17),
    )
    assert sensitivity.traces.shape == (2, 3)
    assert sensitivity.mean_traces.shape == (3,)
