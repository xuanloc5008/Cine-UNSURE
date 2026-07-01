#!/usr/bin/env python3
"""Run verification tests for score C-UNSURE covariance inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_cunsure.checkpoint import load_score_checkpoint
from score_cunsure.cunsure import CUNSUREConfig
from score_cunsure.data import load_frame
from score_cunsure.encoders import build_encoder
from score_cunsure.verification import run_sensitivity_trace_test, run_synthetic_alignment


def default_external_root() -> Path:
    return Path(__file__).resolve().parents[3] / "work" / "external"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Clean or pseudo-clean input frame/4D cine file")
    parser.add_argument("--score-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["synthetic", "sensitivity", "all"], default="all")
    parser.add_argument("--encoder", choices=["identity", "cinema", "medsam2"], default="identity")
    parser.add_argument("--external-root", default=str(default_external_root()))

    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--spatial-dims", type=int, choices=[2, 3], default=3)
    parser.add_argument("--npz-key", default=None)
    parser.add_argument("--time-index", type=int, default=None)
    parser.add_argument("--time-axis", type=int, default=-1)
    parser.add_argument("--frame-layout", default="hwd")
    parser.add_argument("--clamp-input", action="store_true", help="Clamp synthetic noisy images to [0, 1]")

    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--n-probes", type=int, default=32)
    parser.add_argument("--tau", type=float, default=1.0e-2)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--spectral-floor", type=float, default=1.0e-8)

    parser.add_argument("--synthetic-sigma", type=float, default=0.1)
    parser.add_argument("--cosine-threshold", type=float, default=0.95)
    parser.add_argument("--mc-samples", type=int, default=128)
    parser.add_argument("--mc-batch-size", type=int, default=8)

    parser.add_argument("--sensitivity-levels", type=float, nargs="+", default=[0.0, 0.05, 0.20])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--independent-sensitivity-noise", action="store_true")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--identity-dim", type=int, default=16)
    parser.add_argument("--cinema-view", default="sax")
    parser.add_argument("--cinema-pool", default="cls")
    parser.add_argument("--cinema-cache-dir", default=None)
    parser.add_argument("--medsam2-config", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--medsam2-checkpoint", default=None)
    parser.add_argument("--medsam2-pool", default="vision_mean")
    parser.add_argument("--medsam2-volume-pool", default="mean", choices=["mean", "max", "mean_std", "flatten"])
    return parser.parse_args()


def build_requested_encoder(args: argparse.Namespace) -> torch.nn.Module:
    if args.encoder == "identity":
        return build_encoder("identity", out_dim=args.identity_dim)
    if args.encoder == "cinema":
        return build_encoder(
            "cinema",
            external_root=args.external_root,
            view=args.cinema_view,
            pool=args.cinema_pool,
            device=args.device,
            cache_dir=args.cinema_cache_dir,
        )
    return build_encoder(
        "medsam2",
        external_root=args.external_root,
        config_name=args.medsam2_config,
        checkpoint=args.medsam2_checkpoint,
        pool=args.medsam2_pool,
        volume_pool=args.medsam2_volume_pool,
        device=args.device,
    )


def tensor_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().flatten()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    score_model, _ = load_score_checkpoint(args.score_checkpoint, device=device)
    encoder = build_requested_encoder(args).to(device).eval()
    image = load_frame(
        args.image,
        npz_key=args.npz_key,
        channels=args.channels,
        normalize=True,
        spatial_dims=args.spatial_dims,
        frame_index=args.time_index,
        time_axis=args.time_axis if args.time_index is not None else None,
        frame_layout=args.frame_layout,
    ).unsqueeze(0).to(device)
    config = CUNSUREConfig(
        radius=args.radius,
        eps=args.eps,
        spectral_floor=args.spectral_floor,
        n_probes=args.n_probes,
        finite_difference_tau=args.tau,
    )

    summary: dict[str, object] = {
        "args": vars(args),
        "overall_passed": True,
    }

    if args.mode in {"synthetic", "all"}:
        synthetic = run_synthetic_alignment(
            image,
            score_model,
            encoder,
            config,
            sigma=args.synthetic_sigma,
            n_mc_samples=args.mc_samples,
            mc_batch_size=args.mc_batch_size,
            cosine_threshold=args.cosine_threshold,
            generator=generator,
            clamp=args.clamp_input,
        )
        synthetic_summary = {
            "sigma": args.synthetic_sigma,
            "cosine_similarity": tensor_list(synthetic.cosine_similarity),
            "cosine_threshold": args.cosine_threshold,
            "relative_trace_error": tensor_list(synthetic.relative_trace_error),
            "cunsure_trace": tensor_list(synthetic.cunsure_trace),
            "monte_carlo_trace": tensor_list(synthetic.monte_carlo_trace),
            "passed": synthetic.passed,
        }
        summary["synthetic"] = synthetic_summary
        summary["overall_passed"] = bool(summary["overall_passed"]) and synthetic.passed
        torch.save(
            {
                "cunsure_covariance": synthetic.cunsure_covariance.cpu(),
                "monte_carlo_covariance": synthetic.monte_carlo_covariance.cpu(),
                "summary": synthetic_summary,
            },
            out_dir / "synthetic_alignment.pt",
        )

    if args.mode in {"sensitivity", "all"}:
        sensitivity = run_sensitivity_trace_test(
            image,
            score_model,
            encoder,
            config,
            noise_levels=args.sensitivity_levels,
            trials=args.trials,
            generator=generator,
            clamp=args.clamp_input,
            shared_noise_direction=not args.independent_sensitivity_noise,
        )
        sensitivity_summary = {
            "noise_levels": sensitivity.noise_levels,
            "mean_traces": tensor_list(sensitivity.mean_traces),
            "std_traces": tensor_list(sensitivity.std_traces),
            "pass_rate": sensitivity.pass_rate,
            "passed": sensitivity.passed,
        }
        summary["sensitivity"] = sensitivity_summary
        summary["overall_passed"] = bool(summary["overall_passed"]) and sensitivity.passed

        csv_path = out_dir / "sensitivity_traces.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", *[f"sigma_{level:g}" for level in sensitivity.noise_levels], "strictly_increasing"])
            for trial_idx, row in enumerate(sensitivity.traces.detach().cpu()):
                writer.writerow([trial_idx, *[float(v) for v in row], bool(sensitivity.monotonic_per_trial[trial_idx])])

    summary_path = out_dir / "verification_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved verification summary to {summary_path}")


if __name__ == "__main__":
    main()

