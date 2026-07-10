#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import project_root, resolve_path, select_device
from cunsure_monai3d.draft_sde_rnn import DraftNeuralSDERNN
from cunsure_monai3d.sde_data import LatentObservationSequenceDataset


def build_model_from_checkpoint(ckpt: dict, *, latent_dim: int, device: torch.device) -> DraftNeuralSDERNN:
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    model = DraftNeuralSDERNN(
        latent_dim=latent_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        mlp_hidden_dim=int(model_cfg["mlp_hidden_dim"]),
        mlp_layers=int(model_cfg["mlp_layers"]),
        process_noise_floor=float(model_cfg["process_noise_floor"]),
        init_covariance=float(model_cfg["init_covariance"]),
        calibration_lambda=float(model_cfg["calibration_lambda"]),
        covariance_grad=bool(model_cfg.get("covariance_grad", False)),
        jacobian_vectorize=bool(model_cfg.get("jacobian_vectorize", False)),
        innovation_mode=str(model_cfg.get("innovation_mode", "diag")),
        deformation_shape=None
        if model_cfg.get("deformation_shape") is None
        else tuple(int(v) for v in model_cfg["deformation_shape"]),
        deformation_covariance=str(model_cfg.get("deformation_covariance", "none")),
        deformation_jacobian_chunk=int(model_cfg.get("deformation_jacobian_chunk", 512)),
        deformation_scale=float(model_cfg.get("deformation_scale", 4.0)),
        zero_init_deformation=bool(model_cfg.get("zero_init_deformation", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--covariance", default="diag", choices=["diag", "full"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--jitter", type=float, default=1.0e-6)
    args = parser.parse_args()

    root = project_root()
    device = select_device(args.device)
    dataset = LatentObservationSequenceDataset(
        resolve_path(args.h5, root),
        min_length=2,
        covariance=args.covariance,
        normalize_time=True,
    )
    sample = dataset[int(args.sequence_index)]
    latent_dim = int(sample["z"].shape[-1])
    ckpt = torch.load(resolve_path(args.checkpoint, root), map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(ckpt, latent_dim=latent_dim, device=device)

    z = sample["z"].to(device)
    r = sample["R"].to(device)
    times = sample["times"].to(device)
    states = model.infer_sequence_states(z, r, times, jitter=float(args.jitter))

    displacements: list[torch.Tensor] = []
    covariance_diag: list[torch.Tensor] = []
    covariance_blocks: list[torch.Tensor] = []
    calibrations: list[float] = []
    for state in states:
        pred = model.deformation_output(state["h"], state["P"], state["time"])
        displacements.append(pred.displacement.detach().cpu())
        if pred.covariance_diag is not None:
            covariance_diag.append(pred.covariance_diag.detach().cpu())
        if pred.covariance_blocks is not None:
            covariance_blocks.append(pred.covariance_blocks.detach().cpu())
        calibrations.append(float(state["calibration"].detach().cpu()))

    payload = {
        "displacement": torch.stack(displacements),
        "times": sample["times"],
        "raw_time_indices": sample["raw_time_indices"],
        "source_path": sample["source_path"],
        "dataset": sample["dataset"],
        "calibration": torch.tensor(calibrations),
        "config": ckpt["config"],
    }
    if covariance_diag:
        payload["deformation_covariance_diag"] = torch.stack(covariance_diag)
    if covariance_blocks:
        payload["deformation_covariance_blocks"] = torch.stack(covariance_blocks)

    output = resolve_path(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "dataset": sample["dataset"],
                "source_path": sample["source_path"],
                "displacement_shape": list(payload["displacement"].shape),
                "has_covariance_diag": "deformation_covariance_diag" in payload,
                "has_covariance_blocks": "deformation_covariance_blocks" in payload,
                "note": "deformation decoder is meaningful only after training with deformation supervision or warping loss",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
