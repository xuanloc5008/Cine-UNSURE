#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import as_tuple_int, project_root, resolve_path, select_device
from cunsure_monai3d.deformation_data import DeformationSequenceDataset
from cunsure_monai3d.draft_sde_rnn import DraftNeuralSDERNN
from cunsure_monai3d.nodeo_ops import SpatialTransformer3D, identity_grid_voxel


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
        deformation_shape=as_tuple_int(model_cfg["deformation_shape"], name="deformation_shape"),
        deformation_covariance=str(model_cfg.get("deformation_covariance", "diag")),
        deformation_jacobian_chunk=int(model_cfg.get("deformation_jacobian_chunk", 512)),
        deformation_scale=float(model_cfg.get("deformation_scale", 4.0)),
        zero_init_deformation=False,
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
    ckpt = torch.load(resolve_path(args.checkpoint, root), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    deformation_shape = as_tuple_int(cfg["model"]["deformation_shape"], name="deformation_shape")
    dataset = DeformationSequenceDataset(
        resolve_path(args.h5, root),
        root=root,
        min_length=int(data_cfg.get("min_length", 2)),
        covariance=args.covariance,
        normalize_time=bool(data_cfg.get("normalize_time", True)),
        time_axis=int(data_cfg.get("time_axis", -1)),
        volume_size=as_tuple_int(data_cfg["volume_size"], name="volume_size"),
        image_size=deformation_shape,
        normalize=str(data_cfg.get("normalize", "percentile")),
        percentile_low=float(data_cfg.get("percentile_low", 1.0)),
        percentile_high=float(data_cfg.get("percentile_high", 99.0)),
        source_path_remap=list(data_cfg.get("source_path_remap", [])),
    )
    sample = dataset[int(args.sequence_index)]
    latent_dim = int(sample["z"].shape[-1])
    model = build_model_from_checkpoint(ckpt, latent_dim=latent_dim, device=device)

    images = sample["images"].to(device)
    z = sample["z"].to(device)
    r = sample["R"].to(device)
    times = sample["times"].to(device)
    states = model.infer_sequence_states(z, r, times, jitter=float(args.jitter))
    transformer = SpatialTransformer3D(deformation_shape).to(device)
    identity = identity_grid_voxel(deformation_shape, device=device, dtype=images.dtype)

    displacements: list[torch.Tensor] = []
    phi: list[torch.Tensor] = []
    warped: list[torch.Tensor] = []
    covariance_diag: list[torch.Tensor] = []
    covariance_blocks: list[torch.Tensor] = []
    calibrations: list[float] = []
    reference = images[0:1]
    for state in states:
        pred = model.deformation_output(state["h"], state["P"], state["time"])
        flow = pred.displacement.detach()
        displacements.append(flow.cpu())
        phi.append((identity + flow[None]).squeeze(0).cpu())
        warped.append(transformer(reference, flow[None]).detach().squeeze(0).cpu())
        if pred.covariance_diag is not None:
            covariance_diag.append(pred.covariance_diag.detach().cpu())
        if pred.covariance_blocks is not None:
            covariance_blocks.append(pred.covariance_blocks.detach().cpu())
        calibrations.append(float(state["calibration"].detach().cpu()))

    payload = {
        "displacement": torch.stack(displacements),
        "phi": torch.stack(phi),
        "warped": torch.stack(warped),
        "images": sample["images"],
        "times": sample["times"],
        "raw_time_indices": sample["raw_time_indices"],
        "source_path": sample["source_path"],
        "dataset": sample["dataset"],
        "calibration": torch.tensor(calibrations),
        "config": cfg,
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
