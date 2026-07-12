#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import as_tuple_int, load_yaml, project_root, resolve_path, select_device
from cunsure_monai3d.deformation_data import DeformationSequenceDataset
from cunsure_monai3d.nodeo_ops import SpatialTransformer3D
from cunsure_monai3d.nodeo_roi_data import NODEOTrajectoryStore
from cunsure_monai3d.sde_rnn_uncertainty import NeuralSDERNNUncertainty


def build_sde_from_checkpoint(path: Path, *, latent_dim: int, device: torch.device) -> tuple[NeuralSDERNNUncertainty, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    model = NeuralSDERNNUncertainty(
        latent_dim=int(latent_dim),
        hidden_dim=int(model_cfg["hidden_dim"]),
        image_shape=as_tuple_int(model_cfg["image_shape"], name="image_shape"),
        mlp_hidden_dim=int(model_cfg["mlp_hidden_dim"]),
        mlp_layers=int(model_cfg["mlp_layers"]),
        diffusion_scale=float(model_cfg["diffusion_scale"]),
        init_covariance=float(model_cfg["init_covariance"]),
        residual_scale=float(model_cfg["residual_scale"]),
        sde_steps_per_interval=int(model_cfg["sde_steps_per_interval"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--nodeo-summary", action="append")
    parser.add_argument("--h5")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    root = project_root()
    device = select_device(args.device)
    sde_ckpt = torch.load(resolve_path(args.checkpoint, root), map_location="cpu", weights_only=False)
    cfg = sde_ckpt["config"]
    data_cfg = cfg["data"]
    image_shape = as_tuple_int(cfg["model"]["image_shape"], name="image_shape")
    h5_path = args.h5 or data_cfg["h5"]
    dataset = DeformationSequenceDataset(
        resolve_path(h5_path, root),
        root=root,
        min_length=int(data_cfg.get("min_length", 2)),
        covariance=str(data_cfg.get("covariance", "full")),
        normalize_time=bool(data_cfg.get("normalize_time", True)),
        time_axis=int(data_cfg.get("time_axis", -1)),
        volume_size=as_tuple_int(data_cfg["volume_size"], name="volume_size"),
        image_size=image_shape,
        normalize=str(data_cfg.get("normalize", "percentile")),
        percentile_low=float(data_cfg.get("percentile_low", 1.0)),
        percentile_high=float(data_cfg.get("percentile_high", 99.0)),
        source_path_remap=list(data_cfg.get("source_path_remap", [])),
        cache_data=False,
        roi_mask_crop=bool(data_cfg.get("roi_mask_crop", False)),
        roi_mask_margin=as_tuple_int(data_cfg.get("roi_mask_margin", [0, 12, 12]), name="roi_mask_margin"),
        require_roi_mask=bool(data_cfg.get("require_roi_mask", False)),
    )
    sample = dataset[int(args.sequence_index)]
    latent_dim = int(sample["z"].shape[-1])
    summaries = args.nodeo_summary or list(cfg["nodeo"]["trajectory_summaries"])
    trajectory_store = NODEOTrajectoryStore(summaries, root=root)
    sde_rnn, _ = build_sde_from_checkpoint(resolve_path(args.checkpoint, root), latent_dim=latent_dim, device=device)

    images = sample["images"].to(device)
    times = sample["times"].to(device)
    z = sample["z"].to(device)
    r = sample["R"].to(device)
    phi_bar = trajectory_store.load(str(sample["source_path"])).to(device)
    if phi_bar.shape[0] != images.shape[0]:
        raise ValueError(f"NODEO/latent sequence length mismatch for {sample['source_path']}")
    identity = phi_bar[0:1]
    nodeo_displacement = phi_bar - identity
    sde_out = sde_rnn(times=times, z=z, r=r, phi_bar=phi_bar)

    factors: list[torch.Tensor] = []
    for idx in range(int(times.numel())):
        factor = sde_rnn.output.covariance_factor(
            sde_out.hidden_mean[idx],
            sde_out.hidden_covariance[idx],
            times[idx],
            jitter=float(cfg["loss"].get("jitter", 1.0e-6)),
        )
        factors.append(factor.cpu())
    deformation_covariance_factor = torch.stack(factors)

    with torch.no_grad():
        transformer = SpatialTransformer3D(image_shape).to(device)
        reference = images[0:1].expand(images.shape[0], -1, -1, -1, -1)
        warped_mean = transformer(reference, nodeo_displacement)
        warped_sde = transformer(reference, sde_out.total_displacement)

    payload = {
        "phi_bar": phi_bar.detach().cpu(),
        "phi": sde_out.phi.detach().cpu(),
        "nodeo_displacement": nodeo_displacement.detach().cpu(),
        "residual_displacement": sde_out.residual_displacement.detach().cpu(),
        "total_displacement": sde_out.total_displacement.detach().cpu(),
        "hidden_mean": sde_out.hidden_mean.detach().cpu(),
        "hidden_covariance": sde_out.hidden_covariance.detach().cpu(),
        "deformation_covariance_factor": deformation_covariance_factor,
        "warped_nodeo_mean": warped_mean.detach().cpu(),
        "warped_sde_mean": warped_sde.detach().cpu(),
        "images": sample["images"],
        "z": sample["z"],
        "latent_covariance": sample["R"],
        "times": sample["times"],
        "raw_time_indices": sample["raw_time_indices"],
        "source_path": sample["source_path"],
        "dataset": sample["dataset"],
        "config": cfg,
    }
    output = resolve_path(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "dataset": sample["dataset"],
                "source_path": sample["source_path"],
                "phi_shape": list(payload["phi"].shape),
                "residual_displacement_shape": list(payload["residual_displacement"].shape),
                "hidden_covariance_shape": list(payload["hidden_covariance"].shape),
                "deformation_covariance_factor_shape": list(payload["deformation_covariance_factor"].shape),
                "covariance_definition": "R_phi[k] = deformation_covariance_factor[k] @ deformation_covariance_factor[k].T",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
