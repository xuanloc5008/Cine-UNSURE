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
from cunsure_monai3d.nodeo_mean import NODEOMeanDeformationModel
from cunsure_monai3d.nodeo_ops import SpatialTransformer3D


def build_model_from_checkpoint(ckpt: dict, *, device: torch.device) -> NODEOMeanDeformationModel:
    cfg = ckpt["config"]
    model_cfg = cfg["model"]
    model = NODEOMeanDeformationModel(
        image_shape=as_tuple_int(model_cfg["image_shape"], name="image_shape"),
        channels=tuple(int(v) for v in model_cfg.get("channels", [32, 32, 32])),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        velocity_scale=float(model_cfg.get("velocity_scale", 4.0)),
        smoothing_kernel=int(model_cfg.get("smoothing_kernel", 3)),
        ode_steps_per_interval=int(model_cfg.get("ode_steps_per_interval", 4)),
        zero_init_velocity=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--covariance", default="diag", choices=["diag", "full"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    root = project_root()
    device = select_device(args.device)
    ckpt = torch.load(resolve_path(args.checkpoint, root), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    image_shape = as_tuple_int(cfg["model"]["image_shape"], name="image_shape")
    h5_path = args.h5 or data_cfg["h5"]
    dataset = DeformationSequenceDataset(
        resolve_path(h5_path, root),
        root=root,
        min_length=int(data_cfg.get("min_length", 2)),
        covariance=args.covariance,
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
    model = build_model_from_checkpoint(ckpt, device=device)

    images = sample["images"].to(device)
    times = sample["times"].to(device)
    with torch.no_grad():
        nodeo = model.integrate_sequence(times)
        transformer = SpatialTransformer3D(image_shape).to(device)
        reference = images[0:1].expand(images.shape[0], -1, -1, -1, -1)
        warped = transformer(reference, nodeo.displacement)

    payload = {
        "phi_bar": nodeo.phi.detach().cpu(),
        "displacement": nodeo.displacement.detach().cpu(),
        "velocity": nodeo.velocity.detach().cpu(),
        "warped": warped.detach().cpu(),
        "images": sample["images"],
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
                "phi_bar_shape": list(payload["phi_bar"].shape),
                "displacement_shape": list(payload["displacement"].shape),
                "warped_shape": list(payload["warped"].shape),
                "has_uncertainty": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

