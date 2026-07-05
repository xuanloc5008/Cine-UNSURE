#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import load_yaml, project_root, resolve_path
from cunsure_monai3d.foundation import build_foundation, full_jacobian_rows, latent_covariance_from_full_jacobian
from cunsure_monai3d.preprocess import FrameRef, center_crop_or_pad, load_frame, normalize_volume


def load_input_volume(path: Path, time_index: int | None, target_shape: tuple[int, int, int]) -> torch.Tensor:
    ref = FrameRef(path=path, time_index=time_index)
    vol = load_frame(ref, time_axis=-1)
    vol = normalize_volume(vol, mode="percentile", percentile_low=1.0, percentile_high=99.0)
    vol = center_crop_or_pad(vol, target_shape)
    return torch.from_numpy(vol[None, None]).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_path(cfg["cunsure"]["checkpoint"], root)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    eta = ckpt["eta"].to(device)
    train_cfg = ckpt["config"]
    volume_size = tuple(int(v) for v in train_cfg["data"].get("volume_size", [16, 128, 128]))
    if "volume_size" not in train_cfg["data"]:
        volume_size = tuple(int(v) for v in eta.new_tensor([16, 128, 128]).cpu().tolist())

    image_path = resolve_path(cfg["input"]["image"], root)
    time_index = cfg["input"].get("time_index")
    x = load_input_volume(image_path, None if time_index is None else int(time_index), volume_size).to(device)

    foundation_cfg = dict(cfg["foundation"])
    foundation_cfg["repo_path"] = resolve_path(foundation_cfg["repo_path"], root)
    if foundation_cfg["name"] == "cinema" and foundation_cfg.get("cache_dir"):
        foundation_cfg["cache_dir"] = resolve_path(foundation_cfg["cache_dir"], root)
    if foundation_cfg["name"] == "medsam2":
        foundation_cfg["checkpoint"] = resolve_path(foundation_cfg["checkpoint"], root)
    encoder = build_foundation(foundation_cfg, device=device)

    z, jac = full_jacobian_rows(encoder, x, chunk_size=int(cfg["jacobian"]["chunk_size"]))
    sigma_z = latent_covariance_from_full_jacobian(
        jac,
        input_shape=tuple(x.shape[1:]),
        eta=eta,
        device=device,
    )

    out_dir = resolve_path(cfg["input"]["output_dir"], root)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "z": z,
        "eta": eta.detach().cpu(),
        "latent_covariance": sigma_z,
        "image_shape": tuple(x.shape),
        "config": cfg,
    }
    if bool(cfg["jacobian"]["save_jacobian"]):
        payload["jacobian"] = jac
    torch.save(payload, out_dir / "foundation_noise_covariance.pt")
    metrics = {
        "latent_dim": int(z.numel()),
        "jacobian_shape": list(jac.shape),
        "covariance_shape": list(sigma_z.shape),
        "covariance_trace": float(torch.trace(sigma_z)),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
