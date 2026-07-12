#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import as_tuple_int, load_yaml, project_root, resolve_path, select_device
from cunsure_monai3d.nodeo_dir import NODEODIRModel
from cunsure_monai3d.nodeo_ops import (
    LocalNCC3D,
    SpatialTransformer3D,
    nodeo_jacobian_metrics,
    smoothness_loss,
    velocity_magnitude_loss,
)
from cunsure_monai3d.nodeo_roi_data import NODEOROISequenceDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def build_model(cfg: dict, device: torch.device) -> NODEODIRModel:
    model_cfg = cfg["model"]
    return NODEODIRModel(
        image_shape=as_tuple_int(model_cfg["image_shape"], name="image_shape"),
        solver=str(model_cfg["solver"]),
        step_size=float(model_cfg["step_size"]),
        encoder_channels=int(model_cfg["encoder_channels"]),
        encoder_depth=int(model_cfg["encoder_depth"]),
        output_downsamples=int(model_cfg["output_downsamples"]),
        bottleneck_dim=int(model_cfg["bottleneck_dim"]),
        smoothing_kernel=str(model_cfg["smoothing_kernel"]),
        smoothing_window=int(model_cfg["smoothing_window"]),
        smoothing_sigma=float(model_cfg["smoothing_sigma"]),
        smoothing_passes=int(model_cfg["smoothing_passes"]),
    ).to(device)


def compute_loss(
    model: NODEODIRModel,
    images: torch.Tensor,
    times: torch.Tensor,
    *,
    ncc: LocalNCC3D,
    transformer: SpatialTransformer3D,
    loss_cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], object, torch.Tensor]:
    output = model.integrate_sequence(times)
    target = images[1:]
    reference = images[0:1].expand(target.shape[0], -1, -1, -1, -1)
    warped = transformer(reference, output.displacement_voxel[1:])
    image_loss = ncc(target, warped)
    jdet_loss, fold_fraction, volume_deviation = nodeo_jacobian_metrics(
        output.phi_voxel[1:], minimum=float(loss_cfg["minimum_jacobian"])
    )
    magnitude_loss = velocity_magnitude_loss(output.velocity_normalized[1:, None])
    deformation_smoothness = smoothness_loss(output.displacement_voxel[1:])
    total = (
        image_loss
        + float(loss_cfg["lambda_j"]) * jdet_loss
        + float(loss_cfg["lambda_v"]) * magnitude_loss
        + float(loss_cfg["lambda_df"]) * deformation_smoothness
    )
    terms = {
        "loss": total,
        "image": image_loss,
        "jdet": jdet_loss,
        "mag": magnitude_loss,
        "smooth": deformation_smoothness,
        "fold_fraction": fold_fraction,
        "abs_jdet_minus_one": volume_deviation,
    }
    return total, terms, output, warped


def detached_metrics(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in terms.items()}


def fit_sequence(
    batch: dict[str, object],
    cfg: dict,
    *,
    device: torch.device,
    output_path: Path,
) -> dict[str, object]:
    images = batch["images"].to(device, non_blocking=True)
    times = batch["times"].to(device, non_blocking=True)
    model = build_model(cfg, device)
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        amsgrad=True,
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
    image_shape = as_tuple_int(cfg["model"]["image_shape"], name="image_shape")
    ncc = LocalNCC3D(win=int(cfg["loss"]["ncc_win"])).to(device)
    transformer = SpatialTransformer3D(image_shape).to(device)
    epochs = int(optim_cfg["epochs_per_sequence"])
    selection_tail = min(int(optim_cfg.get("selection_tail", 50)), epochs)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        loss, terms, _, _ = compute_loss(
            model,
            images,
            times,
            ncc=ncc,
            transformer=transformer,
            loss_cfg=cfg["loss"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics = detached_metrics(terms)
        if epoch > epochs - selection_tail:
            balance_score = 1000.0 * metrics["image"] * (
                float(cfg["loss"]["lambda_j"]) * metrics["jdet"] + 1.0e-12
            )
            if balance_score < best_score:
                best_score = balance_score
                best_epoch = epoch
                best_metrics = metrics
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % int(optim_cfg.get("log_every", 20)) == 0:
            print(
                json.dumps(
                    {
                        "sequence_id": batch["sequence_id"],
                        "epoch": epoch,
                        **metrics,
                    }
                )
            )

    if best_state is None or best_metrics is None:
        raise RuntimeError("NODEO did not produce a selected state")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        _, final_terms, output, warped = compute_loss(
            model,
            images,
            times,
            ncc=ncc,
            transformer=transformer,
            loss_cfg=cfg["loss"],
        )
    final_metrics = detached_metrics(final_terms)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sequence_id": batch["sequence_id"],
            "split": batch["split"],
            "dataset": batch["dataset"],
            "source_path": batch["source_path"],
            "raw_time_indices": batch["raw_time_indices"],
            "times": batch["times"],
            "images": batch["images"].half(),
            "warped": warped.detach().cpu().half(),
            "phi_bar": output.phi_voxel.detach().cpu().half(),
            "displacement": output.displacement_voxel.detach().cpu().half(),
            "velocity": output.velocity_normalized.detach().cpu().half(),
            "model_state": best_state,
            "config": cfg,
            "best_epoch": best_epoch,
            "metrics": final_metrics,
        },
        output_path,
    )
    return {
        "sequence_id": batch["sequence_id"],
        "split": batch["split"],
        "dataset": batch["dataset"],
        "source_path": batch["source_path"],
        "output": str(output_path),
        "best_epoch": best_epoch,
        "runtime_seconds": time.monotonic() - started,
        "metrics": final_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_nodeo_dir_roi.yaml")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    set_seed(int(cfg.get("seed", 2026)))
    device = select_device(cfg.get("device", "auto"))
    configure_torch(device)
    dataset = NODEOROISequenceDataset(
        resolve_path(cfg["data"]["manifest"], root),
        root=root,
        split=args.split,
        min_length=int(cfg["data"].get("min_length", 2)),
    )
    stop = len(dataset) if args.limit is None else min(len(dataset), args.start_index + args.limit)
    output_dir = resolve_path(cfg["output"]["run_dir"], root) / args.split
    summary_path = output_dir / "summary.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(str(json.loads(line)["sequence_id"]))

    for index in tqdm(range(args.start_index, stop), desc=f"NODEO {args.split}"):
        batch = dataset[index]
        output_path = output_dir / f"{index:06d}_{batch['sequence_id']}.pt"
        if output_path.exists() and not args.overwrite:
            if str(batch["sequence_id"]) not in completed:
                payload = torch.load(output_path, map_location="cpu", weights_only=False)
                recovered = {
                    "sequence_id": payload["sequence_id"],
                    "split": payload["split"],
                    "dataset": payload["dataset"],
                    "source_path": payload["source_path"],
                    "output": str(output_path),
                    "best_epoch": payload["best_epoch"],
                    "runtime_seconds": None,
                    "metrics": payload["metrics"],
                }
                with summary_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(recovered) + "\n")
                completed.add(str(batch["sequence_id"]))
            continue
        row = fit_sequence(batch, cfg, device=device, output_path=output_path)
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        completed.add(str(batch["sequence_id"]))
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
