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

from cardiac_nodeo_uq.config import as_tuple_int, load_yaml, project_root, resolve_path, select_device
from cardiac_nodeo_uq.nodeo_dir import NODEODIRModel
from cardiac_nodeo_uq.nodeo_ops import (
    LocalNCC3D,
    SpatialTransformer3D,
    nodeo_jacobian_metrics,
    smoothness_loss,
    velocity_magnitude_loss,
)
from cardiac_nodeo_uq.nodeo_roi_data import NODEOROISequenceDataset


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
        step_size=float(model_cfg.get("step_size", 0.05)),
        rtol=float(model_cfg.get("rtol", 1.0e-6)),
        atol=float(model_cfg.get("atol", 1.0e-8)),
        encoder_channels=int(model_cfg["encoder_channels"]),
        encoder_depth=int(model_cfg["encoder_depth"]),
        output_downsamples=int(model_cfg["output_downsamples"]),
        bottleneck_dim=int(model_cfg["bottleneck_dim"]),
        smoothing_kernel=str(model_cfg["smoothing_kernel"]),
        smoothing_window=int(model_cfg["smoothing_window"]),
        smoothing_sigma=float(model_cfg["smoothing_sigma"]),
        smoothing_passes=int(model_cfg["smoothing_passes"]),
        time_encoding=str(model_cfg.get("time_encoding", "scalar")),
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
    (
        jdet_loss,
        jdet_lower_loss,
        jdet_upper_loss,
        fold_fraction,
        volume_deviation,
        jacobian_minimum,
        jacobian_maximum,
    ) = nodeo_jacobian_metrics(
        output.phi_voxel[1:],
        minimum=float(loss_cfg["minimum_jacobian"]),
        maximum=float(loss_cfg.get("maximum_jacobian", 4.0)),
    )
    magnitude_loss = velocity_magnitude_loss(output.velocity_normalized[1:, None])
    deformation_smoothness = smoothness_loss(output.displacement_voxel[1:])
    cycle_loss = output.displacement_voxel[-1].square().mean()
    total = (
        image_loss
        + float(loss_cfg["lambda_j"]) * jdet_loss
        + float(loss_cfg["lambda_v"]) * magnitude_loss
        + float(loss_cfg["lambda_df"]) * deformation_smoothness
        + float(loss_cfg.get("lambda_cycle", 0.0)) * cycle_loss
    )
    terms = {
        "loss": total,
        "image": image_loss,
        "jdet": jdet_loss,
        "jdet_lower": jdet_lower_loss,
        "jdet_upper": jdet_upper_loss,
        "mag": magnitude_loss,
        "smooth": deformation_smoothness,
        "cycle": cycle_loss,
        "cycle_displacement_rms": cycle_loss.sqrt(),
        "fold_fraction": fold_fraction,
        "abs_jdet_minus_one": volume_deviation,
        "jacobian_min": jacobian_minimum,
        "jacobian_max": jacobian_maximum,
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
    uncertainty_cfg = cfg.get("uncertainty", {})
    uncertainty_method = str(
        uncertainty_cfg.get("method", "late_checkpoint_ensemble")
    )
    if uncertainty_method != "late_checkpoint_ensemble":
        raise ValueError(
            "uncertainty.method must be 'late_checkpoint_ensemble'"
        )
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
    max_runtime_seconds = optim_cfg.get("max_runtime_seconds")
    max_runtime_seconds = None if max_runtime_seconds is None else float(max_runtime_seconds)
    minimum_epochs = min(int(optim_cfg.get("minimum_epochs", 1)), epochs)
    best_score = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    uncertainty_tail = min(
        int(uncertainty_cfg.get("model_uncertainty_tail", selection_tail)), epochs
    )
    uncertainty_stride = max(int(uncertainty_cfg.get("model_uncertainty_stride", 5)), 1)
    uncertainty_scale = float(uncertainty_cfg.get("model_uncertainty_scale", 1.0))
    displacement_sample_count = 0
    displacement_sample_mean: torch.Tensor | None = None
    displacement_sample_m2: torch.Tensor | None = None
    started = time.monotonic()

    for epoch in range(1, epochs + 1):
        model.train()
        loss, terms, epoch_output, _ = compute_loss(
            model,
            images,
            times,
            ncc=ncc,
            transformer=transformer,
            loss_cfg=cfg["loss"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        metrics = detached_metrics(terms)
        if epoch > epochs - selection_tail:
            selection_score = metrics["loss"]
            if selection_score < best_score:
                best_score = selection_score
                best_epoch = epoch
                best_metrics = metrics
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        uncertainty_start = epochs - uncertainty_tail + 1
        if epoch >= uncertainty_start and (epoch - uncertainty_start) % uncertainty_stride == 0:
            sample = epoch_output.displacement_voxel.detach().cpu().float()
            displacement_sample_count += 1
            if displacement_sample_mean is None:
                displacement_sample_mean = sample.clone()
                displacement_sample_m2 = torch.zeros_like(sample)
            else:
                assert displacement_sample_m2 is not None
                delta = sample - displacement_sample_mean
                displacement_sample_mean.add_(delta / float(displacement_sample_count))
                displacement_sample_m2.add_(delta * (sample - displacement_sample_mean))
        optimizer.step()
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
        if (
            max_runtime_seconds is not None
            and epoch >= minimum_epochs
            and time.monotonic() - started >= max_runtime_seconds
        ):
            print(
                json.dumps(
                    {
                        "sequence_id": batch["sequence_id"],
                        "stopped_at_epoch": epoch,
                        "reason": "max_runtime_seconds",
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
            )
            break

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
    if displacement_sample_count >= 2:
        assert displacement_sample_m2 is not None
        model_uncertainty_variance = (
            displacement_sample_m2 / float(displacement_sample_count - 1)
        ) * uncertainty_scale
    else:
        model_uncertainty_variance = torch.zeros_like(output.displacement_voxel.detach().cpu())
        uncertainty_method = "insufficient_samples_zero_fallback"
    model_uncertainty_variance[0].zero_()
    final_metrics["mean_nodeo_model_variance"] = float(model_uncertainty_variance.mean())
    final_metrics["nodeo_model_uncertainty_samples"] = float(displacement_sample_count)
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
            "model_uncertainty_variance_diag": model_uncertainty_variance,
            "model_uncertainty_method": uncertainty_method,
            "model_uncertainty_samples": displacement_sample_count,
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
    parser.add_argument("--config", default="configs/acdc/train_nodeo_dir_roi.yaml")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("--shard-id must satisfy 0 <= shard-id < num-shards")

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
    summary_path = (
        output_dir / "summary.jsonl"
        if args.num_shards == 1
        else output_dir / f"summary.shard{args.shard_id:03d}.jsonl"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(str(json.loads(line)["sequence_id"]))

    indices = [
        index
        for index in range(args.start_index, stop)
        if index % args.num_shards == args.shard_id
    ]
    for index in tqdm(indices, desc=f"NODEO {args.split} shard {args.shard_id + 1}/{args.num_shards}"):
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
