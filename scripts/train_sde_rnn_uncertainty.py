#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import as_tuple_int, load_yaml, project_root, resolve_path, select_device
from cunsure_monai3d.deformation_data import DeformationSequenceDataset
from cunsure_monai3d.nodeo_ops import LocalNCC3D, SpatialTransformer3D, negative_jacobian_loss, smoothness_loss
from cunsure_monai3d.nodeo_roi_data import NODEOTrajectoryStore, canonical_source_key
from cunsure_monai3d.sde_rnn_uncertainty import NeuralSDERNNUncertainty


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_speed(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def sequence_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    if len(batch) != 1:
        raise ValueError("batch_size must be 1 for variable-length SDE-RNN sequence training")
    return batch[0]


def load_split_indices(index_path: Path | None, split: str) -> list[int] | None:
    if index_path is None or not index_path.exists():
        return None
    indices: list[int] = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == split:
                indices.append(int(row["sequence_id"]))
    return indices


def make_loader(
    dataset: DeformationSequenceDataset,
    indices: list[int] | None,
    *,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> DataLoader:
    subset = Subset(dataset, indices) if indices is not None else dataset
    kwargs = {
        "batch_size": 1,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "collate_fn": sequence_collate,
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(subset, **kwargs)


def build_sde_rnn(cfg: dict, *, latent_dim: int, device: torch.device) -> NeuralSDERNNUncertainty:
    model_cfg = cfg["model"]
    return NeuralSDERNNUncertainty(
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


def sde_rnn_sequence_loss(
    *,
    sde_rnn: NeuralSDERNNUncertainty,
    trajectory_store: NODEOTrajectoryStore,
    batch: dict[str, object],
    ncc: LocalNCC3D,
    transformer: SpatialTransformer3D,
    device: torch.device,
    lambda_j: float,
    lambda_df: float,
    lambda_unc: float,
    uncertainty_variance_floor: float,
    uncertainty_jitter: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    images = batch["images"].to(device)
    z = batch["z"].to(device)
    r = batch["R"].to(device)
    times = batch["times"].to(device)
    if images.shape[0] < 2:
        zero = images.new_tensor(0.0)
        return zero, {"loss": 0.0, "image": 0.0, "jdet": 0.0, "smooth": 0.0, "unc_nll": 0.0, "mean_image_variance": 0.0}

    phi_bar = trajectory_store.load(str(batch["source_path"])).to(device)
    if phi_bar.shape[0] != images.shape[0]:
        raise ValueError(f"NODEO/latent sequence length mismatch for {batch['source_path']}")
    out = sde_rnn(times=times, z=z, r=r, phi_bar=phi_bar)

    target = images[1:]
    displacement = out.total_displacement[1:]
    phi = out.phi[1:]
    reference = images[0:1].expand(target.shape[0], -1, -1, -1, -1)
    warped = transformer(reference, displacement)

    loss_img = ncc(target, warped)
    loss_j = negative_jacobian_loss(phi)
    loss_s = smoothness_loss(out.residual_displacement[1:])

    # First-order image covariance: delta I = grad(I_warped)^T delta phi.
    # L_phi is kept low rank, so only diag(J_I L_phi L_phi^T J_I^T)
    # is formed and the dense voxel covariance is never materialized.
    uncertainty_terms: list[torch.Tensor] = []
    variance_terms: list[torch.Tensor] = []
    for frame in range(1, images.shape[0]):
        factor = sde_rnn.output.covariance_factor(
            out.hidden_mean[frame],
            out.hidden_covariance[frame],
            times[frame],
            jitter=float(uncertainty_jitter),
            covariance_grad=True,
        ).reshape(3, *sde_rnn.image_shape, sde_rnn.hidden_dim)
        warped_frame = warped[frame - 1, 0]
        gradients = torch.stack(torch.gradient(warped_frame, dim=(0, 1, 2)), dim=0)
        image_factor = torch.einsum("cdhw,cdhwq->dhwq", gradients, factor)
        image_variance = image_factor.pow(2).sum(dim=-1) + float(uncertainty_variance_floor)
        residual = target[frame - 1, 0] - warped_frame
        uncertainty_terms.append(0.5 * (residual.pow(2) / image_variance + image_variance.log()).mean())
        variance_terms.append(image_variance.mean())
    loss_unc = torch.stack(uncertainty_terms).mean()
    mean_image_variance = torch.stack(variance_terms).mean()
    loss = loss_img + float(lambda_j) * loss_j + float(lambda_df) * loss_s + float(lambda_unc) * loss_unc
    metrics = {
        "loss": float(loss.detach().cpu()),
        "image": float(loss_img.detach().cpu()),
        "jdet": float(loss_j.detach().cpu()),
        "smooth": float(loss_s.detach().cpu()),
        "unc_nll": float(loss_unc.detach().cpu()),
        "mean_image_variance": float(mean_image_variance.detach().cpu()),
    }
    return loss, metrics


def run_epoch(
    *,
    sde_rnn: NeuralSDERNNUncertainty,
    trajectory_store: NODEOTrajectoryStore,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_cfg: dict,
    image_shape: tuple[int, int, int],
    grad_clip: float | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    sde_rnn.train(is_train)
    ncc = LocalNCC3D(win=int(loss_cfg.get("ncc_win", 9))).to(device)
    transformer = SpatialTransformer3D(image_shape).to(device)
    totals = {"loss": 0.0, "image": 0.0, "jdet": 0.0, "smooth": 0.0, "unc_nll": 0.0, "mean_image_variance": 0.0}
    count = 0
    for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
        loss, metrics = sde_rnn_sequence_loss(
            sde_rnn=sde_rnn,
            trajectory_store=trajectory_store,
            batch=batch,
            ncc=ncc,
            transformer=transformer,
            device=device,
            lambda_j=float(loss_cfg["lambda_j"]),
            lambda_df=float(loss_cfg["lambda_df"]),
            lambda_unc=float(loss_cfg.get("lambda_unc", 0.01)),
            uncertainty_variance_floor=float(loss_cfg.get("uncertainty_variance_floor", 1.0e-4)),
            uncertainty_jitter=float(loss_cfg.get("jitter", 1.0e-6)),
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(sde_rnn.parameters(), float(grad_clip))
            optimizer.step()
        for key in totals:
            totals[key] += metrics[key]
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()} | {"sequences": float(count)}


def save_checkpoint(path: Path, model: NeuralSDERNNUncertainty, cfg: dict, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "config": cfg, "metrics": metrics}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_sde_rnn_uncertainty.yaml")
    parser.add_argument("--mode", choices=("full", "pilot"), default="full")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    checkpoint_payload = None
    if args.checkpoint:
        checkpoint_payload = torch.load(resolve_path(args.checkpoint, root), map_location="cpu", weights_only=False)
    set_seed(int(cfg.get("seed", 2026)))
    device = select_device(cfg.get("device", "auto"))
    configure_torch_speed(device)

    data_cfg = cfg["data"]
    image_shape = as_tuple_int(cfg["model"]["image_shape"], name="image_shape")
    dataset = DeformationSequenceDataset(
        resolve_path(data_cfg["h5"], root),
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
        cache_data=bool(data_cfg.get("cache_data", False)),
        roi_mask_crop=bool(data_cfg.get("roi_mask_crop", False)),
        roi_mask_margin=as_tuple_int(data_cfg.get("roi_mask_margin", [0, 12, 12]), name="roi_mask_margin"),
        require_roi_mask=bool(data_cfg.get("require_roi_mask", False)),
    )
    if not len(dataset):
        raise ValueError("no SDE-RNN uncertainty sequences found")

    index_path = resolve_path(data_cfg.get("sequence_index"), root)
    train_indices = load_split_indices(index_path, "train")
    val_indices = load_split_indices(index_path, "val")
    if train_indices is None:
        order = list(range(len(dataset)))
        random.Random(int(cfg.get("seed", 2026))).shuffle(order)
        val_count = max(1, int(0.1 * len(order)))
        val_indices = order[:val_count]
        train_indices = order[val_count:]

    configured_summaries = list(cfg["nodeo"]["trajectory_summaries"])
    if args.mode == "pilot":
        existing_summaries = [
            summary
            for summary in configured_summaries
            if (Path(summary).is_absolute() and Path(summary).exists())
            or (not Path(summary).is_absolute() and (root / summary).exists())
        ]
        if not existing_summaries:
            raise FileNotFoundError("pilot mode found no completed NODEO summary files")
        missing_summaries = [summary for summary in configured_summaries if summary not in existing_summaries]
        if missing_summaries:
            print(json.dumps({"pilot_ignored_missing_nodeo_summaries": missing_summaries}, indent=2))
        trajectory_store = NODEOTrajectoryStore(existing_summaries, root=root)
    else:
        trajectory_store = NODEOTrajectoryStore(configured_summaries, root=root)
    available = [
        index
        for index, ref in enumerate(dataset.refs)
        if canonical_source_key(ref.source_path) in trajectory_store.outputs
    ]
    if args.mode == "pilot":
        saved_selection = None if checkpoint_payload is None else checkpoint_payload["config"].get("pilot_selection")
        if args.eval_only and saved_selection:
            train_indices = [int(index) for index in saved_selection["train_indices"]]
            val_indices = [int(index) for index in saved_selection["val_indices"]]
            calibration_indices = [int(index) for index in saved_selection["calibration_indices"]]
            cfg["pilot_selection"] = saved_selection
            cfg["run_mode"] = "pilot"
        else:
            pilot_cfg = cfg.get("pilot", {})
            rng = random.Random(int(pilot_cfg.get("seed", cfg.get("seed", 2026))))
            rng.shuffle(available)
            selected = available[: int(pilot_cfg.get("max_sequences", 62))]
            if len(selected) < 2:
                raise ValueError(f"pilot mode requires at least 2 completed NODEO trajectories, found {len(selected)}")
            calibration_count = max(1, round(len(selected) * float(pilot_cfg.get("calibration_fraction", 0.15))))
            val_count = max(1, round(len(selected) * float(pilot_cfg.get("val_fraction", 0.15))))
            if calibration_count + val_count >= len(selected):
                raise ValueError("pilot validation and calibration splits leave no training sequences")
            calibration_indices = sorted(selected[:calibration_count])
            val_indices = sorted(selected[calibration_count : calibration_count + val_count])
            train_indices = sorted(selected[calibration_count + val_count :])
            cfg["run_mode"] = "pilot"
            cfg["pilot_selection"] = {
                "available_trajectories": len(available),
                "selected_sequences": len(selected),
                "train_sequences": len(train_indices),
                "val_sequences": len(val_indices),
                "calibration_sequences": len(calibration_indices),
                "train_indices": train_indices,
                "val_indices": val_indices,
                "calibration_indices": calibration_indices,
                "train_sources": [dataset.refs[index].source_path for index in train_indices],
                "val_sources": [dataset.refs[index].source_path for index in val_indices],
                "calibration_sources": [dataset.refs[index].source_path for index in calibration_indices],
            }
    else:
        train_indices = [index for index in (train_indices or []) if index in available]
        val_indices = [index for index in (val_indices or []) if index in available]
        cfg["run_mode"] = "full"
    if not train_indices or not val_indices:
        raise ValueError(
            f"mode={args.mode} has train={len(train_indices or [])}, val={len(val_indices or [])} sequences "
            "with completed NODEO trajectories"
        )
    split_report = {"mode": args.mode, "train_sequences": len(train_indices), "val_sequences": len(val_indices)}
    if args.mode == "pilot":
        split_report["calibration_sequences"] = len(calibration_indices)
    print(json.dumps(split_report, indent=2))
    latent_dim = int(dataset[0]["z"].shape[-1])
    sde_rnn = build_sde_rnn(cfg, latent_dim=latent_dim, device=device)
    if checkpoint_payload is not None:
        sde_rnn.load_state_dict(checkpoint_payload["model_state"], strict=True)

    train_loader = make_loader(
        dataset,
        train_indices,
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", device.type == "cuda")),
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        prefetch_factor=data_cfg.get("prefetch_factor"),
    )
    val_loader = make_loader(
        dataset,
        val_indices,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", device.type == "cuda")),
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        prefetch_factor=data_cfg.get("prefetch_factor"),
    )

    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        sde_rnn.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
    configured_run_dir = str(cfg["output"]["run_dir"])
    run_dir = resolve_path(
        f"{configured_run_dir}_pilot62" if args.mode == "pilot" else configured_run_dir,
        root,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "pilot":
        (run_dir / "pilot_split.json").write_text(
            json.dumps(cfg["pilot_selection"], indent=2),
            encoding="utf-8",
        )
    metrics_path = run_dir / "metrics.jsonl"
    if args.eval_only:
        if not args.checkpoint:
            parser.error("--eval-only requires --checkpoint")
        val_metrics = run_epoch(
            sde_rnn=sde_rnn,
            trajectory_store=trajectory_store,
            loader=val_loader,
            optimizer=None,
            device=device,
            loss_cfg=cfg["loss"],
            image_shape=image_shape,
            grad_clip=None,
        )
        evaluation = {"mode": args.mode, "checkpoint": args.checkpoint, "val": val_metrics}
        (run_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        print(json.dumps(evaluation, indent=2))
        return
    best_val = float("inf")
    patience = int(optim_cfg.get("early_stopping_patience", 0))
    min_delta = float(optim_cfg.get("early_stopping_min_delta", 0.0))
    stale_epochs = 0

    for epoch in range(1, int(optim_cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            sde_rnn=sde_rnn,
            trajectory_store=trajectory_store,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_cfg=cfg["loss"],
            image_shape=image_shape,
            grad_clip=optim_cfg.get("grad_clip"),
        )
        val_metrics = run_epoch(
            sde_rnn=sde_rnn,
            trajectory_store=trajectory_store,
            loader=val_loader,
            optimizer=None,
            device=device,
            loss_cfg=cfg["loss"],
            image_shape=image_shape,
            grad_clip=None,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        print(json.dumps(row, indent=2))
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if val_metrics["loss"] < best_val - min_delta:
            best_val = val_metrics["loss"]
            stale_epochs = 0
            save_checkpoint(run_dir / "best.pt", sde_rnn, cfg, epoch, val_metrics)
        else:
            stale_epochs += 1

        if epoch % int(cfg["output"].get("checkpoint_every", 10)) == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", sde_rnn, cfg, epoch, val_metrics)

        if patience > 0 and stale_epochs >= patience:
            print(json.dumps({"early_stopped": True, "epoch": epoch, "best_val_loss": best_val}, indent=2))
            break


if __name__ == "__main__":
    main()
