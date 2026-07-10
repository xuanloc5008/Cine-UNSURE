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
from cunsure_monai3d.draft_sde_rnn import DraftNeuralSDERNN
from cunsure_monai3d.nodeo_ops import (
    LocalNCC3D,
    SpatialTransformer3D,
    identity_grid_voxel,
    negative_jacobian_loss,
    smoothness_loss,
    velocity_magnitude_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def sequence_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    if len(batch) != 1:
        raise ValueError("batch_size must be 1 for variable-length sequence training")
    return batch[0]


def make_loader(dataset: DeformationSequenceDataset, indices: list[int] | None, *, shuffle: bool) -> DataLoader:
    subset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(subset, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=sequence_collate)


def build_model(cfg: dict, *, latent_dim: int, device: torch.device) -> DraftNeuralSDERNN:
    model_cfg = cfg["model"]
    deformation_shape = as_tuple_int(model_cfg["deformation_shape"], name="deformation_shape")
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
        deformation_shape=deformation_shape,
        deformation_covariance=str(model_cfg.get("deformation_covariance", "diag")),
        deformation_jacobian_chunk=int(model_cfg.get("deformation_jacobian_chunk", 512)),
        deformation_scale=float(model_cfg.get("deformation_scale", 4.0)),
        zero_init_deformation=bool(model_cfg.get("zero_init_deformation", True)),
    )
    return model.to(device)


def nodeo_sequence_loss(
    model: DraftNeuralSDERNN,
    batch: dict[str, object],
    *,
    ncc: LocalNCC3D,
    transformer: SpatialTransformer3D,
    device: torch.device,
    jitter: float,
    lambda_j: float,
    lambda_v: float,
    lambda_df: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    images = batch["images"].to(device)
    z = batch["z"].to(device)
    covariance = batch["R"].to(device)
    times = batch["times"].to(device)
    if images.shape[0] < 2:
        zero = z.new_tensor(0.0)
        return zero, {"loss": 0.0, "image": 0.0, "jdet": 0.0, "mag": 0.0, "smooth": 0.0}

    states = model.rollout_sequence(z, covariance, times, jitter=jitter)
    flows = torch.stack([model.decode_deformation(state["h"], state["time"]) for state in states], dim=0)
    reference = images[0:1]
    identity = identity_grid_voxel(tuple(images.shape[-3:]), device=device, dtype=images.dtype)

    image_losses: list[torch.Tensor] = []
    jdet_losses: list[torch.Tensor] = []
    smooth_losses: list[torch.Tensor] = []
    for idx in range(1, images.shape[0]):
        flow = flows[idx][None]
        warped = transformer(reference, flow)
        image_losses.append(ncc(images[idx : idx + 1], warped))
        phi = identity + flow
        jdet_losses.append(negative_jacobian_loss(phi))
        smooth_losses.append(smoothness_loss(flow))

    flow_velocity = (flows[1:] - flows[:-1])[:, None]
    loss_img = torch.stack(image_losses).mean()
    loss_j = torch.stack(jdet_losses).mean()
    loss_s = torch.stack(smooth_losses).mean()
    loss_v = velocity_magnitude_loss(flow_velocity)
    loss = loss_img + float(lambda_j) * loss_j + float(lambda_v) * loss_v + float(lambda_df) * loss_s
    metrics = {
        "loss": float(loss.detach().cpu()),
        "image": float(loss_img.detach().cpu()),
        "jdet": float(loss_j.detach().cpu()),
        "mag": float(loss_v.detach().cpu()),
        "smooth": float(loss_s.detach().cpu()),
    }
    return loss, metrics


def run_epoch(
    model: DraftNeuralSDERNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    loss_cfg: dict,
    deformation_shape: tuple[int, int, int],
    grad_clip: float | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ncc = LocalNCC3D(win=int(loss_cfg.get("ncc_win", 9))).to(device)
    transformer = SpatialTransformer3D(deformation_shape).to(device)
    totals = {"loss": 0.0, "image": 0.0, "jdet": 0.0, "mag": 0.0, "smooth": 0.0}
    count = 0
    for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
        loss, metrics = nodeo_sequence_loss(
            model,
            batch,
            ncc=ncc,
            transformer=transformer,
            device=device,
            jitter=float(loss_cfg.get("jitter", 1.0e-6)),
            lambda_j=float(loss_cfg["lambda_j"]),
            lambda_v=float(loss_cfg["lambda_v"]),
            lambda_df=float(loss_cfg["lambda_df"]),
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
        for key in totals:
            totals[key] += metrics[key]
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()} | {"sequences": float(count)}


def save_checkpoint(path: Path, model: DraftNeuralSDERNN, cfg: dict, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "config": cfg, "metrics": metrics}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_nodeo_sde_deformation.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    set_seed(int(cfg.get("seed", 2026)))
    device = select_device(cfg.get("device", "auto"))

    data_cfg = cfg["data"]
    deformation_shape = as_tuple_int(cfg["model"]["deformation_shape"], name="deformation_shape")
    dataset = DeformationSequenceDataset(
        resolve_path(data_cfg["h5"], root),
        root=root,
        min_length=int(data_cfg.get("min_length", 2)),
        covariance=str(data_cfg.get("covariance", "diag")),
        normalize_time=bool(data_cfg.get("normalize_time", True)),
        time_axis=int(data_cfg.get("time_axis", -1)),
        volume_size=as_tuple_int(data_cfg["volume_size"], name="volume_size"),
        image_size=deformation_shape,
        normalize=str(data_cfg.get("normalize", "percentile")),
        percentile_low=float(data_cfg.get("percentile_low", 1.0)),
        percentile_high=float(data_cfg.get("percentile_high", 99.0)),
        source_path_remap=list(data_cfg.get("source_path_remap", [])),
    )
    if not len(dataset):
        raise ValueError("no deformation sequences found")
    latent_dim = int(dataset[0]["z"].shape[-1])

    index_path = resolve_path(data_cfg.get("sequence_index"), root)
    train_indices = load_split_indices(index_path, "train")
    val_indices = load_split_indices(index_path, "val")
    if train_indices is None:
        order = list(range(len(dataset)))
        random.Random(int(cfg.get("seed", 2026))).shuffle(order)
        val_count = max(1, int(0.1 * len(order)))
        val_indices = order[:val_count]
        train_indices = order[val_count:]

    model = build_model(cfg, latent_dim=latent_dim, device=device)
    train_loader = make_loader(dataset, train_indices, shuffle=True)
    val_loader = make_loader(dataset, val_indices, shuffle=False)
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    run_dir = resolve_path(cfg["output"]["run_dir"], root)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    last_val = {"loss": float("inf")}
    for epoch in range(1, int(optim_cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss_cfg=cfg["loss"],
            deformation_shape=deformation_shape,
            grad_clip=optim_cfg.get("grad_clip"),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device=device,
            loss_cfg=cfg["loss"],
            deformation_shape=deformation_shape,
            grad_clip=None,
        )
        last_val = val_metrics
        log = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        print(json.dumps(log, indent=2))
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(run_dir / "best.pt", model, cfg, epoch, val_metrics)
        if epoch % int(cfg["output"].get("checkpoint_every", 10)) == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", model, cfg, epoch, val_metrics)
    save_checkpoint(run_dir / "last.pt", model, cfg, int(optim_cfg["epochs"]), last_val)


if __name__ == "__main__":
    main()
