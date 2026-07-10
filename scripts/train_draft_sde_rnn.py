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

from cunsure_monai3d.config import load_yaml, project_root, resolve_path, select_device
from cunsure_monai3d.draft_sde_rnn import DraftNeuralSDERNN
from cunsure_monai3d.sde_data import LatentObservationSequenceDataset


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


def make_loader(dataset: LatentObservationSequenceDataset, indices: list[int] | None, *, shuffle: bool) -> DataLoader:
    subset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(subset, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=sequence_collate)


def run_epoch(
    model: DraftNeuralSDERNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    jitter: float,
    loss_name: str,
    grad_clip: float | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total = {"loss": 0.0, "mse": 0.0, "nll": 0.0, "innovation_ratio": 0.0, "calibration": 0.0}
    n = 0
    for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
        z = batch["z"].to(device)
        r = batch["R"].to(device)
        times = batch["times"].to(device)
        if z.shape[0] < 2:
            continue
        out = model.forward_sequence(z, r, times, jitter=jitter, loss_name=loss_name)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
        steps = int(z.shape[0] - 1)
        total["loss"] += float(out.loss.detach().cpu()) * steps
        total["mse"] += float(out.mse.detach().cpu()) * steps
        total["nll"] += float(out.nll.detach().cpu()) * steps
        total["innovation_ratio"] += float(out.mean_innovation_ratio.detach().cpu()) * steps
        total["calibration"] += float(out.final_calibration.detach().cpu()) * steps
        n += steps
    return {k: v / max(n, 1) for k, v in total.items()} | {"steps": float(n)}


def save_checkpoint(path: Path, model: DraftNeuralSDERNN, cfg: dict, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": cfg,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_draft_sde_rnn.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    set_seed(int(cfg.get("seed", 2026)))
    device = select_device(cfg.get("device", "auto"))

    data_cfg = cfg["data"]
    dataset = LatentObservationSequenceDataset(
        resolve_path(data_cfg["h5"], root),
        min_length=int(data_cfg.get("min_length", 2)),
        covariance=str(data_cfg.get("covariance", "diag")),
        normalize_time=bool(data_cfg.get("normalize_time", True)),
    )
    if not len(dataset):
        raise ValueError("no sequences found")
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
        zero_init_deformation=bool(model_cfg.get("zero_init_deformation", True)),
    ).to(device)

    train_loader = make_loader(dataset, train_indices, shuffle=True)
    val_loader = make_loader(dataset, val_indices, shuffle=False)
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )

    loss_cfg = cfg["loss"]
    run_dir = resolve_path(cfg["output"]["run_dir"], root)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    val_metrics = {"loss": float("inf")}

    for epoch in range(1, int(optim_cfg["epochs"]) + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            jitter=float(loss_cfg["jitter"]),
            loss_name=str(loss_cfg["name"]),
            grad_clip=optim_cfg.get("grad_clip"),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device=device,
            jitter=float(loss_cfg["jitter"]),
            loss_name=str(loss_cfg["name"]),
            grad_clip=None,
        )
        log = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        print(json.dumps(log, indent=2))
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log) + "\n")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(run_dir / "best.pt", model, cfg, epoch, val_metrics)
        if epoch % int(cfg["output"].get("checkpoint_every", 10)) == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", model, cfg, epoch, val_metrics)

    save_checkpoint(run_dir / "last.pt", model, cfg, int(optim_cfg["epochs"]), val_metrics)


if __name__ == "__main__":
    main()
