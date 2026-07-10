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
from cunsure_monai3d.sde_data import LatentObservationSequenceDataset
from cunsure_monai3d.sde_models import LatentEulerDrift, gaussian_nll_diag, mse_loss


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
        raise ValueError("train_latent_sde_observation expects batch_size=1 for variable-length sequences")
    return batch[0]


def make_loader(dataset: LatentObservationSequenceDataset, indices: list[int] | None, *, shuffle: bool) -> DataLoader:
    subset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(subset, batch_size=1, shuffle=shuffle, num_workers=0, collate_fn=sequence_collate)


def run_epoch(
    model: LatentEulerDrift,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    loss_name: str,
    jitter: float,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total = {"loss": 0.0, "mse": 0.0, "nll": 0.0}
    n_steps = 0

    for batch in tqdm(loader, desc="train" if train else "val", leave=False):
        z = batch["z"].to(device)
        r = batch["R"].to(device)
        times = batch["times"].to(device)
        if z.shape[0] < 2:
            continue

        pred = model(z, times)
        target = z[1:]
        variance = r[1:] if r.ndim == 2 else r[1:].diagonal(dim1=-2, dim2=-1)
        nll = gaussian_nll_diag(pred, target, variance, jitter=jitter)
        mse = mse_loss(pred, target)
        loss = nll if loss_name == "nll_diag" else mse

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        steps = int(target.shape[0])
        total["loss"] += float(loss.detach().cpu()) * steps
        total["mse"] += float(mse.detach().cpu()) * steps
        total["nll"] += float(nll.detach().cpu()) * steps
        n_steps += steps

    return {k: v / max(n_steps, 1) for k, v in total.items()} | {"steps": float(n_steps)}


def save_checkpoint(path: Path, model: LatentEulerDrift, cfg: dict, epoch: int, metrics: dict[str, float]) -> None:
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
    parser.add_argument("--config", default="configs/train_latent_sde_observation.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    set_seed(int(cfg.get("seed", 2026)))
    device = select_device(cfg.get("device", "auto"))

    data_cfg = cfg["data"]
    h5_path = resolve_path(data_cfg["h5"], root)
    dataset = LatentObservationSequenceDataset(
        h5_path,
        min_length=int(data_cfg.get("min_length", 2)),
        covariance=str(data_cfg.get("covariance", "diag")),
        normalize_time=bool(data_cfg.get("normalize_time", True)),
    )
    if not len(dataset):
        raise ValueError("no latent observation sequences found")
    latent_dim = int(dataset[0]["z"].shape[-1])

    index_path = resolve_path(data_cfg.get("sequence_index"), root)
    train_indices = load_split_indices(index_path, "train")
    val_indices = load_split_indices(index_path, "val")
    if train_indices is None:
        n = len(dataset)
        order = list(range(n))
        random.Random(int(cfg.get("seed", 2026))).shuffle(order)
        val_count = max(1, int(0.1 * n))
        val_indices = order[:val_count]
        train_indices = order[val_count:]

    train_loader = make_loader(dataset, train_indices, shuffle=True)
    val_loader = make_loader(dataset, val_indices, shuffle=False)

    model_cfg = cfg["model"]
    model = LatentEulerDrift(
        latent_dim=latent_dim,
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        time_invariant=bool(model_cfg.get("time_invariant", False)),
    ).to(device)

    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
    loss_name = str(cfg["loss"].get("name", "nll_diag"))
    jitter = float(cfg["loss"].get("jitter", 1.0e-6))
    run_dir = resolve_path(cfg["output"]["run_dir"], root)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, int(optim_cfg["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device=device, loss_name=loss_name, jitter=jitter)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device=device, loss_name=loss_name, jitter=jitter)
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
