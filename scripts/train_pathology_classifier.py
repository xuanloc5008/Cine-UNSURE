#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nodeo_uq.config import load_yaml, project_root, resolve_path, select_device
from cardiac_nodeo_uq.pathology import (
    ACDC_PATHOLOGIES,
    PathologyClassifier,
    clinical_feature_names,
    load_clinical_samples,
)


def paths_from_glob(pattern: str, root: Path) -> list[Path]:
    path = Path(pattern)
    query = str(path if path.is_absolute() else root / path)
    return [Path(value) for value in sorted(glob.glob(query))]


def macro_f1(target: np.ndarray, predicted: np.ndarray, num_classes: int) -> float:
    scores = []
    for class_index in range(num_classes):
        true_positive = np.sum((target == class_index) & (predicted == class_index))
        false_positive = np.sum((target != class_index) & (predicted == class_index))
        false_negative = np.sum((target == class_index) & (predicted != class_index))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    targets = []
    predictions = []
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        losses.append(float(criterion(logits, labels)) * labels.numel())
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    target = np.asarray(targets)
    predicted = np.asarray(predictions)
    return {
        "loss": float(sum(losses) / len(targets)),
        "accuracy": float(np.mean(target == predicted)),
        "macro_f1": macro_f1(target, predicted, len(ACDC_PATHOLOGIES)),
        "samples": float(len(targets)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acdc/train_pathology_classifier.yaml")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(resolve_path(args.config, root))
    seed = int(cfg.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = select_device(cfg.get("device", "auto"))

    data_cfg = cfg["data"]
    time_points = int(data_cfg.get("time_points", 20))
    train_samples = load_clinical_samples(
        paths_from_glob(str(data_cfg["train_glob"]), root), root=root, time_points=time_points
    )
    val_samples = load_clinical_samples(
        paths_from_glob(str(data_cfg["val_glob"]), root), root=root, time_points=time_points
    )
    class_to_index = {name: index for index, name in enumerate(ACDC_PATHOLOGIES)}

    train_x_raw = np.stack([sample.features for sample in train_samples])
    val_x_raw = np.stack([sample.features for sample in val_samples])
    feature_mean = train_x_raw.mean(axis=0)
    feature_std = train_x_raw.std(axis=0)
    feature_std = np.where(feature_std < 1.0e-6, 1.0, feature_std)
    train_x = torch.from_numpy(((train_x_raw - feature_mean) / feature_std).astype(np.float32))
    val_x = torch.from_numpy(((val_x_raw - feature_mean) / feature_std).astype(np.float32))
    train_y = torch.tensor([class_to_index[sample.pathology] for sample in train_samples], dtype=torch.long)
    val_y = torch.tensor([class_to_index[sample.pathology] for sample in val_samples], dtype=torch.long)

    counts = torch.bincount(train_y, minlength=len(ACDC_PATHOLOGIES)).float()
    if torch.any(counts == 0):
        missing = [ACDC_PATHOLOGIES[i] for i, count in enumerate(counts) if count == 0]
        raise ValueError(f"training clinical predictions contain no samples for classes: {missing}")
    class_weights = (len(train_y) / (len(ACDC_PATHOLOGIES) * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model_cfg = cfg["model"]
    model = PathologyClassifier(
        input_dim=train_x.shape[1],
        hidden_dims=tuple(int(value) for value in model_cfg["hidden_dims"]),
        num_classes=len(ACDC_PATHOLOGIES),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(optim_cfg["lr"]), weight_decay=float(optim_cfg.get("weight_decay", 0.0))
    )
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=min(32, len(train_y)), shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=min(64, len(val_y)), shuffle=False)

    run_dir = resolve_path(cfg["output"]["run_dir"], root)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    best_loss = float("inf")
    stale_epochs = 0
    patience = int(optim_cfg.get("early_stopping_patience", 25))
    min_delta = float(optim_cfg.get("early_stopping_min_delta", 0.0))

    for epoch in range(1, int(optim_cfg["epochs"]) + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * labels.numel()
            train_count += labels.numel()
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss / train_count, "val": val_metrics}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row, indent=2))

        if val_metrics["loss"] < best_loss - min_delta:
            best_loss = val_metrics["loss"]
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "classes": list(ACDC_PATHOLOGIES),
                    "time_points": time_points,
                    "feature_names": clinical_feature_names(time_points),
                    "feature_mean": torch.from_numpy(feature_mean.astype(np.float32)),
                    "feature_std": torch.from_numpy(feature_std.astype(np.float32)),
                    "model": dict(model_cfg),
                    "config": cfg,
                    "val_metrics": val_metrics,
                },
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(json.dumps({"early_stopping_epoch": epoch, "best_val_loss": best_loss}, indent=2))
                break


if __name__ == "__main__":
    main()
