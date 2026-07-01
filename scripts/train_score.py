#!/usr/bin/env python3
"""Train a score network on cine-MRI frames with the UNSURE AR-DAE objective."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_cunsure.checkpoint import save_score_checkpoint
from score_cunsure.data import FrameFolderDataset, TensorFrameDataset, VolumeFrameDataset
from score_cunsure.score_model import ScoreLossConfig, ScoreUNet, ardae_score_loss


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, nargs="+", help="One or more dataset roots/files")
    parser.add_argument("--val-data", nargs="*", default=None, help="Optional validation roots/files")
    parser.add_argument("--output", required=True, help="Path to save score checkpoint .pt")
    parser.add_argument("--best-output", default=None, help="Path to save best validation checkpoint")
    parser.add_argument("--metrics-csv", default=None, help="Path to write per-epoch metrics CSV")
    parser.add_argument("--resume", default=None, help="Resume from a previous checkpoint")
    parser.add_argument("--npz-key", default=None, help="Key to use for .npz/.pt dict inputs")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--spatial-dims", type=int, choices=[2, 3], default=2)
    parser.add_argument("--image-size", type=int, default=None, help="Optional square resize before training")
    parser.add_argument("--depth-size", type=int, default=None, help="Optional depth resize for 3D frames")
    parser.add_argument("--time-axis", type=int, default=-1, help="Time axis for 4D cine arrays when --spatial-dims 3")
    parser.add_argument(
        "--frame-layout",
        default="hwd",
        help="3D frame layout after time extraction: hwd, dhw, chwd, cdhw, hwdc, or auto",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.05, help="Validation fraction when --val-data is not set")
    parser.add_argument(
        "--preprocessed",
        action="store_true",
        help="Read cached tensor frames `[C,*spatial]` instead of raw cine volumes",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help='Optional filename/path patterns to include, e.g. "*_4d.nii.gz" "*_sa.nii.gz"',
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help='Optional filename/path patterns to exclude, e.g. "*_gt.nii.gz" "*_ED.nii.gz"',
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--delta-min", type=float, default=1.0e-3)
    parser.add_argument("--delta-max", type=float, default=1.0e-1)
    parser.add_argument("--augment", action="store_true", help="Enable light 3D augmentation during training")
    parser.add_argument("--flip-prob", type=float, default=0.5)
    parser.add_argument("--intensity-jitter", type=float, default=0.05)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--log-every", type=int, default=100, help="Print batch progress every N optimizer steps")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def maybe_resize(batch: torch.Tensor, image_size: int | None, depth_size: int | None, spatial_dims: int) -> torch.Tensor:
    if image_size is None:
        return batch
    if spatial_dims == 2:
        return F.interpolate(batch, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return F.interpolate(
        batch,
        size=(image_size, image_size, depth_size or image_size),
        mode="trilinear",
        align_corners=False,
    )


def collate_and_resize(
    samples: list[torch.Tensor],
    *,
    image_size: int | None,
    depth_size: int | None,
    spatial_dims: int,
) -> torch.Tensor:
    """Resize variable-size frames before stacking them into a batch."""
    if image_size is None:
        return torch.stack(samples, dim=0)
    resized = [
        maybe_resize(sample.unsqueeze(0), image_size=image_size, depth_size=depth_size, spatial_dims=spatial_dims).squeeze(0)
        for sample in samples
    ]
    return torch.stack(resized, dim=0)


def build_dataset(args: argparse.Namespace, roots: list[str]) -> Dataset[torch.Tensor]:
    datasets = []
    for data_root in roots:
        if args.preprocessed:
            dataset_i = TensorFrameDataset(
                data_root,
                channels=args.channels,
                npz_key=args.npz_key,
                spatial_dims=args.spatial_dims,
                limit=args.limit,
                include_patterns=args.include,
                exclude_patterns=args.exclude,
            )
        elif args.spatial_dims == 2:
            dataset_i = FrameFolderDataset(
                data_root,
                channels=args.channels,
                npz_key=args.npz_key,
                normalize=True,
                limit=args.limit,
                include_patterns=args.include,
                exclude_patterns=args.exclude,
            )
        else:
            dataset_i = VolumeFrameDataset(
                data_root,
                channels=args.channels,
                npz_key=args.npz_key,
                normalize=True,
                time_axis=args.time_axis,
                frame_layout=args.frame_layout,
                limit=args.limit,
                include_patterns=args.include,
                exclude_patterns=args.exclude,
            )
        print(f"loaded root={data_root} files={len(dataset_i.paths)} frames={len(dataset_i)}")
        datasets.append(dataset_i)
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def split_train_val(dataset: Dataset[torch.Tensor], val_fraction: float, seed: int) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor] | None]:
    if val_fraction <= 0:
        return dataset, None
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be in [0, 1)")
    val_len = max(1, int(round(len(dataset) * val_fraction)))
    train_len = len(dataset) - val_len
    if train_len < 1:
        raise ValueError("validation split leaves no training samples")
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    return train_set, val_set


def make_loader(args: argparse.Namespace, dataset: Dataset[torch.Tensor], *, shuffle: bool) -> DataLoader[torch.Tensor]:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.device == "cuda",
        collate_fn=lambda samples: collate_and_resize(
            samples,
            image_size=args.image_size,
            depth_size=args.depth_size,
            spatial_dims=args.spatial_dims,
        ),
    )


def augment_batch(batch: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if not args.augment:
        return batch
    for dim in range(2, batch.ndim):
        if torch.rand((), device=batch.device) < args.flip_prob:
            batch = torch.flip(batch, dims=(dim,))
    if args.intensity_jitter > 0:
        scale = 1.0 + args.intensity_jitter * torch.randn((batch.shape[0], 1, *([1] * args.spatial_dims)), device=batch.device)
        shift = args.intensity_jitter * torch.randn((batch.shape[0], 1, *([1] * args.spatial_dims)), device=batch.device)
        batch = (batch * scale + shift).clamp(0.0, 1.0)
    if args.input_noise_std > 0:
        batch = (batch + args.input_noise_std * torch.randn_like(batch)).clamp(0.0, 1.0)
    return batch


def run_epoch(
    model: ScoreUNet,
    loader: DataLoader[torch.Tensor],
    *,
    optimizer: torch.optim.Optimizer | None,
    loss_config: ScoreLossConfig,
    args: argparse.Namespace,
    device: torch.device,
    step: int,
    epoch: int,
) -> tuple[float, int, float]:
    training = optimizer is not None
    model.train(training)
    running = 0.0
    last_tau = math.nan
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=True)
            if training:
                batch = augment_batch(batch, args)
            loss, metrics = ardae_score_loss(model, batch, step=step, config=loss_config)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
            running += float(loss.detach())
            last_tau = float(metrics["tau"])
            if training and args.log_every > 0 and (batch_idx % args.log_every == 0 or batch_idx == len(loader)):
                print(
                    f"epoch={epoch}/{args.epochs} "
                    f"batch={batch_idx}/{len(loader)} "
                    f"loss={running / batch_idx:.6f} "
                    f"tau={last_tau:.6f}",
                    flush=True,
                )
    return running / max(len(loader), 1), step, last_tau


def append_metrics(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device}")

    dataset = build_dataset(args, args.data)
    if args.val_data:
        train_set = dataset
        val_set = build_dataset(args, args.val_data)
    else:
        train_set, val_set = split_train_val(dataset, args.val_fraction, args.seed)
    print(f"training frames={len(train_set)}")
    if val_set is not None:
        print(f"validation frames={len(val_set)}")
    train_loader = make_loader(args, train_set, shuffle=True)
    val_loader = make_loader(args, val_set, shuffle=False) if val_set is not None else None

    model_config = {
        "in_channels": args.channels,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "spatial_dims": args.spatial_dims,
    }
    model = ScoreUNet(**model_config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_config = ScoreLossConfig(
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        total_steps=max(args.epochs * len(train_loader), 1),
    )

    step = 0
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        print(f"resumed checkpoint={args.resume} start_epoch={start_epoch} step={step} best_val_loss={best_val_loss}")

    output = Path(args.output)
    best_output = Path(args.best_output) if args.best_output else output.with_name(f"{output.stem}.best{output.suffix}")
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else output.with_name(f"{output.stem}.metrics.csv")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, step, tau = run_epoch(
            model,
            train_loader,
            optimizer=opt,
            loss_config=loss_config,
            args=args,
            device=device,
            step=step,
            epoch=epoch,
        )
        val_loss = float("nan")
        if val_loader is not None:
            val_loss, _, _ = run_epoch(
                model,
                val_loader,
                optimizer=None,
                loss_config=loss_config,
                args=args,
                device=device,
                step=step,
                epoch=epoch,
            )
        print(f"epoch={epoch}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} tau={tau:.6f}", flush=True)

        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "tau": tau,
            "lr": args.lr,
        }
        append_metrics(metrics_csv, row)

        metrics = dict(row)
        save_score_checkpoint(
            output,
            model,
            model_config=model_config,
            loss_config=loss_config,
            step=step,
            optimizer=opt,
            epoch=epoch,
            best_val_loss=None if math.isinf(best_val_loss) else best_val_loss,
            metrics=metrics,
        )

        score_for_best = val_loss if val_loader is not None else train_loss
        if score_for_best < best_val_loss:
            best_val_loss = score_for_best
            save_score_checkpoint(
                best_output,
                model,
                model_config=model_config,
                loss_config=loss_config,
                step=step,
                optimizer=opt,
                epoch=epoch,
                best_val_loss=best_val_loss,
                metrics=metrics,
            )
            print(f"saved best checkpoint to {best_output} best_loss={best_val_loss:.6f}", flush=True)

    save_score_checkpoint(
        output,
        model,
        model_config=model_config,
        loss_config=loss_config,
        step=step,
        optimizer=opt,
        epoch=args.epochs,
        best_val_loss=None if math.isinf(best_val_loss) else best_val_loss,
    )
    print(f"saved final checkpoint to {args.output}")
    print(f"best checkpoint: {best_output}")
    print(f"metrics csv: {metrics_csv}")


if __name__ == "__main__":
    main()
