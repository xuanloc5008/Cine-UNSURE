#!/usr/bin/env python3
"""Train a score network on cine-MRI frames with the UNSURE AR-DAE objective."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

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
    parser.add_argument("--output", required=True, help="Path to save score checkpoint .pt")
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
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--delta-min", type=float, default=1.0e-3)
    parser.add_argument("--delta-max", type=float, default=1.0e-1)
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device}")

    datasets = []
    for data_root in args.data:
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

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    print(f"total training frames={len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=lambda samples: collate_and_resize(
            samples,
            image_size=args.image_size,
            depth_size=args.depth_size,
            spatial_dims=args.spatial_dims,
        ),
    )

    model_config = {
        "in_channels": args.channels,
        "base_channels": args.base_channels,
        "depth": args.depth,
        "spatial_dims": args.spatial_dims,
    }
    model = ScoreUNet(**model_config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_config = ScoreLossConfig(
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        total_steps=max(args.epochs * len(loader), 1),
    )

    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            loss, metrics = ardae_score_loss(model, batch, step=step, config=loss_config)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            step += 1
            if args.log_every > 0 and (batch_idx % args.log_every == 0 or batch_idx == len(loader)):
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"batch={batch_idx}/{len(loader)} "
                    f"loss={running / batch_idx:.6f} "
                    f"tau={float(metrics['tau']):.6f}",
                    flush=True,
                )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={running / max(len(loader), 1):.6f} "
            f"tau={float(metrics['tau']):.6f}"
        )

    save_score_checkpoint(
        Path(args.output),
        model,
        model_config=model_config,
        loss_config=loss_config,
        step=step,
    )
    print(f"saved score checkpoint to {args.output}")


if __name__ == "__main__":
    main()
