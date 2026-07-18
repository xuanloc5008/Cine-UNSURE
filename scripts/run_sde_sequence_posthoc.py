#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import Tensor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import load_yaml, project_root, resolve_path, select_device
from cunsure_monai3d.nodeo_ops import SpatialTransformer3D
from cunsure_monai3d.nodeo_roi_data import canonical_source_key
from cunsure_monai3d.sde_data import LatentObservationSequenceDataset
from cunsure_monai3d.sde_sequence_posthoc import (
    PerSequencePostHocSDERNN,
    fit_linear_basis,
    project_observation_covariance,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def configured_split(value: object, split: str) -> str:
    if isinstance(value, dict):
        if split not in value:
            raise KeyError(f"configuration has no value for split={split}")
        return str(value[split])
    return str(value)


def load_summary(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_nodeo_output(row: dict[str, object], summary_path: Path, root: Path) -> Path:
    output = Path(str(row["output"]))
    candidates = [output]
    if not output.is_absolute():
        candidates.append(root / output)
    candidates.append(summary_path.parent / output.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"NODEO output not found for {row['source_path']}: {output}")


def sequence_id(source_path: str) -> str:
    return hashlib.sha1(portable_source_key(source_path).encode("utf-8")).hexdigest()[:16]


def portable_source_key(path: str) -> str:
    normalized = path.replace("\\", "/")
    for marker in ("/ACDC/", "/M&M1/", "/MnM2/"):
        if marker in normalized:
            return marker.strip("/") + "/" + normalized.split(marker, 1)[1]
    return canonical_source_key(normalized)


def validation_mask(length: int, fraction: float, device: torch.device) -> Tensor:
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    if length <= 2 or fraction <= 0:
        mask[-1] = length > 1
        return mask
    count = min(max(int(round((length - 1) * fraction)), 1), length - 1)
    positions = torch.linspace(1, length - 1, count + 2, device=device)[1:-1].round().long().unique()
    mask[positions] = True
    return mask


def random_target_mask(*, candidates: Tensor, fraction: float, generator: torch.Generator) -> Tensor:
    indices = torch.nonzero(candidates, as_tuple=False).flatten().cpu()
    mask = torch.zeros_like(candidates)
    if len(indices) == 0:
        return mask
    count = min(max(int(round(len(indices) * fraction)), 1), len(indices))
    selected = indices[torch.randperm(len(indices), generator=generator)[:count]].to(mask.device)
    mask[selected] = True
    return mask


def normalize_codes(codes: Tensor, covariance: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
    scale = codes.std(dim=0, unbiased=False).clamp_min(1.0e-4)
    normalized = codes / scale
    if covariance is None:
        return normalized, scale, None
    normalized_covariance = covariance / (scale[None, :, None] * scale[None, None, :])
    return normalized, scale, normalized_covariance


def fit_sequence(
    *,
    sample: dict[str, object],
    nodeo_payload: dict[str, object],
    cfg: dict[str, object],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, object], dict[str, float | int | str]]:
    started = time.monotonic()
    model_cfg = dict(cfg["model"])  # type: ignore[arg-type]
    fit_cfg = dict(cfg["fit"])  # type: ignore[arg-type]
    covariance_cfg = dict(cfg["covariance"])  # type: ignore[arg-type]

    if "images" not in nodeo_payload:
        raise KeyError("NODEO output must contain the cropped ROI image sequence under 'images'")
    images = nodeo_payload["images"].float().to(device)  # type: ignore[union-attr]
    times = sample["times"].float().to(device)  # type: ignore[union-attr]
    z = sample["z"].float().to(device)  # type: ignore[union-attr]
    latent_covariance = sample["R"].float().to(device)  # type: ignore[union-attr]
    phi_bar = nodeo_payload["phi_bar"].float().to(device)  # type: ignore[union-attr]
    nodeo_displacement = nodeo_payload.get("displacement")
    if nodeo_displacement is None:
        nodeo_displacement = phi_bar - phi_bar[0:1]
    nodeo_displacement = nodeo_displacement.float().to(device)  # type: ignore[union-attr]

    lengths = {len(times), len(z), len(images), len(phi_bar), len(nodeo_displacement)}
    if len(lengths) != 1:
        raise ValueError(
            "CineMA/C-UNSURE and NODEO sequence lengths differ: "
            f"times={len(times)}, z={len(z)}, images={len(images)}, "
            f"phi_bar={len(phi_bar)}, displacement={len(nodeo_displacement)}"
        )
    length = len(times)
    if length < 2:
        raise ValueError("a cine sequence must contain at least two frames")
    nodeo_raw_times = nodeo_payload.get("raw_time_indices")
    if nodeo_raw_times is not None:
        latent_raw_times = sample["raw_time_indices"].long()  # type: ignore[union-attr]
        nodeo_raw_times = nodeo_raw_times.long()  # type: ignore[union-attr]
        if not torch.equal(latent_raw_times.cpu(), nodeo_raw_times.cpu()):
            raise ValueError(
                "CineMA/C-UNSURE and NODEO frame indices differ: "
                f"latent={latent_raw_times.tolist()}, NODEO={nodeo_raw_times.tolist()}"
            )
    images = images[:length]
    times = times[:length]
    z = z[:length]
    latent_covariance = latent_covariance[:length]
    phi_bar = phi_bar[:length]
    nodeo_displacement = nodeo_displacement[:length]
    if tuple(images.shape[-3:]) != tuple(nodeo_displacement.shape[-3:]):
        raise ValueError(
            f"image/NODEO shape mismatch: {tuple(images.shape[-3:])} vs "
            f"{tuple(nodeo_displacement.shape[-3:])}"
        )

    motion_mean, motion_basis, motion_code = fit_linear_basis(
        nodeo_displacement.reshape(length, -1), int(model_cfg["motion_rank"])
    )
    observation_mean, observation_basis, observation_code = fit_linear_basis(
        z, int(model_cfg["observation_rank"])
    )
    observation_covariance = project_observation_covariance(latent_covariance, observation_basis)
    motion_code, motion_scale, _ = normalize_codes(motion_code)
    observation_code, observation_scale, observation_covariance = normalize_codes(
        observation_code, observation_covariance
    )
    assert observation_covariance is not None

    model = PerSequencePostHocSDERNN(
        observation_dim=int(observation_code.shape[1]),
        motion_dim=int(motion_code.shape[1]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        mlp_hidden_dim=int(model_cfg["mlp_hidden_dim"]),
        mlp_layers=int(model_cfg["mlp_layers"]),
        integration_steps=int(model_cfg["integration_steps"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(fit_cfg["lr"]),
        weight_decay=float(fit_cfg["weight_decay"]),
    )
    held_out = validation_mask(length, float(fit_cfg["validation_fraction"]), device)
    fit_candidates = ~held_out
    fit_candidates[0] = False
    validation_observation_mask = ~held_out
    validation_observation_mask[0] = True
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state: dict[str, Tensor] | None = None
    best_iteration = 0
    best_validation = float("inf")
    stale_checks = 0
    last_train_loss = float("nan")

    for iteration in range(1, int(fit_cfg["iterations"]) + 1):
        model.train()
        target_mask = random_target_mask(
            candidates=fit_candidates,
            fraction=float(fit_cfg["mask_fraction"]),
            generator=generator,
        )
        observed = ~(target_mask | held_out)
        observed[0] = True
        output = model.forward_mean(times=times, observation=observation_code, mask=observed)
        masked_mse = torch.nn.functional.mse_loss(output.motion_code[target_mask], motion_code[target_mask])
        reference_mse = torch.nn.functional.mse_loss(output.motion_code[0], motion_code[0])
        loss = masked_mse + reference_mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(fit_cfg["grad_clip"]))
        optimizer.step()
        last_train_loss = float(loss.detach())

        if iteration % int(fit_cfg["validate_every"]) != 0 and iteration != int(fit_cfg["iterations"]):
            continue
        model.eval()
        with torch.no_grad():
            validation_output = model.forward_mean(
                times=times,
                observation=observation_code,
                mask=validation_observation_mask,
            )
            validation_loss = float(
                torch.nn.functional.mse_loss(
                    validation_output.motion_code[held_out], motion_code[held_out]
                )
            )
        if validation_loss < best_validation - float(fit_cfg["min_delta"]):
            best_validation = validation_loss
            best_iteration = iteration
            best_state = copy.deepcopy(model.state_dict())
            stale_checks = 0
        else:
            stale_checks += 1
        if stale_checks >= int(fit_cfg["patience"]):
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_iteration = iteration
        best_validation = validation_loss
    model.load_state_dict(best_state, strict=True)
    model.eval()

    process_covariance = model.estimate_process_covariance(
        times=times,
        observation=observation_code,
        observation_covariance=observation_covariance,
        covariance_floor=float(covariance_cfg["floor"]),
        shrinkage=float(covariance_cfg["process_shrinkage"]),
    ).detach()
    analytical = model.propagate_analytical(
        times=times,
        observation=observation_code,
        observation_covariance=observation_covariance,
        process_covariance=process_covariance,
        init_covariance=float(covariance_cfg["init_covariance"]),
        covariance_floor=float(covariance_cfg["floor"]),
    )

    predicted_motion_code = analytical.motion_code * motion_scale
    motion_factor = analytical.motion_covariance_factor * motion_scale[None, :, None]
    sde_displacement_flat = motion_mean[None] + predicted_motion_code @ motion_basis.T
    # The deformation is defined relative to I0, so frame zero is deterministically identity.
    sde_displacement_flat = sde_displacement_flat - sde_displacement_flat[0:1]
    motion_factor = motion_factor.clone()
    motion_factor[0].zero_()
    sde_reconstructed_displacement = sde_displacement_flat.reshape_as(nodeo_displacement)
    identity = phi_bar - nodeo_displacement
    sde_reconstructed_deformation = identity + sde_reconstructed_displacement

    # NODEO defines the mean deformation trajectory. The fitted SDE-CVGRU is
    # retained only for analytical Jacobian/covariance propagation; replacing
    # the NODEO mean with its low-rank point reconstruction can introduce a
    # systematic motion-amplitude and phase bias.
    mean_displacement = nodeo_displacement
    mean_deformation = phi_bar
    transformer = SpatialTransformer3D(tuple(int(v) for v in images.shape[-3:])).to(device)
    with torch.no_grad():
        predicted_frames = transformer(images[0:1].expand(length, -1, -1, -1, -1), mean_displacement)
        sde_reconstructed_frames = transformer(
            images[0:1].expand(length, -1, -1, -1, -1),
            sde_reconstructed_displacement,
        )

        target_motion_code = motion_code * motion_scale
        pca_displacement_flat = motion_mean[None] + target_motion_code @ motion_basis.T
        pca_displacement_flat = pca_displacement_flat - pca_displacement_flat[0:1]
        nodeo_flat = nodeo_displacement.reshape(length, -1)
        displacement_denominator = nodeo_flat[1:].norm().clamp_min(1.0e-8)
        pca_reconstruction_mse = torch.nn.functional.mse_loss(pca_displacement_flat, nodeo_flat)
        sde_reconstruction_mse = torch.nn.functional.mse_loss(sde_displacement_flat, nodeo_flat)
        sde_relative_displacement_error = (
            (sde_displacement_flat[1:] - nodeo_flat[1:]).norm() / displacement_denominator
        )
        nodeo_frame_mse = torch.nn.functional.mse_loss(predicted_frames, images)
        sde_frame_mse = torch.nn.functional.mse_loss(sde_reconstructed_frames, images)
        nodeo_peak_index = int(nodeo_flat.square().mean(dim=1).argmax())
        sde_peak_index = int(sde_displacement_flat.square().mean(dim=1).argmax())

    voxel_variance: list[Tensor] = []
    for frame_factor in motion_factor:
        full_factor = motion_basis @ frame_factor
        voxel_variance.append(full_factor.square().sum(dim=1).reshape_as(nodeo_displacement[0]))

    payload: dict[str, object] = {
        "method": "nodeo_mean_posthoc_analytical_sde_cvgru",
        "mean_source": "nodeo_locked",
        "dataset": sample["dataset"],
        "source_path": sample["source_path"],
        "raw_time_indices": sample["raw_time_indices"],
        "times": times.detach().cpu(),
        "images": images.detach().cpu().half(),
        "nodeo_mean_deformation": phi_bar.detach().cpu().half(),
        "nodeo_displacement": nodeo_displacement.detach().cpu().half(),
        "mean_deformation": mean_deformation.detach().cpu().half(),
        "mean_displacement": mean_displacement.detach().cpu().half(),
        "total_displacement": mean_displacement.detach().cpu().half(),
        "predicted_frames": predicted_frames.detach().cpu().half(),
        "sde_reconstructed_deformation": sde_reconstructed_deformation.detach().cpu().half(),
        "sde_reconstructed_displacement": sde_reconstructed_displacement.detach().cpu().half(),
        "deformation_variance_diag": torch.stack(voxel_variance).detach().cpu().float(),
        "motion_basis": motion_basis.detach().cpu().float(),
        "motion_covariance_factor": motion_factor.detach().cpu().float(),
        "hidden_mean": analytical.hidden_mean.detach().cpu().float(),
        "hidden_covariance": analytical.hidden_covariance.detach().cpu().float(),
        "process_covariance": process_covariance.detach().cpu().float(),
        "observation_mean": observation_mean.detach().cpu().float(),
        "observation_basis": observation_basis.detach().cpu().float(),
        "observation_scale": observation_scale.detach().cpu().float(),
        "motion_mean": motion_mean.detach().cpu().float(),
        "motion_scale": motion_scale.detach().cpu().float(),
        "model_state": {key: value.detach().cpu() for key, value in best_state.items()},
        "config": cfg,
        "best_iteration": best_iteration,
        "validation_frame_indices": torch.nonzero(held_out).flatten().cpu(),
        "covariance_definition": (
            "L_phi[k] = motion_basis @ motion_covariance_factor[k]; "
            "R_phi[k] = L_phi[k] @ L_phi[k].T"
        ),
        "uncertainty_training": (
            "post-hoc only; SDE-CVGRU weights optimized with masked-frame MSE, "
            "while the final mean trajectory remains locked to NODEO"
        ),
    }
    metrics: dict[str, float | int | str] = {
        "best_iteration": best_iteration,
        "train_mse_last": last_train_loss,
        "held_out_mse": best_validation,
        "runtime_seconds": time.monotonic() - started,
        "frames": length,
        "motion_rank": int(motion_basis.shape[1]),
        "observation_rank": int(observation_basis.shape[1]),
        "mean_source": "nodeo_locked",
        "pca_reconstruction_mse": float(pca_reconstruction_mse),
        "sde_reconstruction_mse": float(sde_reconstruction_mse),
        "sde_relative_displacement_error": float(sde_relative_displacement_error),
        "nodeo_frame_mse": float(nodeo_frame_mse),
        "sde_reconstructed_frame_mse": float(sde_frame_mse),
        "nodeo_peak_motion_index": nodeo_peak_index,
        "sde_peak_motion_index": sde_peak_index,
        "peak_motion_index_error": abs(sde_peak_index - nodeo_peak_index),
        "mean_deformation_variance": float(torch.stack(voxel_variance).mean()),
    }
    return payload, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acdc/run_sde_sequence_posthoc.yaml")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    seed = int(cfg.get("seed", 2026))
    set_seed(seed)
    device = select_device(cfg.get("device", "auto"))
    configure_torch(device)
    data_cfg = cfg["data"]
    h5_path = resolve_path(configured_split(data_cfg["h5"], args.split), root)
    summary_path = resolve_path(configured_split(cfg["nodeo"]["summaries"], args.split), root)
    assert h5_path is not None and summary_path is not None
    dataset = LatentObservationSequenceDataset(
        h5_path,
        min_length=int(data_cfg.get("min_length", 2)),
        covariance=str(data_cfg.get("covariance", "diag")),
        normalize_time=True,
    )
    dataset_index = {portable_source_key(ref.source_path): index for index, ref in enumerate(dataset.refs)}
    rows = load_summary(summary_path)
    stop = len(rows) if args.limit is None else min(len(rows), args.start_index + args.limit)
    output_root = resolve_path(cfg["output"]["run_dir"], root)
    assert output_root is not None
    output_dir = output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_output = output_dir / "summary.jsonl"
    completed: set[str] = set()
    if summary_output.exists() and not args.overwrite:
        with summary_output.open("r", encoding="utf-8") as handle:
            completed = {str(json.loads(line)["sequence_id"]) for line in handle if line.strip()}

    for row_index in tqdm(range(args.start_index, stop), desc=f"post-hoc SDE-CVGRU {args.split}"):
        row = rows[row_index]
        source_path = str(row["source_path"])
        sid = sequence_id(source_path)
        output_path = output_dir / f"{row_index:06d}_{sid}.pt"
        if sid in completed and output_path.exists() and not args.overwrite:
            continue
        key = portable_source_key(source_path)
        if key not in dataset_index:
            raise KeyError(f"latent H5 has no sequence matching NODEO source: {source_path}")
        sample = dataset[dataset_index[key]]
        nodeo_path = resolve_nodeo_output(row, summary_path, root)
        nodeo_payload = torch.load(nodeo_path, map_location="cpu", weights_only=False)
        payload, metrics = fit_sequence(
            sample=sample,
            nodeo_payload=nodeo_payload,
            cfg=cfg,
            device=device,
            seed=seed + row_index,
        )
        payload["sequence_id"] = sid
        payload["split"] = args.split
        payload["nodeo_output"] = str(nodeo_path)
        torch.save(payload, output_path)
        summary_row = {
            "sequence_id": sid,
            "split": args.split,
            "dataset": sample["dataset"],
            "source_path": sample["source_path"],
            "output": str(output_path),
            **metrics,
        }
        with summary_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary_row) + "\n")
        print(json.dumps(summary_row, indent=2))


if __name__ == "__main__":
    main()
