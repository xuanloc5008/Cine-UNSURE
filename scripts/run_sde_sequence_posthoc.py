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
from torch.nn import functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nodeo_uq.config import load_yaml, project_root, resolve_path, select_device
from cardiac_nodeo_uq.nodeo_ops import SpatialTransformer3D
from cardiac_nodeo_uq.nodeo_roi_data import canonical_source_key
from cardiac_nodeo_uq.sde_sequence_posthoc import (
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


def gaussian_smooth_ambiguity(
    values: Tensor,
    *,
    kernel_size: int,
    sigma: float,
) -> Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("ambiguity kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("ambiguity sigma must be positive")
    radius = kernel_size // 2
    coordinates = torch.arange(kernel_size, device=values.device, dtype=values.dtype) - radius
    kernel_1d = torch.exp(-0.5 * (coordinates / float(sigma)).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.einsum("i,j,k->ijk", kernel_1d, kernel_1d, kernel_1d)
    kernel = kernel[None, None]
    padded = F.pad(values[:, None], (radius,) * 6, mode="replicate")
    return F.conv3d(padded, kernel)[:, 0]


def residual_ambiguity(
    images: Tensor,
    predicted_frames: Tensor,
    *,
    mode: str,
    kernel_size: int,
    sigma: float,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    residual_squared = (images - predicted_frames).square().mean(dim=1)
    if mode == "gaussian":
        variance_proxy = gaussian_smooth_ambiguity(
            residual_squared,
            kernel_size=kernel_size,
            sigma=sigma,
        )
    elif mode == "none":
        variance_proxy = residual_squared
    else:
        raise ValueError("ambiguity mode must be 'none' or 'gaussian'")
    variance_proxy = variance_proxy + float(epsilon)
    denominator = variance_proxy.flatten(1).amax(dim=1).clamp_min(float(epsilon))
    ambiguity = variance_proxy / denominator[:, None, None, None]
    # I0 is mapped to itself and is treated as a deterministic initial state.
    ambiguity[0].zero_()
    return residual_squared, ambiguity.clamp(0.0, 1.0)


def fit_sequence(
    *,
    nodeo_payload: dict[str, object],
    cfg: dict[str, object],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, object], dict[str, float | int | str]]:
    started = time.monotonic()
    model_cfg = dict(cfg["model"])  # type: ignore[arg-type]
    fit_cfg = dict(cfg["fit"])  # type: ignore[arg-type]
    covariance_cfg = dict(cfg["covariance"])  # type: ignore[arg-type]
    ambiguity_cfg = dict(cfg["ambiguity"])  # type: ignore[arg-type]

    if "images" not in nodeo_payload:
        raise KeyError("NODEO output must contain the cropped ROI image sequence under 'images'")
    images = nodeo_payload["images"].float().to(device)  # type: ignore[union-attr]
    times = nodeo_payload["times"].float().to(device)  # type: ignore[union-attr]
    phi_bar = nodeo_payload["phi_bar"].float().to(device)  # type: ignore[union-attr]
    nodeo_displacement = nodeo_payload.get("displacement")
    if nodeo_displacement is None:
        nodeo_displacement = phi_bar - phi_bar[0:1]
    nodeo_displacement = nodeo_displacement.float().to(device)  # type: ignore[union-attr]

    lengths = {len(times), len(images), len(phi_bar), len(nodeo_displacement)}
    if len(lengths) != 1:
        raise ValueError(
            "NODEO sequence tensors have different lengths: "
            f"times={len(times)}, images={len(images)}, "
            f"phi_bar={len(phi_bar)}, displacement={len(nodeo_displacement)}"
        )
    length = len(times)
    if length < 2:
        raise ValueError("a cine sequence must contain at least two frames")
    images = images[:length]
    times = times[:length]
    phi_bar = phi_bar[:length]
    nodeo_displacement = nodeo_displacement[:length]
    if tuple(images.shape[-3:]) != tuple(nodeo_displacement.shape[-3:]):
        raise ValueError(
            f"image/NODEO shape mismatch: {tuple(images.shape[-3:])} vs "
            f"{tuple(nodeo_displacement.shape[-3:])}"
        )

    transformer = SpatialTransformer3D(tuple(int(v) for v in images.shape[-3:])).to(device)
    with torch.no_grad():
        nodeo_predicted_frames = transformer(
            images[0:1].expand(length, -1, -1, -1, -1),
            nodeo_displacement,
        )
        residual_squared, ambiguity_map = residual_ambiguity(
            images,
            nodeo_predicted_frames,
            mode=str(ambiguity_cfg.get("mode", "gaussian")),
            kernel_size=int(ambiguity_cfg.get("kernel_size", 5)),
            sigma=float(ambiguity_cfg.get("sigma", 1.0)),
            epsilon=float(ambiguity_cfg.get("epsilon", 1.0e-8)),
        )

    motion_mean, motion_basis, motion_code = fit_linear_basis(
        nodeo_displacement.reshape(length, -1), int(model_cfg["motion_rank"])
    )
    # U_ambiguity is used directly as an isotropic voxel-space covariance
    # proxy, then projected into the same low-rank basis as NODEO motion.
    ambiguity_diagonal = ambiguity_map[:, None].expand(-1, 3, -1, -1, -1).reshape(length, -1)
    ambiguity_diagonal = ambiguity_diagonal * float(ambiguity_cfg.get("variance_scale", 1.0))
    observation_covariance = project_observation_covariance(
        ambiguity_diagonal,
        motion_basis,
    )
    cvgru_input, motion_scale, observation_covariance = normalize_codes(
        motion_code, observation_covariance
    )
    assert observation_covariance is not None
    motion_code = cvgru_input

    model = PerSequencePostHocSDERNN(
        observation_dim=int(cvgru_input.shape[1]),
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
        output = model.forward_mean(times=times, observation=cvgru_input, mask=observed)
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
                observation=cvgru_input,
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
        observation=cvgru_input,
        observation_covariance=observation_covariance,
        covariance_floor=float(covariance_cfg["floor"]),
        shrinkage=float(covariance_cfg["process_shrinkage"]),
    ).detach()
    analytical = model.propagate_analytical(
        times=times,
        observation=cvgru_input,
        observation_covariance=observation_covariance,
        process_covariance=process_covariance,
        init_covariance=float(covariance_cfg["init_covariance"]),
        covariance_floor=float(covariance_cfg["floor"]),
    )

    predicted_motion_code = analytical.motion_code * motion_scale
    motion_factor = analytical.motion_covariance_factor * motion_scale[None, :, None]
    ambiguity_motion_factor = (
        analytical.ambiguity_motion_covariance_factor * motion_scale[None, :, None]
    )
    process_motion_factor = (
        analytical.process_motion_covariance_factor * motion_scale[None, :, None]
    )
    sde_displacement_flat = motion_mean[None] + predicted_motion_code @ motion_basis.T
    # The deformation is defined relative to I0, so frame zero is deterministically identity.
    sde_displacement_flat = sde_displacement_flat - sde_displacement_flat[0:1]
    motion_factor = motion_factor.clone()
    ambiguity_motion_factor = ambiguity_motion_factor.clone()
    process_motion_factor = process_motion_factor.clone()
    motion_factor[0].zero_()
    ambiguity_motion_factor[0].zero_()
    process_motion_factor[0].zero_()
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

    def voxel_variance_from_factor(factors: Tensor) -> Tensor:
        values: list[Tensor] = []
        for frame_factor in factors:
            full_factor = motion_basis @ frame_factor
            values.append(full_factor.square().sum(dim=1).reshape_as(nodeo_displacement[0]))
        return torch.stack(values)

    ambiguity_voxel_variance = voxel_variance_from_factor(ambiguity_motion_factor)
    process_voxel_variance = voxel_variance_from_factor(process_motion_factor)
    # Report the decomposition exactly. Separate PSD stabilization of the two
    # covariance streams can otherwise introduce a small diagonal mismatch.
    voxel_variance = ambiguity_voxel_variance + process_voxel_variance

    payload: dict[str, object] = {
        "method": "nodeo_residual_ambiguity_posthoc_sde_cvgru",
        "mean_source": "nodeo_locked",
        "dataset": nodeo_payload["dataset"],
        "source_path": nodeo_payload["source_path"],
        "raw_time_indices": nodeo_payload["raw_time_indices"],
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
        "residual_squared": residual_squared.detach().cpu().float(),
        "ambiguity_map": ambiguity_map.detach().cpu().float(),
        "ambiguity_definition": (
            "U_ambiguity = normalize_frame(G_sigma * "
            "(I_t - warp(I_0, phi_NODEO_t))^2); U_ambiguity[0] = 0"
        ),
        "deformation_variance_diag": voxel_variance.detach().cpu().float(),
        "ambiguity_deformation_variance_diag": ambiguity_voxel_variance.detach().cpu().float(),
        "process_deformation_variance_diag": process_voxel_variance.detach().cpu().float(),
        "motion_basis": motion_basis.detach().cpu().float(),
        "motion_covariance_factor": motion_factor.detach().cpu().float(),
        "ambiguity_motion_covariance_factor": ambiguity_motion_factor.detach().cpu().float(),
        "process_motion_covariance_factor": process_motion_factor.detach().cpu().float(),
        "hidden_mean": analytical.hidden_mean.detach().cpu().float(),
        "hidden_covariance": analytical.hidden_covariance.detach().cpu().float(),
        "hidden_ambiguity_covariance": analytical.hidden_ambiguity_covariance.detach().cpu().float(),
        "hidden_process_covariance": analytical.hidden_process_covariance.detach().cpu().float(),
        "process_covariance": process_covariance.detach().cpu().float(),
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
            "U_ambiguity enters analytical covariance directly, and the final "
            "mean trajectory remains locked to NODEO"
        ),
    }
    metrics: dict[str, float | int | str] = {
        "best_iteration": best_iteration,
        "train_mse_last": last_train_loss,
        "held_out_mse": best_validation,
        "runtime_seconds": time.monotonic() - started,
        "frames": length,
        "motion_rank": int(motion_basis.shape[1]),
        "cvgru_input_rank": int(motion_basis.shape[1]),
        "mean_source": "nodeo_locked",
        "pca_reconstruction_mse": float(pca_reconstruction_mse),
        "sde_reconstruction_mse": float(sde_reconstruction_mse),
        "sde_relative_displacement_error": float(sde_relative_displacement_error),
        "nodeo_frame_mse": float(nodeo_frame_mse),
        "sde_reconstructed_frame_mse": float(sde_frame_mse),
        "nodeo_peak_motion_index": nodeo_peak_index,
        "sde_peak_motion_index": sde_peak_index,
        "peak_motion_index_error": abs(sde_peak_index - nodeo_peak_index),
        "mean_deformation_variance": float(voxel_variance.mean()),
        "mean_ambiguity_deformation_variance": float(ambiguity_voxel_variance.mean()),
        "mean_process_deformation_variance": float(process_voxel_variance.mean()),
        "mean_ambiguity": float(ambiguity_map.mean()),
    }
    return payload, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acdc/run_sde_sequence_posthoc.yaml")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--nodeo-summary")
    parser.add_argument("--output-root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    seed = int(cfg.get("seed", 2026))
    set_seed(seed)
    device = select_device(cfg.get("device", "auto"))
    configure_torch(device)
    summary_value = args.nodeo_summary or configured_split(cfg["nodeo"]["summaries"], args.split)
    summary_path = resolve_path(summary_value, root)
    assert summary_path is not None
    rows = load_summary(summary_path)
    stop = len(rows) if args.limit is None else min(len(rows), args.start_index + args.limit)
    output_root = resolve_path(args.output_root or cfg["output"]["run_dir"], root)
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
        nodeo_path = resolve_nodeo_output(row, summary_path, root)
        nodeo_payload = torch.load(nodeo_path, map_location="cpu", weights_only=False)
        payload, metrics = fit_sequence(
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
            "dataset": nodeo_payload["dataset"],
            "source_path": nodeo_payload["source_path"],
            "output": str(output_path),
            **metrics,
        }
        with summary_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary_row) + "\n")
        print(json.dumps(summary_row, indent=2))


if __name__ == "__main__":
    main()
