#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nodeo_uq.config import load_yaml, project_root, resolve_path, select_device
from cardiac_nodeo_uq.nodeo_dir import GaussianKernel3D
from cardiac_nodeo_uq.nodeo_ops import (
    GradientOrientationLoss3D,
    MultiScaleLocalNCC3D,
    SpatialTransformer3D,
    compose_displacements,
    identity_grid_voxel,
    nodeo_jacobian_metrics,
    smoothness_loss,
)


class ResidualSVF(nn.Module):
    def __init__(
        self,
        *,
        frames: int,
        image_shape: tuple[int, int, int],
        coarse_scale: float,
        smoothing_window: int,
        smoothing_sigma: float,
        integration_steps: int,
    ) -> None:
        super().__init__()
        coarse_shape = tuple(max(2, int(round(value * coarse_scale))) for value in image_shape)
        self.image_shape = image_shape
        self.integration_steps = int(integration_steps)
        self.velocity = nn.Parameter(torch.zeros(frames - 1, 3, *coarse_shape))
        self.smoother = GaussianKernel3D(smoothing_window, smoothing_sigma)

    def full_velocity(self) -> Tensor:
        velocity = F.interpolate(
            self.velocity,
            size=self.image_shape,
            mode="trilinear",
            align_corners=True,
        )
        return self.smoother(velocity)

    def exponentiate(self, velocity: Tensor) -> Tensor:
        displacement = velocity / float(2**self.integration_steps)
        transformer = SpatialTransformer3D(self.image_shape).to(velocity.device)
        for _ in range(self.integration_steps):
            displacement = compose_displacements(
                displacement,
                displacement,
                transformer=transformer,
            )
        return displacement

    def forward(self) -> tuple[Tensor, Tensor, Tensor]:
        velocity = self.full_velocity()
        return self.exponentiate(velocity), self.exponentiate(-velocity), velocity


def temporal_smoothness(value: Tensor) -> Tensor:
    if value.shape[0] < 2:
        return value.new_zeros(())
    velocity = value[1:] - value[:-1]
    first_order = velocity.square().mean()
    if velocity.shape[0] < 2:
        return first_order
    acceleration = velocity[1:] - velocity[:-1]
    return first_order + acceleration.square().mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acdc/experiments/06_residual_refinement.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    seed = int(cfg.get("seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(args.device or cfg.get("device", "auto"))

    input_path = resolve_path(args.input, root)
    output_path = resolve_path(args.output, root)
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    images = payload["images"].float().to(device)
    base_displacement = payload["displacement"].float().to(device)
    base_inverse = payload["inverse_displacement"].float().to(device)
    image_shape = tuple(int(value) for value in images.shape[-3:])
    frames = int(images.shape[0])

    model_cfg = cfg["model"]
    residual = ResidualSVF(
        frames=frames,
        image_shape=image_shape,
        coarse_scale=float(model_cfg.get("coarse_scale", 0.5)),
        smoothing_window=int(model_cfg.get("smoothing_window", 7)),
        smoothing_sigma=float(model_cfg.get("smoothing_sigma", 1.25)),
        integration_steps=int(model_cfg.get("integration_steps", 5)),
    ).to(device)
    transformer = SpatialTransformer3D(image_shape).to(device)
    loss_cfg = cfg["loss"]
    similarity = MultiScaleLocalNCC3D(
        scales=tuple(float(value) for value in loss_cfg["ncc_scales"]),
        windows=tuple(int(value) for value in loss_cfg["ncc_windows"]),
        weights=tuple(float(value) for value in loss_cfg["ncc_weights"]),
    ).to(device)
    gradient_loss = GradientOrientationLoss3D().to(device)
    optimizer = torch.optim.Adam(
        residual.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"].get("weight_decay", 0.0)),
    )

    target = images[1:]
    reference = images[0:1].expand(frames - 1, -1, -1, -1, -1)
    identity = identity_grid_voxel(
        image_shape,
        device=device,
        dtype=images.dtype,
    ).expand(frames - 1, -1, -1, -1, -1)
    with torch.no_grad():
        (
            _,
            _,
            _,
            base_fold,
            _,
            base_jac_min,
            base_jac_max,
        ) = nodeo_jacobian_metrics(
            identity + base_displacement[1:],
            minimum=float(loss_cfg["minimum_jacobian"]),
            maximum=float(loss_cfg["maximum_jacobian"]),
        )
    selection_cfg = cfg.get("selection", {})
    maximum_fold = max(
        float(selection_cfg.get("max_fold_fraction", 0.0)),
        float(base_fold),
    )
    minimum_jacobian = float(base_jac_min) * float(
        selection_cfg.get("min_jacobian_ratio_to_base", 0.90)
    )
    maximum_jacobian = float(base_jac_max) * float(
        selection_cfg.get("max_jacobian_ratio_to_base", 1.10)
    )
    best_loss = float("inf")
    best_iteration = 0
    best_velocity: Tensor | None = None
    best_metrics: dict[str, float] | None = None

    for iteration in range(1, int(cfg["optim"]["iterations"]) + 1):
        residual_forward, residual_inverse, velocity = residual()
        total_forward = compose_displacements(
            base_displacement[1:],
            residual_forward,
            transformer=transformer,
        )
        total_inverse = compose_displacements(
            residual_inverse,
            base_inverse[1:],
            transformer=transformer,
        )
        warped = transformer(reference, total_forward)
        backward_warped = transformer(target, total_inverse)
        image_forward = similarity(target, warped)
        image_backward = similarity(reference, backward_warped)
        image_loss = 0.5 * (image_forward + image_backward)
        structural = 0.5 * (
            gradient_loss(target, warped)
            + gradient_loss(reference, backward_warped)
        )
        jdet, _, _, fold, _, jac_min, jac_max = nodeo_jacobian_metrics(
            identity + total_forward,
            minimum=float(loss_cfg["minimum_jacobian"]),
            maximum=float(loss_cfg["maximum_jacobian"]),
        )
        residual_magnitude = velocity.square().mean()
        residual_spatial = smoothness_loss(residual_forward)
        residual_temporal = temporal_smoothness(residual_forward)
        cycle = total_forward[-1].square().mean()
        loss = (
            image_loss
            + float(loss_cfg.get("lambda_gradient", 0.0)) * structural
            + float(loss_cfg["lambda_j"]) * jdet
            + float(loss_cfg["lambda_magnitude"]) * residual_magnitude
            + float(loss_cfg["lambda_spatial"]) * residual_spatial
            + float(loss_cfg["lambda_temporal"]) * residual_temporal
            + float(loss_cfg["lambda_cycle"]) * cycle
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        metrics = {
            "loss": float(loss.detach()),
            "image": float(image_loss.detach()),
            "gradient": float(structural.detach()),
            "jdet": float(jdet.detach()),
            "fold_fraction": float(fold.detach()),
            "jacobian_min": float(jac_min.detach()),
            "jacobian_max": float(jac_max.detach()),
            "residual_magnitude": float(residual_magnitude.detach()),
            "residual_spatial": float(residual_spatial.detach()),
            "residual_temporal": float(residual_temporal.detach()),
            "cycle": float(cycle.detach()),
        }
        topology_valid = (
            metrics["fold_fraction"] <= maximum_fold + 1.0e-8
            and metrics["jacobian_min"] >= minimum_jacobian
            and metrics["jacobian_max"] <= maximum_jacobian
        )
        metrics["topology_valid"] = float(topology_valid)
        if topology_valid and metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_iteration = iteration
            best_velocity = residual.velocity.detach().cpu().clone()
            best_metrics = metrics
        optimizer.step()
        if iteration == 1 or iteration % int(cfg["optim"].get("log_every", 25)) == 0:
            print(json.dumps({"iteration": iteration, **metrics}))

    if best_velocity is None or best_metrics is None:
        raise RuntimeError(
            "residual refinement produced no topology-valid state; "
            "keep the original NODEO result"
        )
    residual.velocity.data.copy_(best_velocity.to(device))
    residual.eval()
    with torch.no_grad():
        residual_forward, residual_inverse, velocity = residual()
        refined_forward = compose_displacements(
            base_displacement[1:], residual_forward, transformer=transformer
        )
        refined_inverse = compose_displacements(
            residual_inverse, base_inverse[1:], transformer=transformer
        )
        warped = transformer(reference, refined_forward)
        backward_warped = transformer(target, refined_inverse)

    zero = torch.zeros_like(base_displacement[0:1])
    refined_displacement = torch.cat((zero, refined_forward), dim=0)
    refined_inverse_displacement = torch.cat((zero, refined_inverse), dim=0)
    identity_full = identity_grid_voxel(
        image_shape, device=device, dtype=images.dtype
    ).expand(frames, -1, -1, -1, -1)
    result = dict(payload)
    result.update(
        {
            "base_displacement": payload["displacement"],
            "base_inverse_displacement": payload["inverse_displacement"],
            "base_warped": payload["warped"],
            "displacement": refined_displacement.cpu().half(),
            "inverse_displacement": refined_inverse_displacement.cpu().half(),
            "phi_bar": (identity_full + refined_displacement).cpu().half(),
            "inverse_phi_bar": (identity_full + refined_inverse_displacement).cpu().half(),
            "warped": warped.cpu().half(),
            "backward_warped": backward_warped.cpu().half(),
            "residual_velocity": torch.cat((zero, velocity), dim=0).cpu().half(),
            "residual_displacement": torch.cat((zero, residual_forward), dim=0).cpu().half(),
            "residual_refinement": {
                "best_iteration": best_iteration,
                "metrics": best_metrics,
                "config": cfg,
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "best_iteration": best_iteration,
                "metrics": best_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
