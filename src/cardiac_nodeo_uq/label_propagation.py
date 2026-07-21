from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from cardiac_nodeo_uq.nodeo_ops import SpatialTransformer3D


def _inplane_gaussian_kernel(
    *, kernel_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    coordinates = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (coordinates / float(sigma)).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.view(1, 1, 1, kernel_size, kernel_size).expand(channels, 1, -1, -1, -1)


@torch.no_grad()
def propagate_label_probabilities(
    reference_label: Tensor,
    displacement: Tensor,
    *,
    num_classes: int = 4,
    sigma_inplane: float = 1.0,
    kernel_size: int = 5,
) -> tuple[Tensor, Tensor]:
    """Warp a hard label as one-hot probabilities and smooth it in-plane.

    Returns ``(hard_label, probabilities)`` with shapes ``[T,1,D,H,W]`` and
    ``[T,C,D,H,W]``. Smoothing is not applied through-plane because short-axis
    cine MRI commonly has much larger slice spacing than in-plane spacing.
    """
    if reference_label.shape[0] != 1 or reference_label.shape[1] != 1:
        raise ValueError(f"reference_label must be [1,1,D,H,W], got {tuple(reference_label.shape)}")
    if displacement.ndim != 5 or displacement.shape[1] != 3:
        raise ValueError(f"displacement must be [T,3,D,H,W], got {tuple(displacement.shape)}")
    spatial_shape = tuple(int(v) for v in reference_label.shape[-3:])
    if tuple(displacement.shape[-3:]) != spatial_shape:
        raise ValueError("reference label and displacement spatial shapes differ")

    labels = reference_label[:, 0].long().clamp(0, int(num_classes) - 1)
    one_hot = F.one_hot(labels, num_classes=int(num_classes)).permute(0, 4, 1, 2, 3).float()
    source = one_hot.expand(displacement.shape[0], -1, -1, -1, -1)
    transformer = SpatialTransformer3D(spatial_shape, mode="bilinear").to(displacement.device)
    probabilities = transformer(source.to(displacement), displacement)

    if sigma_inplane > 0:
        kernel = _inplane_gaussian_kernel(
            kernel_size=int(kernel_size),
            sigma=float(sigma_inplane),
            channels=int(num_classes),
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        probabilities = F.conv3d(
            probabilities,
            kernel,
            padding=(0, int(kernel_size) // 2, int(kernel_size) // 2),
            groups=int(num_classes),
        )

    probabilities = probabilities.clamp_min(0.0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    hard_label = probabilities.argmax(dim=1, keepdim=True)
    return hard_label, probabilities
