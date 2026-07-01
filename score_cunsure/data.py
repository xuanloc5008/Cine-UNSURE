"""Lightweight dataset utilities for cine-MRI 2D slices and 3D time frames."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".npy", ".npz", ".pt", ".pth", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nii", ".nii.gz"}


def is_supported_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in IMAGE_EXTENSIONS)


def matches_any_pattern(path: Path, patterns: Sequence[str] | None) -> bool:
    """Return true if path basename or full path matches any shell-style pattern."""
    if not patterns:
        return False
    name = path.name.lower()
    full = path.as_posix().lower()
    return any(fnmatch.fnmatch(name, p.lower()) or fnmatch.fnmatch(full, p.lower()) for p in patterns)


def scan_supported_paths(
    root: str | Path,
    *,
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> list[Path]:
    """Scan supported image paths with optional include/exclude filters."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if root.is_file():
        paths = [root] if is_supported_path(root) else []
    else:
        paths = [p for p in root.rglob("*") if is_supported_path(p)]
    if include_patterns:
        paths = [p for p in paths if matches_any_pattern(p, include_patterns)]
    if exclude_patterns:
        paths = [p for p in paths if not matches_any_pattern(p, exclude_patterns)]
    return sorted(paths)


def minmax_normalize(x: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    low = x.amin(dim=tuple(range(1, x.ndim)), keepdim=True)
    high = x.amax(dim=tuple(range(1, x.ndim)), keepdim=True)
    return (x - low) / (high - low + eps)


def configure_sitk_warnings(sitk: object) -> None:
    """Keep noisy NIfTI header warnings out of training logs by default."""
    show_warnings = os.environ.get("SCORE_CUNSURE_SHOW_ITK_WARNINGS", "0").lower() in {"1", "true", "yes", "on"}
    try:
        sitk.ProcessObject_SetGlobalWarningDisplay(show_warnings)
    except AttributeError:
        pass


def reverse_axis_order(x: torch.Tensor) -> torch.Tensor:
    """Match SimpleITK's array convention for NIfTI volumes."""
    return x.permute(*range(x.ndim - 1, -1, -1)).contiguous()


def load_array(path: str | Path, *, npz_key: str | None = None) -> torch.Tensor:
    """Load an array-like medical image file as a float tensor."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".npy"):
        return torch.as_tensor(np.load(path)).float()
    if name.endswith(".npz"):
        data = np.load(path)
        key = npz_key or sorted(data.files)[0]
        return torch.as_tensor(data[key]).float()
    if name.endswith((".pt", ".pth")):
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            loaded = loaded[npz_key or sorted(loaded.keys())[0]]
        return torch.as_tensor(loaded).float()
    if name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        return torch.as_tensor(np.asarray(Image.open(path))).float()
    if name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib  # type: ignore

            data = np.asanyarray(nib.load(str(path)).dataobj)
            return reverse_axis_order(torch.as_tensor(data).float())
        except ImportError:
            pass
        try:
            import SimpleITK as sitk  # type: ignore
        except ImportError as exc:
            raise ImportError("nibabel or SimpleITK is required to read .nii/.nii.gz files") from exc
        configure_sitk_warnings(sitk)
        return torch.as_tensor(sitk.GetArrayFromImage(sitk.ReadImage(str(path)))).float()
    raise ValueError(f"unsupported image extension: {path.suffix}")


def load_array_shape(path: str | Path, *, npz_key: str | None = None) -> tuple[int, ...]:
    """Return array shape without loading full medical volumes when possible."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".npy"):
        return tuple(np.load(path, mmap_mode="r").shape)
    if name.endswith(".npz"):
        data = np.load(path)
        key = npz_key or sorted(data.files)[0]
        return tuple(data[key].shape)
    if name.endswith((".pt", ".pth")):
        return tuple(load_array(path, npz_key=npz_key).shape)
    if name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        with Image.open(path) as image:
            width, height = image.size
            return (height, width, len(image.getbands())) if len(image.getbands()) > 1 else (height, width)
    if name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib  # type: ignore

            return tuple(reversed(nib.load(str(path)).shape))
        except ImportError:
            pass
        try:
            import SimpleITK as sitk  # type: ignore
        except ImportError as exc:
            raise ImportError("nibabel or SimpleITK is required to read .nii/.nii.gz files") from exc
        configure_sitk_warnings(sitk)
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        return tuple(reversed(reader.GetSize()))
    raise ValueError(f"unsupported image extension: {path.suffix}")


def apply_frame_layout(x: torch.Tensor, layout: str = "auto") -> torch.Tensor:
    """Convert common frame layouts before channel normalization.

    Supported 2D layouts: `hw`, `chw`, `hwc`.
    Supported 3D layouts: `hwd`, `dhw`, `chwd`, `cdhw`, `hwdc`.
    `auto` leaves the tensor unchanged and lets `ensure_channels_first` infer.
    """
    layout = layout.lower()
    if layout in {"auto", "none"}:
        return x
    if layout in {"hw", "hwd", "chw", "chwd"}:
        return x
    if layout == "hwc":
        return x.permute(2, 0, 1)
    if layout == "dhw":
        return x.permute(1, 2, 0)
    if layout == "cdhw":
        return x.permute(0, 2, 3, 1)
    if layout == "hwdc":
        return x.permute(3, 0, 1, 2)
    raise ValueError(f"unsupported frame layout: {layout}")


def extract_time_frame(x: torch.Tensor, frame_index: int | None, time_axis: int | None) -> torch.Tensor:
    """Extract one 3D frame from a 4D or 5D cine sequence."""
    if frame_index is None or time_axis is None:
        return x
    axis = time_axis if time_axis >= 0 else x.ndim + time_axis
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"time_axis {time_axis} is invalid for shape {tuple(x.shape)}")
    if frame_index < 0 or frame_index >= x.shape[axis]:
        raise IndexError(f"frame_index {frame_index} is invalid for time length {x.shape[axis]}")
    return x.select(dim=axis, index=frame_index)


def ensure_channels_first(x: torch.Tensor, *, channels: int = 1, spatial_dims: int = 2) -> torch.Tensor:
    """Convert a loaded frame to `[C, *spatial]`."""
    if spatial_dims not in {2, 3}:
        raise ValueError("spatial_dims must be 2 or 3")
    if x.ndim == spatial_dims:
        x = x.unsqueeze(0)
    elif x.ndim == spatial_dims + 1:
        # Prefer channel-last only when the last axis looks like a channel axis.
        if x.shape[-1] in {1, 3, 4} and x.shape[0] not in {1, 3, 4}:
            x = x.movedim(-1, 0)
    else:
        raise ValueError(f"expected {spatial_dims}D frame with optional channel, got shape {tuple(x.shape)}")

    if x.shape[0] == channels:
        return x
    if channels == 3 and x.shape[0] == 1:
        return x.repeat(3, *([1] * spatial_dims))
    if channels == 1 and x.shape[0] > 1:
        return x[:1]
    raise ValueError(f"cannot convert {tuple(x.shape)} to {channels} channels")


def load_frame(
    path: str | Path,
    *,
    npz_key: str | None = None,
    channels: int = 1,
    normalize: bool = True,
    spatial_dims: int = 2,
    frame_index: int | None = None,
    time_axis: int | None = None,
    frame_layout: str = "auto",
) -> torch.Tensor:
    """Load one 2D slice `[C,H,W]` or 3D cine frame `[C,H,W,D]`."""
    tensor = load_array(path, npz_key=npz_key)
    tensor = extract_time_frame(tensor, frame_index=frame_index, time_axis=time_axis)
    tensor = apply_frame_layout(tensor, frame_layout)
    tensor = ensure_channels_first(tensor, channels=channels, spatial_dims=spatial_dims)
    if normalize:
        tensor = minmax_normalize(tensor.unsqueeze(0)).squeeze(0)
    return tensor


class FrameFolderDataset(Dataset[torch.Tensor]):
    """Dataset over image/array files, returning normalized `[C, H, W]` frames."""

    def __init__(
        self,
        root: str | Path,
        *,
        channels: int = 1,
        npz_key: str | None = None,
        normalize: bool = True,
        limit: int | None = None,
        include_patterns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        paths = scan_supported_paths(self.root, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        self.paths: Sequence[Path] = sorted(paths)
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"no supported frame files found under {self.root}")
        self.channels = channels
        self.npz_key = npz_key
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return load_frame(self.paths[idx], npz_key=self.npz_key, channels=self.channels, normalize=self.normalize)


class VolumeFrameDataset(Dataset[torch.Tensor]):
    """Dataset over 3D frames from one or more 4D cine-MRI files.

    Each item is `[C, H, W, D]`. For arrays shaped `[H, W, D, T]`, use the
    defaults `time_axis=-1, frame_layout=hwd`. NIfTI files are normalized to
    `[T, D, H, W]`, so use `time_axis=0, frame_layout=dhw`.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        channels: int = 1,
        npz_key: str | None = None,
        normalize: bool = True,
        time_axis: int | None = -1,
        frame_layout: str = "hwd",
        limit: int | None = None,
        include_patterns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = None,
    ) -> None:
        self.root = Path(root)
        paths = scan_supported_paths(self.root, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        self.paths: Sequence[Path] = sorted(paths)
        if not self.paths:
            raise FileNotFoundError(f"no supported volume files found under {self.root}")

        self.channels = channels
        self.npz_key = npz_key
        self.normalize = normalize
        self.time_axis = time_axis
        self.frame_layout = frame_layout

        index: list[tuple[Path, int | None]] = []
        for path in self.paths:
            shape = load_array_shape(path, npz_key=npz_key)
            if time_axis is None or len(shape) <= 3:
                index.append((path, None))
            else:
                axis = time_axis if time_axis >= 0 else len(shape) + time_axis
                if axis < 0 or axis >= len(shape):
                    raise ValueError(f"time_axis {time_axis} is invalid for {path} shape {shape}")
                index.extend((path, t) for t in range(shape[axis]))
        self.index: Sequence[tuple[Path, int | None]] = index[:limit] if limit is not None else index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path, frame_index = self.index[idx]
        return load_frame(
            path,
            npz_key=self.npz_key,
            channels=self.channels,
            normalize=self.normalize,
            spatial_dims=3,
            frame_index=frame_index,
            time_axis=self.time_axis if frame_index is not None else None,
            frame_layout=self.frame_layout,
        )
