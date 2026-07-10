from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from cunsure_monai3d.preprocess import center_crop_or_pad, extract_frame_array, normalize_volume
from cunsure_monai3d.sde_data import build_sequence_refs, decode_h5_string


def _remap_source_path(path: str, root: Path, remaps: list[dict[str, str]]) -> Path | None:
    for remap in remaps:
        old_prefix = str(remap.get("from", "")).rstrip("/")
        new_prefix = str(remap.get("to", "")).rstrip("/")
        if not old_prefix or not new_prefix:
            continue
        if not path.startswith(old_prefix):
            continue
        suffix = path[len(old_prefix) :].lstrip("/")
        base = Path(new_prefix)
        candidate = base / suffix if base.is_absolute() else root / base / suffix
        if candidate.exists():
            return candidate
    return None


def _resolve_source_path(path: str, root: Path, remaps: list[dict[str, str]] | None = None) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if not p.is_absolute():
        candidate = root / p
        if candidate.exists():
            return candidate
    remapped = _remap_source_path(path, root, list(remaps or []))
    if remapped is not None:
        return remapped
    raise FileNotFoundError(f"source NIfTI not found: {path}")


def load_sequence_images(
    source_path: str,
    time_indices: list[int],
    *,
    root: Path,
    time_axis: int,
    volume_size: tuple[int, int, int],
    normalize: str,
    percentile_low: float,
    percentile_high: float,
    output_size: tuple[int, int, int] | None,
    source_path_remap: list[dict[str, str]] | None = None,
) -> torch.Tensor:
    path = _resolve_source_path(source_path, root, source_path_remap)
    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    frames: list[np.ndarray] = []
    for time_index in time_indices:
        ref_time = None if int(time_index) < 0 else int(time_index)
        vol = extract_frame_array(arr, ref_time, time_axis=time_axis, path=path)
        vol = normalize_volume(
            vol,
            mode=normalize,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
        vol = center_crop_or_pad(vol, volume_size)
        frames.append(vol)
    images = torch.from_numpy(np.stack(frames)).float()[:, None]
    if output_size is not None and tuple(images.shape[-3:]) != tuple(output_size):
        images = F.interpolate(images, size=tuple(output_size), mode="trilinear", align_corners=True)
    return images.contiguous()


class DeformationSequenceDataset(Dataset[dict[str, object]]):
    """Sequence dataset with real images, CineMA latent observations, and covariance."""

    def __init__(
        self,
        h5_path: str | Path,
        *,
        root: str | Path,
        min_length: int = 2,
        covariance: Literal["full", "diag"] = "diag",
        normalize_time: bool = True,
        time_axis: int = -1,
        volume_size: tuple[int, int, int] = (16, 128, 128),
        image_size: tuple[int, int, int] | None = None,
        normalize: str = "percentile",
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        source_path_remap: list[dict[str, str]] | None = None,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        if covariance not in {"full", "diag"}:
            raise ValueError("covariance must be 'full' or 'diag'")
        self.root = Path(root)
        self.covariance = covariance
        self.normalize_time = normalize_time
        self.time_axis = int(time_axis)
        self.volume_size = tuple(int(v) for v in volume_size)
        self.image_size = None if image_size is None else tuple(int(v) for v in image_size)
        self.normalize = str(normalize)
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)
        self.source_path_remap = list(source_path_remap or [])
        self.refs = build_sequence_refs(self.h5_path, min_length=min_length)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        indices = list(ref.indices)
        with h5py.File(self.h5_path, "r") as h5:
            z = torch.from_numpy(h5["z"][indices]).float()
            covariance = torch.from_numpy(h5["latent_covariance_psd"][indices]).float()
            source_paths = [decode_h5_string(v) for v in h5["source_path"][indices]]
        if len(set(source_paths)) != 1:
            raise ValueError(f"sequence spans multiple source files: {set(source_paths)}")
        if self.covariance == "diag":
            covariance = covariance.diagonal(dim1=-2, dim2=-1)

        raw_times = torch.tensor(ref.time_indices, dtype=torch.float32)
        if self.normalize_time:
            if len(raw_times) > 1 and raw_times.max() > raw_times.min():
                times = (raw_times - raw_times.min()) / (raw_times.max() - raw_times.min())
            elif len(raw_times) > 1:
                times = torch.linspace(0.0, 1.0, len(raw_times))
            else:
                times = torch.zeros_like(raw_times)
        else:
            times = raw_times

        images = load_sequence_images(
            ref.source_path,
            list(ref.time_indices),
            root=self.root,
            time_axis=self.time_axis,
            volume_size=self.volume_size,
            normalize=self.normalize,
            percentile_low=self.percentile_low,
            percentile_high=self.percentile_high,
            output_size=self.image_size,
            source_path_remap=self.source_path_remap,
        )
        return {
            "images": images,
            "z": z,
            "R": covariance,
            "times": times,
            "raw_time_indices": raw_times.long(),
            "dataset": ref.dataset,
            "source_path": ref.source_path,
            "indices": torch.tensor(indices, dtype=torch.long),
        }
