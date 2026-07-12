from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from cunsure_monai3d.preprocess import center_crop_or_pad, extract_frame_array, normalize_volume
from cunsure_monai3d.sde_data import build_sequence_refs, decode_h5_string, load_latent_covariance


def _nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _candidate_mask_paths(source_path: Path) -> list[Path]:
    parent = source_path.parent
    stem = _nifti_stem(source_path)
    patterns = [
        f"{stem}_gt.nii.gz",
        f"{stem}_gt.nii",
        f"{stem.replace('_4d', '')}_frame*_gt.nii.gz",
        f"{stem.replace('_4d', '')}_frame*_gt.nii",
        f"{stem.replace('_CINE', '_ED_gt')}.nii.gz",
        f"{stem.replace('_CINE', '_ES_gt')}.nii.gz",
        f"{stem.replace('_CINE', '_ED_gt')}.nii",
        f"{stem.replace('_CINE', '_ES_gt')}.nii",
        f"{stem.replace('_SA_CINE', '_SA_ED_gt')}.nii.gz",
        f"{stem.replace('_SA_CINE', '_SA_ES_gt')}.nii.gz",
        f"{stem.replace('_LA_CINE', '_LA_ED_gt')}.nii.gz",
        f"{stem.replace('_LA_CINE', '_LA_ES_gt')}.nii.gz",
        f"{stem.replace('_sa', '_sa_gt')}.nii.gz",
        f"{stem.replace('_sa', '_sa_gt')}.nii",
    ]
    seen: set[Path] = set()
    candidates: list[Path] = []
    for pattern in patterns:
        for path in sorted(parent.glob(pattern)):
            if path == source_path or path in seen:
                continue
            seen.add(path)
            candidates.append(path)
    return candidates


def _mask_to_dhw(mask: np.ndarray, *, time_axis: int, path: Path) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 4:
        axis = time_axis if time_axis >= 0 else mask.ndim + time_axis
        mask = np.max(mask, axis=axis)
    mask = np.squeeze(mask)
    if mask.ndim == 2:
        mask = mask[..., None]
    if mask.ndim != 3:
        raise ValueError(f"expected 3D mask after time union, got {mask.shape} from {path}")
    mask = np.moveaxis(mask, -1, 0)
    return np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0) > 0


def _bbox_from_mask(mask: np.ndarray, margin: tuple[int, int, int]) -> tuple[slice, slice, slice] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    starts = coords.min(axis=0)
    stops = coords.max(axis=0) + 1
    slices = []
    for dim, start, stop, pad in zip(mask.shape, starts, stops, margin, strict=True):
        lo = max(int(start) - int(pad), 0)
        hi = min(int(stop) + int(pad), int(dim))
        slices.append(slice(lo, hi))
    return tuple(slices)  # type: ignore[return-value]


def _load_mask_bbox(
    source_path: Path,
    *,
    time_axis: int,
    margin: tuple[int, int, int],
    enabled: bool,
    require_mask: bool,
) -> tuple[slice, slice, slice] | None:
    if not enabled:
        return None
    union_mask: np.ndarray | None = None
    candidates = _candidate_mask_paths(source_path)
    for mask_path in candidates:
        mask = _mask_to_dhw(nib.load(str(mask_path)).get_fdata(dtype=np.float32), time_axis=time_axis, path=mask_path)
        union_mask = mask if union_mask is None else np.logical_or(union_mask, mask)
    if union_mask is None:
        if require_mask:
            raise FileNotFoundError(f"no ED/ES segmentation mask found next to {source_path}")
        return None
    bbox = _bbox_from_mask(union_mask, margin)
    if bbox is None and require_mask:
        raise ValueError(f"empty ED/ES segmentation masks next to {source_path}")
    return bbox


def crop_or_pad_around_bbox(
    volume: np.ndarray,
    bbox: tuple[slice, slice, slice] | None,
    target_size: tuple[int, int, int],
) -> np.ndarray:
    if bbox is None:
        return center_crop_or_pad(volume, target_size)
    starts = np.array([s.start or 0 for s in bbox], dtype=np.int64)
    stops = np.array([s.stop or dim for s, dim in zip(bbox, volume.shape, strict=True)], dtype=np.int64)
    center = (starts + stops) // 2
    crop_slices = []
    pads: list[tuple[int, int]] = []
    for dim, target, c in zip(volume.shape, target_size, center, strict=True):
        lo = int(c) - int(target) // 2
        hi = lo + int(target)
        src_lo = max(lo, 0)
        src_hi = min(hi, int(dim))
        crop_slices.append(slice(src_lo, src_hi))
        pads.append((max(0, -lo), max(0, hi - int(dim))))
    out = volume[tuple(crop_slices)]
    if any(lo or hi for lo, hi in pads):
        out = np.pad(out, pads, mode="constant")
    return out.astype(np.float32, copy=False)


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
    roi_mask_crop: bool = False,
    roi_mask_margin: tuple[int, int, int] = (0, 12, 12),
    require_roi_mask: bool = False,
) -> torch.Tensor:
    path = _resolve_source_path(source_path, root, source_path_remap)
    arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    roi_bbox = _load_mask_bbox(
        path,
        time_axis=time_axis,
        margin=roi_mask_margin,
        enabled=roi_mask_crop,
        require_mask=require_roi_mask,
    )
    frames: list[np.ndarray] = []
    for time_index in time_indices:
        ref_time = None if int(time_index) < 0 else int(time_index)
        vol = extract_frame_array(arr, ref_time, time_axis=time_axis, path=path)
        vol = crop_or_pad_around_bbox(vol, roi_bbox, volume_size)
        vol = normalize_volume(
            vol,
            mode=normalize,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
        )
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
        cache_data: bool = False,
        roi_mask_crop: bool = False,
        roi_mask_margin: tuple[int, int, int] = (0, 12, 12),
        require_roi_mask: bool = False,
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
        self.roi_mask_crop = bool(roi_mask_crop)
        self.roi_mask_margin = tuple(int(v) for v in roi_mask_margin)
        self.require_roi_mask = bool(require_roi_mask)
        self.refs = build_sequence_refs(self.h5_path, min_length=min_length)
        self.cache_data = bool(cache_data)
        self._cache: list[dict[str, object]] | None = None
        if self.cache_data:
            self._cache = [self._load_item(index) for index in tqdm(range(len(self.refs)), desc="preload deformation data")]

    def __len__(self) -> int:
        return len(self.refs)

    def _load_item(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        indices = list(ref.indices)
        with h5py.File(self.h5_path, "r") as h5:
            z = torch.from_numpy(h5["z"][indices]).float()
            covariance = load_latent_covariance(h5, indices, mode=self.covariance)
            source_paths = [decode_h5_string(v) for v in h5["source_path"][indices]]
        if len(set(source_paths)) != 1:
            raise ValueError(f"sequence spans multiple source files: {set(source_paths)}")
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
            roi_mask_crop=self.roi_mask_crop,
            roi_mask_margin=self.roi_mask_margin,
            require_roi_mask=self.require_roi_mask,
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

    def __getitem__(self, index: int) -> dict[str, object]:
        if self._cache is not None:
            return self._cache[index]
        return self._load_item(index)
