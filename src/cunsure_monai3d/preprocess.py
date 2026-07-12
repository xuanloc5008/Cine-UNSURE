from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from itertools import groupby
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class FrameRef:
    path: Path
    time_index: int | None = None


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def candidate_mask_paths(source_path: Path) -> list[Path]:
    parent = source_path.parent
    stem = nifti_stem(source_path)
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


def mask_to_dhw(mask: np.ndarray, *, time_axis: int, path: Path) -> np.ndarray:
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


def bbox_from_mask(mask: np.ndarray, margin: tuple[int, int, int]) -> tuple[slice, slice, slice] | None:
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


def load_mask_bbox(
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
    for mask_path in candidate_mask_paths(source_path):
        mask = mask_to_dhw(nib.load(str(mask_path)).get_fdata(dtype=np.float32), time_axis=time_axis, path=mask_path)
        union_mask = mask if union_mask is None else np.logical_or(union_mask, mask)
    if union_mask is None:
        if require_mask:
            raise FileNotFoundError(f"no ED/ES segmentation mask found next to {source_path}")
        return None
    bbox = bbox_from_mask(union_mask, margin)
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


def scan_nifti_frames(
    root: Path,
    globs: list[str],
    exclude_substrings: list[str],
    *,
    split_4d: bool,
    time_axis: int,
) -> list[FrameRef]:
    refs: list[FrameRef] = []
    for pattern in globs:
        pattern_path = Path(pattern)
        search_pattern = str(pattern_path if pattern_path.is_absolute() else root / pattern)
        for path_str in sorted(glob(search_pattern, recursive=True)):
            path = Path(path_str)
            name = path.name.lower()
            if any(token.lower() in name for token in exclude_substrings):
                continue
            img = nib.load(str(path))
            shape = img.shape
            if len(shape) == 4 and split_4d:
                axis = time_axis if time_axis >= 0 else len(shape) + time_axis
                n_time = shape[axis]
                refs.extend(FrameRef(path=path, time_index=i) for i in range(n_time))
            else:
                refs.append(FrameRef(path=path, time_index=None))
    return refs


def load_frame(ref: FrameRef, *, time_axis: int) -> np.ndarray:
    arr = np.asarray(nib.load(str(ref.path)).get_fdata(dtype=np.float32))
    return extract_frame_array(arr, ref.time_index, time_axis=time_axis, path=ref.path)


def extract_frame_array(arr: np.ndarray, time_index: int | None, *, time_axis: int, path: Path) -> np.ndarray:
    if time_index is not None:
        axis = time_axis if time_axis >= 0 else arr.ndim + time_axis
        arr = np.take(arr, time_index, axis=axis)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"expected 3D frame after time extraction, got {arr.shape} from {path}")
    arr = np.moveaxis(arr, -1, 0)  # [D,H,W]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def normalize_volume(
    volume: np.ndarray,
    *,
    mode: str,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    if mode == "none":
        return volume.astype(np.float32, copy=False)
    if mode == "zscore":
        mean = float(volume.mean())
        std = float(volume.std())
        return ((volume - mean) / max(std, 1.0e-6)).astype(np.float32)
    if mode == "percentile":
        lo, hi = np.percentile(volume, [percentile_low, percentile_high])
        volume = np.clip(volume, lo, hi)
        return ((volume - lo) / max(float(hi - lo), 1.0e-6)).astype(np.float32)
    raise ValueError(f"unsupported normalize mode: {mode}")


def center_crop_or_pad(volume: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    out = volume
    pads: list[tuple[int, int]] = []
    for dim, target in zip(out.shape, target_size, strict=True):
        total = max(target - dim, 0)
        pads.append((total // 2, total - total // 2))
    if any(lo or hi for lo, hi in pads):
        out = np.pad(out, pads, mode="constant")

    slices = []
    for dim, target in zip(out.shape, target_size, strict=True):
        start = max((dim - target) // 2, 0)
        slices.append(slice(start, start + target))
    return out[tuple(slices)].astype(np.float32, copy=False)


def write_hdf5(
    refs: list[FrameRef],
    output_path: Path,
    *,
    volume_size: tuple[int, int, int],
    channels: int,
    normalize: str,
    percentile_low: float,
    percentile_high: float,
    time_axis: int,
    limit: int | None,
    compression: str | None = "lzf",
    roi_mask_crop: bool = False,
    roi_mask_margin: tuple[int, int, int] = (0, 12, 12),
    require_roi_mask: bool = False,
) -> None:
    if limit is not None:
        refs = refs[:limit]
    if not refs:
        raise ValueError(f"no NIfTI frames found for {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        compression_kwargs = {}
        if compression and compression != "none":
            compression_kwargs["compression"] = compression
            if compression == "gzip":
                compression_kwargs["compression_opts"] = 1
        y = h5.create_dataset(
            "y",
            shape=(len(refs), channels, *volume_size),
            dtype="float32",
            chunks=(1, channels, *volume_size),
            **compression_kwargs,
        )
        paths = h5.create_dataset("source_path", shape=(len(refs),), dtype=h5py.string_dtype())
        times = h5.create_dataset("time_index", shape=(len(refs),), dtype="int32")

        idx = 0
        refs_sorted = sorted(refs, key=lambda item: str(item.path))
        progress = tqdm(total=len(refs_sorted), desc=f"writing {output_path}")
        for path, group in groupby(refs_sorted, key=lambda item: item.path):
            arr = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
            bbox = load_mask_bbox(
                path,
                time_axis=time_axis,
                margin=roi_mask_margin,
                enabled=roi_mask_crop,
                require_mask=require_roi_mask,
            )
            for ref in group:
                vol = extract_frame_array(arr, ref.time_index, time_axis=time_axis, path=path)
                vol = crop_or_pad_around_bbox(vol, bbox, volume_size)
                vol = normalize_volume(
                    vol,
                    mode=normalize,
                    percentile_low=percentile_low,
                    percentile_high=percentile_high,
                )
                if channels == 1:
                    sample = vol[None]
                else:
                    sample = np.repeat(vol[None], channels, axis=0)
                y[idx] = sample.astype(np.float32)
                paths[idx] = str(ref.path)
                times[idx] = -1 if ref.time_index is None else int(ref.time_index)
                idx += 1
                progress.update(1)
        progress.close()

        h5.attrs["volume_size"] = volume_size
        h5.attrs["channels"] = channels
        h5.attrs["normalization"] = normalize
        h5.attrs["roi_mask_crop"] = roi_mask_crop
        h5.attrs["roi_mask_margin"] = roi_mask_margin
        h5.attrs["require_roi_mask"] = require_roi_mask
        h5.attrs["description"] = "Noisy real CMR frames for C-UNSURE; no clean targets stored."
