from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class FrameRef:
    path: Path
    time_index: int | None = None


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
        for path in sorted(root.glob(pattern)):
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
    if ref.time_index is not None:
        axis = time_axis if time_axis >= 0 else arr.ndim + time_axis
        arr = np.take(arr, ref.time_index, axis=axis)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"expected 3D frame after time extraction, got {arr.shape} from {ref.path}")
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
) -> None:
    if limit is not None:
        refs = refs[:limit]
    if not refs:
        raise ValueError(f"no NIfTI frames found for {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        y = h5.create_dataset(
            "y",
            shape=(len(refs), channels, *volume_size),
            dtype="float32",
            chunks=(1, channels, *volume_size),
            compression="gzip",
            compression_opts=4,
        )
        paths = h5.create_dataset("source_path", shape=(len(refs),), dtype=h5py.string_dtype())
        times = h5.create_dataset("time_index", shape=(len(refs),), dtype="int32")

        for idx, ref in enumerate(tqdm(refs, desc=f"writing {output_path}")):
            vol = load_frame(ref, time_axis=time_axis)
            vol = normalize_volume(
                vol,
                mode=normalize,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
            )
            vol = center_crop_or_pad(vol, volume_size)
            if channels == 1:
                sample = vol[None]
            else:
                sample = np.repeat(vol[None], channels, axis=0)
            y[idx] = sample.astype(np.float32)
            paths[idx] = str(ref.path)
            times[idx] = -1 if ref.time_index is None else int(ref.time_index)

        h5.attrs["volume_size"] = volume_size
        h5.attrs["channels"] = channels
        h5.attrs["description"] = "Noisy real CMR frames for C-UNSURE; no clean targets stored."
