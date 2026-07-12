#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import project_root, resolve_path
from cunsure_monai3d.nodeo_roi_data import canonical_source_key


def infer_dataset_name(source_path: str) -> str:
    parts = set(Path(source_path).parts)
    if "ACDC" in parts:
        return "ACDC"
    if "M&M1" in parts or "M_and_M1" in parts:
        return "M&M1"
    if "MnM2" in parts:
        return "MnM2"
    return "unknown"


def read_summary_outputs(input_dirs: list[Path]) -> list[Path]:
    # Files are authoritative. A resumed run can legitimately have a partial
    # summary.jsonl while all numbered frame outputs are present.
    paths: list[Path] = []
    for input_dir in input_dirs:
        paths.extend(path for path in input_dir.glob("[0-9]*.pt") if path.is_file())
    return sorted(paths, key=lambda path: (str(path.parent), path.name))


def resolve_maybe_absolute(path: str | Path, root: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else resolve_path(p, root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, action="append")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    args = parser.parse_args()

    root = project_root()
    input_dirs = [resolve_maybe_absolute(path, root) for path in args.input_dir]
    output_path = resolve_maybe_absolute(args.output, root)
    paths = read_summary_outputs(input_dirs)
    if not paths:
        raise ValueError(f"no .pt outputs found in {input_dirs}")

    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    z0 = first["z"].reshape(-1).float()
    if "latent_covariance_diag" in first:
        covariance_key = "latent_covariance_diag"
        covariance_storage = "diag"
    elif "latent_covariance_psd" in first:
        covariance_key = "latent_covariance_psd"
        covariance_storage = "full"
    else:
        raise KeyError(f"{paths[0]} has no latent covariance")
    eta0 = first["eta"].float()
    n = len(paths)
    latent_dim = int(z0.numel())

    compression_kwargs = {}
    if args.compression != "none":
        compression_kwargs["compression"] = args.compression
        if args.compression == "gzip":
            compression_kwargs["compression_opts"] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    trace_values: list[float] = []
    seen_frames: set[tuple[str, int]] = set()
    eta_mode = "per_frame" if first.get("method") == "score_cunsure_per_frame" else "replicated_global"
    with h5py.File(temporary_path, "w") as h5:
        z = h5.create_dataset("z", shape=(n, latent_dim), dtype="float32", chunks=(1, latent_dim), **compression_kwargs)
        covariance_shape = (n, latent_dim) if covariance_storage == "diag" else (n, latent_dim, latent_dim)
        covariance_chunks = (1, latent_dim) if covariance_storage == "diag" else (1, latent_dim, latent_dim)
        cov = h5.create_dataset(covariance_key, shape=covariance_shape, dtype="float32", chunks=covariance_chunks, **compression_kwargs)
        source_path = h5.create_dataset("source_path", shape=(n,), dtype=h5py.string_dtype())
        dataset = h5.create_dataset("dataset", shape=(n,), dtype=h5py.string_dtype())
        time_index = h5.create_dataset("time_index", shape=(n,), dtype="int32")
        output_file = h5.create_dataset("output_file", shape=(n,), dtype=h5py.string_dtype())
        trace = h5.create_dataset("covariance_trace", shape=(n,), dtype="float32")

        eta = h5.create_dataset(
            "eta",
            shape=(n, *eta0.shape),
            dtype="float32",
            chunks=(1, *eta0.shape),
            **compression_kwargs,
        )
        h5.attrs["latent_dim"] = latent_dim
        h5.attrs["num_samples"] = n
        h5.attrs["covariance_storage"] = covariance_storage
        h5.attrs["eta_mode"] = eta_mode
        h5.attrs["description"] = (
            "CineMA observations with frame-wise score C-UNSURE covariance."
            if eta_mode == "per_frame"
            else "CineMA observations with replicated global C-UNSURE covariance."
        )

        for idx, path in enumerate(tqdm(paths, desc=f"packaging {output_path}")):
            item = first if idx == 0 else torch.load(path, map_location="cpu", weights_only=False)
            zi = item["z"].reshape(-1).float()
            if covariance_key not in item:
                raise KeyError(f"covariance schema mismatch for {path}: expected {covariance_key}")
            covi = item[covariance_key].float()
            etai = item["eta"].float()
            if zi.numel() != latent_dim:
                raise ValueError(f"latent dim mismatch for {path}: {zi.numel()} != {latent_dim}")
            expected_covariance_shape = (latent_dim,) if covariance_storage == "diag" else (latent_dim, latent_dim)
            if tuple(covi.shape) != expected_covariance_shape:
                raise ValueError(f"covariance shape mismatch for {path}: {tuple(covi.shape)}")
            if tuple(etai.shape) != tuple(eta0.shape):
                raise ValueError(f"eta shape mismatch for {path}: {tuple(etai.shape)}")
            src = str(item.get("source_path", ""))
            frame_key = (canonical_source_key(src), int(item.get("time_index", -1)))
            if frame_key in seen_frames:
                raise ValueError(f"duplicate source/time frame across input directories: {frame_key}")
            seen_frames.add(frame_key)
            z[idx] = zi.numpy()
            cov[idx] = covi.numpy()
            eta[idx] = etai.numpy()
            source_path[idx] = src
            dataset[idx] = str(item.get("dataset") or infer_dataset_name(src))
            time_index[idx] = int(item.get("time_index", -1))
            output_file[idx] = str(path)
            trace_value = float(covi.sum()) if covariance_storage == "diag" else float(torch.trace(covi))
            trace[idx] = trace_value
            trace_values.append(trace_value)
    temporary_path.replace(output_path)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "num_samples": n,
                "latent_dim": latent_dim,
                "eta_mode": eta_mode,
                "covariance_storage": covariance_storage,
                "first_eta_norm": float(eta0.norm()),
                "mean_covariance_trace": float(np.mean(trace_values)) if trace_values else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
