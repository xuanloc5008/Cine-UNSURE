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


def infer_dataset_name(source_path: str) -> str:
    parts = set(Path(source_path).parts)
    if "ACDC" in parts:
        return "ACDC"
    if "M&M1" in parts or "M_and_M1" in parts:
        return "M&M1"
    if "MnM2" in parts:
        return "MnM2"
    return "unknown"


def read_summary_outputs(input_dir: Path) -> list[Path]:
    summary = input_dir / "summary.jsonl"
    if not summary.exists():
        return sorted(path for path in input_dir.glob("*.pt") if path.is_file())
    outputs: list[Path] = []
    with summary.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            output = row.get("output")
            if output:
                outputs.append(input_dir / str(output))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    args = parser.parse_args()

    root = project_root()
    input_dir = resolve_path(args.input_dir, root)
    output_path = resolve_path(args.output, root)
    paths = read_summary_outputs(input_dir)
    if not paths:
        raise ValueError(f"no .pt outputs found in {input_dir}")

    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    z0 = first["z"].reshape(-1).float()
    cov0 = first["latent_covariance_psd"].float()
    eta0 = first["eta"].float()
    n = len(paths)
    latent_dim = int(z0.numel())

    compression_kwargs = {}
    if args.compression != "none":
        compression_kwargs["compression"] = args.compression
        if args.compression == "gzip":
            compression_kwargs["compression_opts"] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_values: list[float] = []
    with h5py.File(output_path, "w") as h5:
        z = h5.create_dataset("z", shape=(n, latent_dim), dtype="float32", chunks=(1, latent_dim), **compression_kwargs)
        cov = h5.create_dataset(
            "latent_covariance_psd",
            shape=(n, latent_dim, latent_dim),
            dtype="float32",
            chunks=(1, latent_dim, latent_dim),
            **compression_kwargs,
        )
        source_path = h5.create_dataset("source_path", shape=(n,), dtype=h5py.string_dtype())
        dataset = h5.create_dataset("dataset", shape=(n,), dtype=h5py.string_dtype())
        time_index = h5.create_dataset("time_index", shape=(n,), dtype="int32")
        output_file = h5.create_dataset("output_file", shape=(n,), dtype=h5py.string_dtype())
        trace = h5.create_dataset("covariance_trace", shape=(n,), dtype="float32")

        h5.create_dataset("eta", data=eta0.numpy().astype(np.float32))
        h5.attrs["latent_dim"] = latent_dim
        h5.attrs["num_samples"] = n
        h5.attrs["description"] = "CineMA latent observations and PSD latent covariance from C-UNSURE eta."

        for idx, path in enumerate(tqdm(paths, desc=f"packaging {output_path}")):
            item = first if idx == 0 else torch.load(path, map_location="cpu", weights_only=False)
            zi = item["z"].reshape(-1).float()
            covi = item["latent_covariance_psd"].float()
            if zi.numel() != latent_dim:
                raise ValueError(f"latent dim mismatch for {path}: {zi.numel()} != {latent_dim}")
            if tuple(covi.shape) != (latent_dim, latent_dim):
                raise ValueError(f"covariance shape mismatch for {path}: {tuple(covi.shape)}")
            src = str(item.get("source_path", ""))
            z[idx] = zi.numpy()
            cov[idx] = covi.numpy()
            source_path[idx] = src
            dataset[idx] = str(item.get("dataset") or infer_dataset_name(src))
            time_index[idx] = int(item.get("time_index", -1))
            output_file[idx] = str(path)
            trace_value = float(torch.trace(covi))
            trace[idx] = trace_value
            trace_values.append(trace_value)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "num_samples": n,
                "latent_dim": latent_dim,
                "eta_norm": float(eta0.norm()),
                "mean_covariance_trace": float(np.mean(trace_values)) if trace_values else 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
