#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cunsure_monai3d.config import project_root, resolve_path
from cunsure_monai3d.sde_data import LatentObservationSequenceDataset, build_sequence_refs, decode_h5_string


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--random-checks", type=int, default=5)
    args = parser.parse_args()

    root = project_root()
    h5_path = resolve_path(args.h5, root)

    with h5py.File(h5_path, "r") as h5:
        datasets = [decode_h5_string(v) for v in h5["dataset"][:]]
        trace = np.asarray(h5["covariance_trace"][:], dtype=np.float64)
        report = {
            "path": str(h5_path),
            "keys": list(h5.keys()),
            "attrs": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in h5.attrs.items()},
            "z_shape": list(h5["z"].shape),
            "covariance_shape": list(h5["latent_covariance_psd"].shape),
            "dataset_counts": dict(Counter(datasets)),
            "trace": {
                "min": float(trace.min()) if trace.size else 0.0,
                "mean": float(trace.mean()) if trace.size else 0.0,
                "max": float(trace.max()) if trace.size else 0.0,
            },
        }

        n = int(h5["z"].shape[0])
        checks = []
        if n:
            sample_indices = random.sample(range(n), min(args.random_checks, n))
            for idx in sample_indices:
                cov = torch.from_numpy(h5["latent_covariance_psd"][idx]).float()
                cov = 0.5 * (cov + cov.T)
                eig_min = float(torch.linalg.eigvalsh(cov).min())
                checks.append(
                    {
                        "index": idx,
                        "dataset": decode_h5_string(h5["dataset"][idx]),
                        "time_index": int(h5["time_index"][idx]),
                        "trace": float(torch.trace(cov)),
                        "eig_min": eig_min,
                    }
                )
        report["random_psd_checks"] = checks

    refs = build_sequence_refs(h5_path, min_length=args.min_length)
    lengths = [len(ref.indices) for ref in refs]
    report["sequences"] = {
        "count": len(refs),
        "min_length": int(min(lengths)) if lengths else 0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_length": int(max(lengths)) if lengths else 0,
        "by_dataset": dict(Counter(ref.dataset for ref in refs)),
    }

    ds = LatentObservationSequenceDataset(h5_path, min_length=args.min_length, covariance="full")
    if len(ds):
        sample = ds[0]
        report["first_sequence"] = {
            "dataset": sample["dataset"],
            "source_path": sample["source_path"],
            "z_shape": list(sample["z"].shape),
            "R_shape": list(sample["R"].shape),
            "times_shape": list(sample["times"].shape),
            "time_indices": sample["raw_time_indices"].tolist(),
        }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
