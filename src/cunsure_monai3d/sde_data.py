from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import torch
from torch.utils.data import Dataset


def decode_h5_string(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_latent_covariance(h5: h5py.File, indices: list[int], *, mode: Literal["full", "diag"]) -> torch.Tensor:
    if "latent_covariance_diag" in h5:
        diagonal = torch.from_numpy(h5["latent_covariance_diag"][indices]).float()
        return diagonal if mode == "diag" else torch.diag_embed(diagonal)
    if "latent_covariance_psd" in h5:
        covariance = torch.from_numpy(h5["latent_covariance_psd"][indices]).float()
        return covariance.diagonal(dim1=-2, dim2=-1) if mode == "diag" else covariance
    raise KeyError("latent H5 must contain latent_covariance_diag or latent_covariance_psd")


@dataclass(frozen=True)
class SequenceRef:
    source_path: str
    dataset: str
    indices: tuple[int, ...]
    time_indices: tuple[int, ...]


def build_sequence_refs(
    h5_path: str | Path,
    *,
    min_length: int = 1,
) -> list[SequenceRef]:
    groups: dict[str, list[tuple[int, int, str]]] = {}
    with h5py.File(h5_path, "r") as h5:
        source_paths = [decode_h5_string(v) for v in h5["source_path"][:]]
        datasets = [decode_h5_string(v) for v in h5["dataset"][:]]
        time_indices = [int(v) for v in h5["time_index"][:]]

    for idx, (source, dataset, time_index) in enumerate(zip(source_paths, datasets, time_indices, strict=True)):
        groups.setdefault(source, []).append((idx, time_index, dataset))

    refs: list[SequenceRef] = []
    for source, items in groups.items():
        items = sorted(items, key=lambda item: item[1] if item[1] >= 0 else item[0])
        indices = tuple(item[0] for item in items)
        times = tuple(item[1] for item in items)
        dataset = items[0][2]
        if len(indices) >= min_length:
            refs.append(SequenceRef(source_path=source, dataset=dataset, indices=indices, time_indices=times))
    return sorted(refs, key=lambda ref: (ref.dataset, ref.source_path))


class LatentObservationSequenceDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        h5_path: str | Path,
        *,
        min_length: int = 1,
        covariance: Literal["full", "diag"] = "full",
        normalize_time: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        if covariance not in {"full", "diag"}:
            raise ValueError("covariance must be 'full' or 'diag'")
        self.covariance = covariance
        self.normalize_time = normalize_time
        self.refs = build_sequence_refs(self.h5_path, min_length=min_length)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        indices = list(ref.indices)
        with h5py.File(self.h5_path, "r") as h5:
            z = torch.from_numpy(h5["z"][indices]).float()
            covariance = load_latent_covariance(h5, indices, mode=self.covariance)

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

        return {
            "z": z,
            "R": covariance,
            "times": times,
            "raw_time_indices": raw_times.long(),
            "dataset": ref.dataset,
            "source_path": ref.source_path,
            "indices": torch.tensor(indices, dtype=torch.long),
        }
