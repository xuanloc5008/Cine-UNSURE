from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


def decode_h5_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def canonical_source_key(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "/datasets/"
    if marker in normalized:
        return "datasets/" + normalized.split(marker, 1)[1]
    return normalized.lstrip("/")


def resolve_portable_source_path(
    stored_path: str | Path,
    *,
    datasets_root: str | Path,
    project_root: str | Path,
) -> Path:
    """Resolve source metadata produced on another machine.

    Checkpoints may contain paths rooted at macOS volumes, Kaggle, or a remote
    workspace. The dataset name and its relative suffix are stable across those
    environments, so resolution is based on that portable portion.
    """

    direct = Path(stored_path).expanduser()
    if direct.exists():
        return direct.resolve()

    root = Path(project_root)
    configured_root = Path(datasets_root).expanduser()
    if not configured_root.is_absolute():
        configured_root = root / configured_root

    normalized = str(stored_path).replace("\\", "/")
    portable_candidates: list[Path] = []
    if "/datasets/" in normalized:
        portable_candidates.append(Path(normalized.split("/datasets/", 1)[1]))
    elif normalized.startswith("datasets/"):
        portable_candidates.append(Path(normalized[len("datasets/") :]))

    for dataset in ("ACDC", "M&M1", "MnM2"):
        marker = f"/{dataset}/"
        if marker in normalized:
            portable_candidates.append(
                Path(dataset) / normalized.split(marker, 1)[1]
            )
        elif normalized.startswith(f"{dataset}/"):
            portable_candidates.append(Path(normalized))

    attempted: list[Path] = []
    search_roots = [configured_root]
    default_root = root / "datasets"
    if default_root != configured_root:
        search_roots.append(default_root)
    for portable in portable_candidates:
        for search_root in search_roots:
            if portable.parts and search_root.name == portable.parts[0]:
                candidate = search_root.joinpath(*portable.parts[1:])
            else:
                candidate = search_root / portable
            attempted.append(candidate)
            if candidate.exists():
                return candidate.resolve()

    attempted_text = ", ".join(str(path) for path in attempted) or "no portable dataset suffix found"
    raise FileNotFoundError(
        f"cannot remap source path {stored_path!s}; attempted: {attempted_text}"
    )


@dataclass(frozen=True)
class NODEOSequenceRef:
    sequence_id: str
    split: str
    dataset: str
    source_path: str
    h5_path: str
    indices: tuple[int, ...]
    time_indices: tuple[int, ...]


def load_nodeo_manifest(path: str | Path, *, split: str | None = None) -> list[NODEOSequenceRef]:
    refs: list[NODEOSequenceRef] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if split is not None and row["split"] != split:
                continue
            refs.append(
                NODEOSequenceRef(
                    sequence_id=str(row["sequence_id"]),
                    split=str(row["split"]),
                    dataset=str(row["dataset"]),
                    source_path=str(row["source_path"]),
                    h5_path=str(row["h5_path"]),
                    indices=tuple(int(v) for v in row["indices"]),
                    time_indices=tuple(int(v) for v in row["time_indices"]),
                )
            )
    return refs


class NODEOROISequenceDataset(Dataset[dict[str, object]]):
    """Cine sequences loaded only from pre-cropped ROI HDF5 frames.

    Only the datasets ``y``, ``source_path`` and ``time_index`` are used.
    Only cropped image sequences and their source metadata are read.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        root: str | Path,
        split: str,
        min_length: int = 2,
    ) -> None:
        self.root = Path(root)
        self.refs = [
            ref for ref in load_nodeo_manifest(manifest_path, split=split) if len(ref.indices) >= int(min_length)
        ]

    def __len__(self) -> int:
        return len(self.refs)

    def _resolve_h5(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    def __getitem__(self, index: int) -> dict[str, object]:
        ref = self.refs[index]
        with h5py.File(self._resolve_h5(ref.h5_path), "r") as h5:
            images = torch.from_numpy(h5["y"][list(ref.indices)]).float()
            stored_sources = [decode_h5_string(v) for v in h5["source_path"][list(ref.indices)]]
            stored_times = tuple(int(v) for v in h5["time_index"][list(ref.indices)])
        if len(set(stored_sources)) != 1 or stored_sources[0] != ref.source_path:
            raise ValueError(f"manifest/source mismatch for {ref.sequence_id}")
        if stored_times != ref.time_indices:
            raise ValueError(f"manifest/time mismatch for {ref.sequence_id}")

        raw_times = torch.tensor(ref.time_indices, dtype=torch.float32)
        if len(raw_times) > 1 and raw_times[-1] > raw_times[0]:
            times = (raw_times - raw_times[0]) / (raw_times[-1] - raw_times[0])
        else:
            times = torch.linspace(0.0, 1.0, len(raw_times))
        return {
            "images": images.contiguous(),
            "times": times,
            "raw_time_indices": raw_times.long(),
            "sequence_id": ref.sequence_id,
            "split": ref.split,
            "dataset": ref.dataset,
            "source_path": ref.source_path,
        }


class NODEOTrajectoryStore:
    """Index precomputed per-sequence NODEO mean trajectories by source path."""

    def __init__(self, summaries: list[str | Path], *, root: str | Path) -> None:
        self.root = Path(root)
        self.outputs: dict[str, Path] = {}
        self.cache: OrderedDict[str, Tensor] = OrderedDict()
        self.cache_size = 8
        for configured_summary in summaries:
            summary = Path(configured_summary)
            if not summary.is_absolute():
                summary = self.root / summary
            if not summary.exists():
                raise FileNotFoundError(summary)
            with summary.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = canonical_source_key(str(row["source_path"]))
                    output = Path(str(row["output"]))
                    if not output.is_absolute():
                        output = self.root / output
                    if not output.exists():
                        portable_output = summary.parent / output.name
                        if portable_output.exists():
                            output = portable_output
                    self.outputs[key] = output

    def load(self, source_path: str) -> Tensor:
        key = canonical_source_key(source_path)
        if key not in self.outputs:
            raise KeyError(f"no NODEO trajectory for source: {key}")
        if key not in self.cache:
            payload = torch.load(self.outputs[key], map_location="cpu", weights_only=False)
            self.cache[key] = payload["phi_bar"].float()
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]
