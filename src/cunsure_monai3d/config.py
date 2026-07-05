from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def resolve_path(path: str | Path | None, base: Path | None = None) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path)
    if p.is_absolute():
        raise ValueError(f"absolute paths are disabled by design: {p}")
    return (base or project_root()) / p


def as_tuple_int(values: list[int] | tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(int(v) for v in values)
