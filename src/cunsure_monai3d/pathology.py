from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


ACDC_PATHOLOGIES = ("NOR", "DCM", "HCM", "MINF", "RV")


def resolve_acdc_source(source: str, root: Path) -> Path:
    path = Path(source)
    if path.exists():
        return path
    normalized = source.replace("\\", "/")
    for marker in ("/datasets/ACDC/", "/ACDC/"):
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            candidate = root / "datasets" / "ACDC" / relative
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"cannot resolve ACDC source path: {source}")


def read_acdc_pathology(source: str, root: Path) -> str:
    info = resolve_acdc_source(source, root).parent / "Info.cfg"
    match = re.search(r"^Group\s*:\s*(\S+)", info.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"cannot parse Group from {info}")
    label = match.group(1).upper()
    if label not in ACDC_PATHOLOGIES:
        raise ValueError(f"unsupported ACDC pathology {label!r} in {info}")
    return label


def _resample(values: list[float], time_points: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or not len(array):
        raise ValueError("clinical trajectory must be a non-empty vector")
    if len(array) == 1:
        return np.repeat(array, time_points)
    source_time = np.linspace(0.0, 1.0, len(array), dtype=np.float32)
    target_time = np.linspace(0.0, 1.0, time_points, dtype=np.float32)
    return np.interp(target_time, source_time, array).astype(np.float32)


def _standard_errors(variances: list[float | None]) -> list[float]:
    return [float(np.sqrt(max(float(value or 0.0), 0.0))) for value in variances]


def clinical_feature_names(time_points: int) -> list[str]:
    names = ["ef_mean", "ef_log_se"]
    for metric in ("volume", "wall_motion", "strain_xx", "strain_yy", "strain_zz"):
        names.extend(f"{metric}_mean_t{index:02d}" for index in range(time_points))
        names.extend(f"{metric}_log_se_t{index:02d}" for index in range(time_points))
    return names


def clinical_features(row: dict, *, time_points: int) -> tuple[np.ndarray, np.ndarray]:
    mean_parts: list[np.ndarray] = []
    se_parts: list[np.ndarray] = []

    mean_parts.append(_resample(row["volume_curve"], time_points))
    se_parts.append(_resample(_standard_errors(row["volume_variance"]), time_points))
    mean_parts.append(_resample(row["wall_motion_mean"], time_points))
    se_parts.append(_resample(_standard_errors(row["wall_motion_variance"]), time_points))
    for key in ("strain_xx", "strain_yy", "strain_zz"):
        mean_parts.append(_resample([frame[key] for frame in row["strain_mean"]], time_points))
        se_parts.append(
            _resample(_standard_errors([frame[key] for frame in row["strain_variance"]]), time_points)
        )

    ef_mean = float(row["ef"])
    ef_se = float(np.sqrt(max(float(row.get("ef_variance") or 0.0), 0.0)))
    features = [ef_mean, np.log1p(ef_se)]
    feature_variance = [ef_se**2, 0.0]
    for means, standard_errors in zip(mean_parts, se_parts, strict=True):
        features.extend(means.tolist())
        features.extend(np.log1p(standard_errors).tolist())
        feature_variance.extend(np.square(standard_errors).tolist())
        feature_variance.extend(np.zeros_like(standard_errors).tolist())
    return np.asarray(features, dtype=np.float32), np.asarray(feature_variance, dtype=np.float32)


@dataclass(frozen=True)
class ClinicalPathologySample:
    clinical_path: Path
    source_path: str
    pathology: str
    features: np.ndarray
    feature_variance: np.ndarray


def load_clinical_samples(paths: list[Path], *, root: Path, time_points: int) -> list[ClinicalPathologySample]:
    samples = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        source = str(row["source_path"])
        pathology = read_acdc_pathology(source, root)
        features, feature_variance = clinical_features(row, time_points=time_points)
        if not np.isfinite(features).all() or not np.isfinite(feature_variance).all():
            raise ValueError(f"non-finite pathology features in {path}")
        samples.append(ClinicalPathologySample(path, source, pathology, features, feature_variance))
    if not samples:
        raise ValueError("no clinical prediction JSON files found")
    return samples


class PathologyClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], num_classes: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(previous, hidden), nn.GELU(), nn.Dropout(dropout)))
            previous = hidden
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def pathology_probability_bands(
    model: PathologyClassifier,
    standardized_features: torch.Tensor,
    standardized_variance: torch.Tensor,
    *,
    gaussian_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = standardized_features.detach().clone().requires_grad_(True)
    probabilities = torch.softmax(model(x.unsqueeze(0))[0], dim=0)
    variances = []
    for index in range(probabilities.numel()):
        gradient = torch.autograd.grad(probabilities[index], x, retain_graph=True)[0]
        variances.append((gradient.square() * standardized_variance).sum())
    probability_variance = torch.stack(variances).clamp_min(0.0)
    standard_error = probability_variance.sqrt()
    lower = (probabilities - gaussian_multiplier * standard_error).clamp(0.0, 1.0)
    upper = (probabilities + gaussian_multiplier * standard_error).clamp(0.0, 1.0)
    return probabilities.detach(), probability_variance.detach(), lower.detach(), upper.detach()
