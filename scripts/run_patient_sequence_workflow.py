#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import h5py
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cardiac_nodeo_uq.config import load_yaml, project_root, resolve_path


def portable_source_key(path: str) -> str:
    normalized = path.replace("\\", "/")
    for marker in ("/ACDC/", "/M&M1/", "/MnM2/"):
        if marker in normalized:
            return marker.strip("/") + "/" + normalized.split(marker, 1)[1]
    return normalized.lstrip("/")


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def run(command: list[str], *, root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_manifest(*, configured_path: Path, cfg: dict[str, object], root: Path) -> Path:
    if configured_path.exists():
        return configured_path
    manifest_config = cfg["data"].get("manifest_config")  # type: ignore[union-attr]
    if not manifest_config:
        raise FileNotFoundError(
            f"sequence manifest does not exist: {configured_path}; "
            "set data.manifest_config so it can be generated automatically"
        )
    config_path = resolve_path(manifest_config, root)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"manifest build config does not exist: {config_path}")
    print(
        json.dumps(
            {
                "manifest_missing": str(configured_path),
                "action": "build_nodeo_roi_splits",
                "config": str(config_path),
            },
            indent=2,
        ),
        flush=True,
    )
    run(
        [
            sys.executable,
            "scripts/build_nodeo_roi_splits.py",
            "--config",
            relative_or_absolute(config_path, root),
        ],
        root=root,
    )
    if not configured_path.exists():
        raise FileNotFoundError(
            f"manifest builder completed but did not create expected file: {configured_path}"
        )
    return configured_path


def write_yaml(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def select_solver_profile(
    cfg: dict[str, object], requested_solver: str | None
) -> tuple[str, dict[str, object]]:
    nodeo_cfg = cfg["nodeo"]
    solver = str(requested_solver or nodeo_cfg.get("solver", "rk4")).lower()  # type: ignore[union-attr]
    profiles = nodeo_cfg.get("solver_profiles")  # type: ignore[union-attr]
    if not isinstance(profiles, dict) or solver not in profiles:
        available = sorted(profiles) if isinstance(profiles, dict) else []
        raise ValueError(f"unsupported NODEO solver={solver!r}; available profiles: {available}")
    profile = profiles[solver]
    if not isinstance(profile, dict):
        raise ValueError(f"NODEO solver profile must be a mapping: {solver}")
    for key in ("run_config", "precomputed_dir", "sde_output_dir"):
        if key not in profile:
            raise KeyError(f"NODEO solver profile {solver!r} is missing {key!r}")
    return solver, profile


def select_manifest_row(
    manifest: Path,
    *,
    split: str,
    patient: str,
) -> tuple[dict[str, object], int]:
    split_rows = [row for row in read_jsonl(manifest) if str(row["split"]) == split]
    patient_lower = patient.lower()
    matches = [
        (index, row)
        for index, row in enumerate(split_rows)
        if patient_lower in str(row["source_path"]).lower()
        or patient_lower == str(row.get("sequence_id", "")).lower()
    ]
    if not matches:
        available = [Path(str(row["source_path"])).parent.name for row in split_rows[:20]]
        raise ValueError(
            f"no {split} sequence matches patient={patient!r}; examples: {available}"
        )
    if len(matches) > 1:
        sources = [str(row["source_path"]) for _, row in matches]
        raise ValueError(f"patient selector is ambiguous: {sources}")
    index, row = matches[0]
    return row, index


def resolve_nodeo_output(path_text: str, summary_path: Path, root: Path) -> Path | None:
    path = Path(path_text)
    candidates = [path, root / path, summary_path.parent / path.name]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def select_existing_nodeo(
    *,
    source_path: str,
    directory: Path,
    root: Path,
) -> tuple[Path, dict[str, object]]:
    summary_path = directory / "summary.jsonl"
    source_key = portable_source_key(source_path)
    if summary_path.exists():
        for row in read_jsonl(summary_path):
            if portable_source_key(str(row["source_path"])) != source_key:
                continue
            output = resolve_nodeo_output(str(row["output"]), summary_path, root)
            if output is not None:
                return output, row
    for path in sorted(directory.glob("[0-9]*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if portable_source_key(str(payload.get("source_path", ""))) == source_key:
            return path, {
                "sequence_id": payload.get("sequence_id", path.stem),
                "split": payload.get("split", "test"),
                "dataset": payload.get("dataset", "ACDC"),
                "source_path": payload["source_path"],
                "output": str(path),
                "best_epoch": payload.get("best_epoch"),
                "metrics": payload.get("metrics", {}),
            }
    raise FileNotFoundError(
        f"no existing NODEO result for {source_path} under {directory}"
    )


def run_nodeo_on_demand(
    *,
    root: Path,
    cfg: dict[str, object],
    run_config: str,
    split: str,
    split_index: int,
    work_dir: Path,
    overwrite: bool,
) -> tuple[Path, dict[str, object]]:
    base_path = resolve_path(run_config, root)
    assert base_path is not None
    nodeo_cfg = load_yaml(base_path)
    nodeo_cfg["device"] = cfg.get("device", nodeo_cfg.get("device", "auto"))
    nodeo_cfg["output"]["run_dir"] = relative_or_absolute(work_dir / "nodeo", root)
    runtime_cfg = work_dir / "runtime_nodeo.yaml"
    write_yaml(runtime_cfg, nodeo_cfg)
    command = [
        sys.executable,
        "scripts/run_nodeo_dir.py",
        "--config",
        relative_or_absolute(runtime_cfg, root),
        "--split",
        split,
        "--start-index",
        str(split_index),
        "--limit",
        "1",
    ]
    if overwrite:
        command.append("--overwrite")
    run(command, root=root)
    summary_path = work_dir / "nodeo" / split / "summary.jsonl"
    rows = read_jsonl(summary_path)
    if not rows:
        raise RuntimeError(f"NODEO produced no summary row: {summary_path}")
    row = rows[-1]
    output = resolve_nodeo_output(str(row["output"]), summary_path, root)
    if output is None:
        raise FileNotFoundError(row["output"])
    return output, row


def write_selected_nodeo_summary(
    *,
    path: Path,
    row: dict[str, object],
    output: Path,
    root: Path,
) -> None:
    selected = dict(row)
    selected["output"] = relative_or_absolute(output, root)
    path.write_text(json.dumps(selected) + "\n", encoding="utf-8")


def has_bidirectional_nodeo(path: Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return all(
        key in payload
        for key in ("inverse_displacement", "backward_warped")
    )


def fit_sde(
    *,
    root: Path,
    cfg: dict[str, object],
    split: str,
    nodeo_summary: Path,
    work_dir: Path,
) -> Path:
    base_path = resolve_path(cfg["sde"]["config"], root)  # type: ignore[index]
    assert base_path is not None
    sde_cfg = load_yaml(base_path)
    sde_cfg["device"] = cfg.get("device", sde_cfg.get("device", "auto"))
    summary_value = relative_or_absolute(nodeo_summary, root)
    sde_cfg["nodeo"]["summaries"] = {name: summary_value for name in ("train", "val", "test")}
    sde_cfg["output"]["run_dir"] = relative_or_absolute(work_dir / "sde", root)
    runtime_cfg = work_dir / "runtime_sde.yaml"
    write_yaml(runtime_cfg, sde_cfg)
    run(
        [
            sys.executable,
            "scripts/run_sde_sequence_posthoc.py",
            "--config",
            relative_or_absolute(runtime_cfg, root),
            "--split",
            split,
            "--limit",
            "1",
            "--overwrite",
        ],
        root=root,
    )
    outputs = sorted((work_dir / "sde" / split).glob("[0-9]*.pt"))
    if len(outputs) != 1:
        raise RuntimeError(f"expected one SDE output, found {outputs}")
    return outputs[0]


def local_acdc_source(source: str, root: Path) -> Path:
    path = Path(source)
    if path.exists():
        return path
    normalized = source.replace("\\", "/")
    if "/ACDC/" in normalized:
        candidate = root / "datasets" / "ACDC" / normalized.split("/ACDC/", 1)[1]
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve ACDC source path: {source}")


def acdc_clinical_inputs(
    *,
    source_path: str,
    raw_times: list[int],
    root: Path,
) -> tuple[Path, int, int]:
    source = local_acdc_source(source_path, root)
    info_text = (source.parent / "Info.cfg").read_text(encoding="utf-8")
    ed_match = re.search(r"^ED\s*:\s*(\d+)", info_text, flags=re.MULTILINE)
    es_match = re.search(r"^ES\s*:\s*(\d+)", info_text, flags=re.MULTILINE)
    if ed_match is None or es_match is None:
        raise ValueError(f"cannot parse ED/ES from {source.parent / 'Info.cfg'}")
    ed_raw = int(ed_match.group(1))
    es_raw = int(es_match.group(1))
    stem = source.name.replace("_4d.nii.gz", "").replace("_4d.nii", "")
    mask = source.parent / f"{stem}_frame{ed_raw:02d}_gt.nii.gz"
    if not mask.exists():
        raise FileNotFoundError(mask)
    return mask, raw_times.index(ed_raw - 1), raw_times.index(es_raw - 1)


def compute_ef(
    *,
    root: Path,
    cfg: dict[str, object],
    manifest_row: dict[str, object],
    deformation: Path,
    work_dir: Path,
) -> Path:
    raw_times = [int(value) for value in manifest_row["time_indices"]]  # type: ignore[union-attr]
    mask, ed_index, es_index = acdc_clinical_inputs(
        source_path=str(manifest_row["source_path"]), raw_times=raw_times, root=root
    )
    h5_path = resolve_path(str(manifest_row["h5_path"]), root)
    assert h5_path is not None
    with h5py.File(h5_path, "r") as h5:
        volume_size = [int(value) for value in h5.attrs["volume_size"]]
    output = work_dir / "ef_prediction_band.json"
    clinical_cfg = cfg["clinical"]
    run(
        [
            sys.executable,
            "scripts/compute_clinical_metrics.py",
            "--deformation",
            relative_or_absolute(deformation, root),
            "--reference-mask",
            relative_or_absolute(mask, root),
            "--output",
            relative_or_absolute(output, root),
            "--displacement-key",
            "total_displacement",
            "--volume-size",
            *(str(value) for value in volume_size),
            "--labels",
            str(clinical_cfg.get("lv_label", 3)),  # type: ignore[union-attr]
            "--ed-index",
            str(ed_index),
            "--es-index",
            str(es_index),
            "--coverage",
            str(clinical_cfg.get("coverage", 0.95)),  # type: ignore[union-attr]
            "--roi-mask-crop",
            "--roi-mask-margin",
            *(str(value) for value in clinical_cfg.get("roi_mask_margin", [0, 16, 16])),  # type: ignore[union-attr]
        ],
        root=root,
    )
    return output


def validate_ambiguity(
    *,
    root: Path,
    cfg: dict[str, object],
    deformation: Path,
    work_dir: Path,
) -> Path:
    validation_cfg = cfg.get("validation", {})
    output = work_dir / "ambiguity_ed_es_validation.json"
    run(
        [
            sys.executable,
            "scripts/evaluate_ambiguity_decomposition.py",
            "--input",
            relative_or_absolute(deformation, root),
            "--output",
            relative_or_absolute(output, root),
            "--datasets-root",
            str(validation_cfg.get("datasets_root", "datasets")),  # type: ignore[union-attr]
            "--device",
            str(cfg.get("device", "auto")),
            "--high-error-mm",
            str(validation_cfg.get("high_error_mm", 2.0)),  # type: ignore[union-attr]
            "--roi-mask-margin",
            *(
                str(value)
                for value in validation_cfg.get("roi_mask_margin", [0, 16, 16])  # type: ignore[union-attr]
            ),
        ],
        root=root,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acdc/patient_sequence_workflow.yaml")
    parser.add_argument("--option", type=int, choices=(1, 2))
    parser.add_argument("--patient")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--solver", choices=("euler", "rk4", "dopri5"))
    parser.add_argument("--overwrite-nodeo", action="store_true")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)
    option = int(args.option or cfg.get("option", 2))
    solver, solver_profile = select_solver_profile(cfg, args.solver)
    selection_cfg = cfg["selection"]
    patient = str(args.patient or selection_cfg["patient"])
    split = str(args.split or selection_cfg.get("split", "test"))
    manifest = resolve_path(cfg["data"]["manifest"], root)
    assert manifest is not None
    manifest = ensure_manifest(configured_path=manifest, cfg=cfg, root=root)
    manifest_row, split_index = select_manifest_row(
        manifest, split=split, patient=patient
    )
    patient_name = Path(str(manifest_row["source_path"])).parent.name
    work_root = resolve_path(cfg["output"]["run_dir"], root)
    assert work_root is not None
    work_dir = work_root / solver / split / patient_name
    work_dir.mkdir(parents=True, exist_ok=True)

    if option == 1:
        nodeo_output, nodeo_row = run_nodeo_on_demand(
            root=root,
            cfg=cfg,
            run_config=str(solver_profile["run_config"]),
            split=split,
            split_index=split_index,
            work_dir=work_dir,
            overwrite=bool(args.overwrite_nodeo),
        )
        if not has_bidirectional_nodeo(nodeo_output):
            print(
                json.dumps(
                    {
                        "stale_nodeo_output": str(nodeo_output),
                        "action": "refit_with_bidirectional_loss",
                    },
                    indent=2,
                ),
                flush=True,
            )
            nodeo_output, nodeo_row = run_nodeo_on_demand(
                root=root,
                cfg=cfg,
                run_config=str(solver_profile["run_config"]),
                split=split,
                split_index=split_index,
                work_dir=work_dir,
                overwrite=True,
            )
    else:
        precomputed_root = resolve_path(str(solver_profile["precomputed_dir"]), root)
        assert precomputed_root is not None
        precomputed_dir = precomputed_root / split if (precomputed_root / split).exists() else precomputed_root
        nodeo_output, nodeo_row = select_existing_nodeo(
            source_path=str(manifest_row["source_path"]),
            directory=precomputed_dir,
            root=root,
        )
        if not has_bidirectional_nodeo(nodeo_output):
            raise RuntimeError(
                f"precomputed NODEO output is from the old one-way workflow: {nodeo_output}. "
                "Re-run NODEO with the current configuration or use OPTION=1."
            )

    selected_summary = work_dir / "selected_nodeo_summary.jsonl"
    write_selected_nodeo_summary(
        path=selected_summary,
        row=nodeo_row,
        output=nodeo_output,
        root=root,
    )
    sde_output = fit_sde(
        root=root,
        cfg=cfg,
        split=split,
        nodeo_summary=selected_summary,
        work_dir=work_dir,
    )
    ambiguity_validation = validate_ambiguity(
        root=root,
        cfg=cfg,
        deformation=sde_output,
        work_dir=work_dir,
    )
    ef_output = compute_ef(
        root=root,
        cfg=cfg,
        manifest_row=manifest_row,
        deformation=sde_output,
        work_dir=work_dir,
    )
    ef = json.loads(ef_output.read_text(encoding="utf-8"))
    result = {
        "option": option,
        "nodeo_solver": solver,
        "split": split,
        "patient": patient_name,
        "source_path": manifest_row["source_path"],
        "nodeo_output": relative_or_absolute(nodeo_output, root),
        "uncertainty_source": (
            "photometrically_corrected_structural_bidirectional_nodeo_ambiguity"
        ),
        "sde_output": relative_or_absolute(sde_output, root),
        "ambiguity_validation": relative_or_absolute(ambiguity_validation, root),
        "ef_output": relative_or_absolute(ef_output, root),
        "ef": ef["ef"],
        "ef_prediction_band": ef["prediction_bands"]["ef"],
    }
    result_path = work_dir / "workflow_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
