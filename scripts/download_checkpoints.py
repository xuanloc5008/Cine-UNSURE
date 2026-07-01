#!/usr/bin/env python3
"""Download/cache foundation-model checkpoints used by the C-UNSURE pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_external_root() -> Path:
    return default_workspace_root() / "work" / "external"


CINEMA_FILES = [
    ("mathpluscode/CineMA", "pretrained/cinema.safetensors"),
    ("mathpluscode/CineMA", "pretrained/config.yaml"),
]

MEDSAM2_FILES = [
    ("wanglab/MedSAM2", "MedSAM2_2411.pt"),
    ("wanglab/MedSAM2", "MedSAM2_US_Heart.pt"),
    ("wanglab/MedSAM2", "MedSAM2_MRI_LiverLesion.pt"),
    ("wanglab/MedSAM2", "MedSAM2_CTLesion.pt"),
    ("wanglab/MedSAM2", "MedSAM2_latest.pt"),
]

EFFICIENTTAM_FILES = [
    ("yunyangx/efficient-track-anything", "efficienttam_s_512x512.pt"),
    ("yunyangx/efficient-track-anything", "efficienttam_ti_512x512.pt"),
]

SAM2_BASE_URLS = [
    (
        "sam2.1_hiera_tiny.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", default=str(default_external_root()))
    parser.add_argument("--cinema-cache-dir", default=None, help="Optional HuggingFace cache dir for CineMA")
    parser.add_argument("--medsam2-checkpoint-dir", default=None, help="Output dir for MedSAM2/EfficientTAM/SAM2 checkpoints")
    parser.add_argument("--only", nargs="+", choices=["cinema", "medsam2", "efficienttam", "sam2-base"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download direct URL files even if they exist")
    return parser.parse_args()


def should_download(group: str, selected: list[str] | None) -> bool:
    return selected is None or group in selected


def hf_download(repo_id: str, filename: str, *, cache_dir: str | None, dry_run: bool) -> dict[str, str]:
    if dry_run:
        return {"source": f"hf://{repo_id}/{filename}", "path": "<dry-run>"}
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub or run `python -m pip install -e '.[cinema]'`.") from exc
    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    return {"source": f"hf://{repo_id}/{filename}", "path": path}


def hf_download_to_dir(
    repo_id: str,
    filename: str,
    *,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, str]:
    if dry_run:
        return {"source": f"hf://{repo_id}/{filename}", "path": str(output_dir / filename)}
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub or run `python -m pip install -e '.[medsam2]'`.") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=output_dir)
    return {"source": f"hf://{repo_id}/{filename}", "path": path}


def url_download(url: str, *, output_path: Path, dry_run: bool, force: bool) -> dict[str, str]:
    if dry_run:
        return {"source": url, "path": str(output_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return {"source": url, "path": str(output_path), "status": "exists"}
    with urllib.request.urlopen(url) as response, output_path.open("wb") as f:
        f.write(response.read())
    return {"source": url, "path": str(output_path), "status": "downloaded"}


def main() -> None:
    args = parse_args()
    external_root = Path(args.external_root)
    medsam2_checkpoint_dir = (
        Path(args.medsam2_checkpoint_dir)
        if args.medsam2_checkpoint_dir is not None
        else external_root / "MedSAM2" / "checkpoints"
    )
    manifest: dict[str, list[dict[str, str]]] = {}

    if should_download("cinema", args.only):
        manifest["cinema"] = [
            hf_download(repo_id, filename, cache_dir=args.cinema_cache_dir, dry_run=args.dry_run)
            for repo_id, filename in CINEMA_FILES
        ]

    if should_download("medsam2", args.only):
        manifest["medsam2"] = [
            hf_download_to_dir(repo_id, filename, output_dir=medsam2_checkpoint_dir, dry_run=args.dry_run)
            for repo_id, filename in MEDSAM2_FILES
        ]

    if should_download("efficienttam", args.only):
        manifest["efficienttam"] = [
            hf_download_to_dir(repo_id, filename, output_dir=medsam2_checkpoint_dir, dry_run=args.dry_run)
            for repo_id, filename in EFFICIENTTAM_FILES
        ]

    if should_download("sam2-base", args.only):
        manifest["sam2-base"] = [
            url_download(
                url,
                output_path=medsam2_checkpoint_dir / filename,
                dry_run=args.dry_run,
                force=args.force,
            )
            for filename, url in SAM2_BASE_URLS
        ]

    if not args.dry_run:
        manifest_path = medsam2_checkpoint_dir / "download_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest"] = [{"source": "local", "path": str(manifest_path)}]

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

