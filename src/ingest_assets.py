#!/usr/bin/env python3
"""Unified CLI for object source ingest stage.

This script is intentionally parallel to `process_meshes.py` and `build_object_descriptions.py`:
1) ingest_assets.py: download / organize / verify
2) process_meshes.py: mesh processing (inertia, manifold, decomposition, simplify, visual)
3) build_object_descriptions.py: URDF/MJCF export
"""

from __future__ import annotations

import argparse
from pathlib import Path

from asset_ingest.base import DownloadReport, IngestConfig, OrganizeReport
from asset_ingest.manifest import IngestManifest
from asset_ingest.registry import ADAPTERS, get_adapter


def config_from_args(args: argparse.Namespace, source_name: str) -> IngestConfig:
    repo_root = Path(args.repo_root).resolve()
    raw_root = (
        Path(args.raw_root).resolve()
        if args.raw_root is not None
        else repo_root / "assets" / "objects" / "raw"
    )
    processed_root = (
        Path(args.processed_root).resolve()
        if args.processed_root is not None
        else repo_root / "assets" / "objects" / "processed"
    )
    return IngestConfig(
        repo_root=repo_root,
        source_name=source_name,
        raw_root=raw_root,
        processed_root=processed_root,
        force=bool(args.force),
        workers=int(args.workers),
        sample_n=getattr(args, "sample_n", None),
        sample_seed=int(getattr(args, "sample_seed", 0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object source ingest CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["download", "organize", "verify"]:
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument(
            "--repo-root",
            type=Path,
            default=Path(__file__).resolve().parents[1],
        )
        p.add_argument("--raw-root", type=Path, default=None)
        p.add_argument("--processed-root", type=Path, default=None)
        p.add_argument("--source", type=str, required=True, choices=sorted(ADAPTERS.keys()))
        if name == "organize":
            p.add_argument(
                "--sample-n",
                type=int,
                default=None,
                help="Optional: only organize a deterministic random sample of N objects (used by Objaverse).",
            )
            p.add_argument(
                "--sample-seed",
                type=int,
                default=0,
                help="Random seed for --sample-n (default: 0).",
            )
        if name == "verify":
            p.add_argument("--check-paths", action="store_true")

    return parser.parse_args()


def manifest_path(cfg) -> Path:
    return cfg.source_processed_dir / "manifest.json"


def _print_list_block(title: str, values: list[str], *, max_items: int | None = None) -> None:
    print(f"{title}:")
    if not values:
        print("- none")
        return

    shown = values if max_items is None else values[:max_items]
    for value in shown:
        print(f"- {value}")
    remaining = len(values) - len(shown)
    if remaining > 0:
        print(f"- ... ({remaining} more)")


def print_download_report(report: DownloadReport) -> None:
    print(f"Source: {report.source}")
    print(f"Downloaded files: {len(report.downloaded_files)}")
    _print_list_block("Download outputs", report.downloaded_files)
    _print_list_block("Notes", report.notes)


def print_organize_report(report: OrganizeReport) -> None:
    print(f"Source: {report.source}")
    print(f"Organized objects: {report.organized_objects}")
    print(f"Failed items: {len(report.failed_items)}")
    print(f"Manifest: {report.manifest_path or 'none'}")
    print(f"Manifest errors: {len(report.manifest_errors)}")

    if report.failed_items:
        _print_list_block("Failed item preview", report.failed_items, max_items=20)
    if report.manifest_errors:
        _print_list_block("Manifest error preview", report.manifest_errors, max_items=20)
    _print_list_block("Notes", report.notes)


def main() -> None:
    args = parse_args()
    adapter = get_adapter(args.source)
    cfg = config_from_args(args, source_name=args.source)

    if args.command == "download":
        print_download_report(adapter.download(cfg))
        return

    if args.command == "organize":
        report = adapter.organize(cfg)
        print_organize_report(report)
        return

    if args.command == "verify":
        mpath = manifest_path(cfg)
        if not mpath.exists():
            raise FileNotFoundError(f"Manifest not found: {mpath}")

        manifest = IngestManifest.load(mpath)
        errs = adapter.validate_manifest(
            manifest,
            cfg,
            check_paths=bool(args.check_paths),
        )
        if errs:
            print("manifest validation errors:")
            for err in errs:
                print(f"- {err}")
            raise SystemExit(1)
        print("manifest validation passed")


if __name__ == "__main__":
    main()
