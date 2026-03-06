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

from asset_ingest.base import BaseIngestAdapter
from asset_ingest.manifest import IngestManifest
from asset_ingest.registry import ADAPTERS, get_adapter


def parse_args() -> argparse.Namespace:
    common = BaseIngestAdapter.parse_common_args()

    parser = argparse.ArgumentParser(description="Object source ingest CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["download", "organize", "verify"]:
        p = sub.add_parser(name, parents=[common])
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


def main() -> None:
    args = parse_args()
    adapter = get_adapter(args.source)
    cfg = adapter.config_from_args(args, source_name=args.source)

    if args.command == "download":
        print(adapter.download(cfg))
        return

    if args.command == "organize":
        report = adapter.organize(cfg)
        print(report)
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
