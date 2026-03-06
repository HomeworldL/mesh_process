#!/usr/bin/env python3
"""Validate ShapeNetCore denormalization quality in processed objects."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "assets" / "objects" / "processed" / "ShapeNetCore"
    p = argparse.ArgumentParser(
        description="Check that processed ShapeNetCore raw.obj matches model_normalized.json scale."
    )
    p.add_argument("--processed-root", type=Path, default=default_root)
    p.add_argument("--max-objects", type=int, default=0, help="0 means check all objects.")
    p.add_argument("--tol-rel-extent", type=float, default=0.05, help="Relative tolerance for bbox extent.")
    p.add_argument("--tol-abs-center", type=float, default=0.05, help="Absolute tolerance (meters) for bbox center.")
    p.add_argument("--warn-max-extent", type=float, default=10.0, help="Warn if any axis extent exceeds this (m).")
    p.add_argument("--warn-min-extent", type=float, default=1e-4, help="Warn if any axis extent below this (m).")
    return p.parse_args()


def obj_bounds(obj_path: Path) -> tuple[list[float], list[float]] | None:
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    found = False
    for line in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        toks = line.split()
        if len(toks) < 4:
            continue
        try:
            v = [float(toks[1]), float(toks[2]), float(toks[3])]
        except Exception:
            continue
        found = True
        for i in range(3):
            mins[i] = min(mins[i], v[i])
            maxs[i] = max(maxs[i], v[i])
    if not found:
        return None
    return mins, maxs


def main() -> None:
    args = parse_args()
    root = args.processed_root
    if not root.exists():
        raise FileNotFoundError(f"processed root not found: {root}")

    obj_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.max_objects > 0:
        obj_dirs = obj_dirs[: args.max_objects]

    checked = 0
    failed = 0
    warned = 0
    fail_examples: list[str] = []
    warn_examples: list[str] = []

    for obj_dir in obj_dirs:
        raw_obj = obj_dir / "raw.obj"
        meta_json = obj_dir / "model_normalized.json"
        if (not raw_obj.is_file()) or (not meta_json.is_file()):
            continue

        checked += 1
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            min_t = [float(x) for x in meta["min"]]
            max_t = [float(x) for x in meta["max"]]
            centroid_t = [float(x) for x in meta.get("centroid", [(min_t[i] + max_t[i]) * 0.5 for i in range(3)])]
        except Exception:
            failed += 1
            if len(fail_examples) < 20:
                fail_examples.append(f"{obj_dir.name}: invalid model_normalized.json")
            continue

        bounds = obj_bounds(raw_obj)
        if bounds is None:
            failed += 1
            if len(fail_examples) < 20:
                fail_examples.append(f"{obj_dir.name}: raw.obj has no vertices")
            continue
        min_o, max_o = bounds

        ext_o = [max_o[i] - min_o[i] for i in range(3)]
        ext_t = [max_t[i] - min_t[i] for i in range(3)]
        center_o = [(max_o[i] + min_o[i]) * 0.5 for i in range(3)]

        extent_ok = True
        center_ok = True
        for i in range(3):
            denom = max(abs(ext_t[i]), 1e-9)
            rel_err = abs(ext_o[i] - ext_t[i]) / denom
            if rel_err > float(args.tol_rel_extent):
                extent_ok = False
            if abs(center_o[i] - centroid_t[i]) > float(args.tol_abs_center):
                center_ok = False
            if ext_o[i] > float(args.warn_max_extent) or ext_o[i] < float(args.warn_min_extent):
                warned += 1
                if len(warn_examples) < 20:
                    warn_examples.append(
                        f"{obj_dir.name}: suspicious extent axis={i} value={ext_o[i]:.6g}m"
                    )

        if not (extent_ok and center_ok):
            failed += 1
            if len(fail_examples) < 20:
                fail_examples.append(
                    f"{obj_dir.name}: extent_ok={extent_ok}, center_ok={center_ok}, "
                    f"ext_obj={[round(x,6) for x in ext_o]}, ext_json={[round(x,6) for x in ext_t]}"
                )

    passed = checked - failed
    print(f"checked={checked}, passed={passed}, failed={failed}, warned={warned}")
    if fail_examples:
        print("---- fail examples ----")
        for x in fail_examples:
            print(x)
    if warn_examples:
        print("---- warn examples ----")
        for x in warn_examples:
            print(x)

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
