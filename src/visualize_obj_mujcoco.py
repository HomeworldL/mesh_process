#!/usr/bin/env python3
"""Visualize one built MJCF object in MuJoCo viewer.

Default behavior: randomly pick one object XML from a dataset.
You can also pass --object-id to select a specific object folder.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import mujoco
from mujoco import viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one object MJCF with MuJoCo viewer.")
    parser.add_argument("--dataset", type=str, default="YCB", help="Dataset name under assets/objects/processed.")
    parser.add_argument("--object-id", type=str, default=None, help="Object folder name to visualize.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Processed root dir (default: <repo>/assets/objects/processed).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    processed_root = args.processed_root.resolve() if args.processed_root else repo_root / "assets" / "objects" / "processed"
    dataset_dir = processed_root / args.dataset

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if args.seed is not None:
        random.seed(args.seed)

    if args.object_id:
        obj_dir = dataset_dir / args.object_id
        if not obj_dir.exists() or not obj_dir.is_dir():
            raise FileNotFoundError(f"Object directory not found: {obj_dir}")
        xml_path = obj_dir / f"{args.object_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"MJCF not found: {xml_path}")
    else:
        candidates = list(dataset_dir.rglob("*.xml"))
        # Prefer object xml files named <object_id>.xml.
        preferred = [p for p in candidates if p.parent.name == p.stem]
        pool = preferred if preferred else candidates
        if not pool:
            raise RuntimeError(f"No XML files found in: {dataset_dir}")
        xml_path = random.choice(pool)

    print(f"[visualize_mujoco] dataset={args.dataset}")
    print(f"[visualize_mujoco] object_id={xml_path.parent.name}")
    print(f"[visualize_mujoco] selected={xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
