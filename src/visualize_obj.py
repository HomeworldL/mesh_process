"""Preview one OBJ from assets/objects/processed/<dataset> using trimesh."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one OBJ from a processed dataset."
    )
    parser.add_argument("--dataset", type=str, default="YCB", help="Dataset name under assets/objects/processed.")
    parser.add_argument(
        "--object-id",
        type=str,
        default=None,
        help="Object folder name to visualize (default: randomly pick one).",
    )
    parser.add_argument(
        "--mesh-type",
        type=str,
        choices=["raw", "manifold", "visual"],
        default="raw",
        help="Choose which canonical mesh to display.",
    )
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

    if args.mesh_type == "raw":
        target_name = "raw.obj"
    elif args.mesh_type == "manifold":
        target_name = "manifold.obj"
    else:
        target_name = "visual.obj"

    if args.object_id:
        obj_dir = dataset_dir / args.object_id
        if not obj_dir.exists() or not obj_dir.is_dir():
            raise FileNotFoundError(f"Object directory not found: {obj_dir}")
        chosen = obj_dir / target_name
        if not chosen.exists():
            raise FileNotFoundError(f"Mesh not found: {chosen}")
    else:
        candidates = list(dataset_dir.rglob(target_name))
        if not candidates:
            raise RuntimeError(f"No {target_name} files found in: {dataset_dir}")
        chosen = random.choice(candidates)

    print(f"[visualize] dataset={args.dataset}")
    print(f"[visualize] mesh_type={args.mesh_type}")
    print(f"[visualize] object_id={chosen.parent.name}")
    print(f"[visualize] selected={chosen}")

    scene_or_mesh = trimesh.load(chosen)
    if isinstance(scene_or_mesh, trimesh.Scene):
        geometry = getattr(scene_or_mesh, "geometry", {})
        if not geometry:
            raise RuntimeError(
                f"Loaded scene has no geometry: {chosen}. "
                "Please rerun stage-2 with --force to regenerate visual assets."
            )
    else:
        if getattr(scene_or_mesh, "vertices", None) is None or len(scene_or_mesh.vertices) == 0:
            raise RuntimeError(
                f"Loaded mesh has no geometry: {chosen}. "
                "Please rerun stage-2 with --force to regenerate visual assets."
            )
    try:
        bounds = scene_or_mesh.bounds
    except Exception as e:
        raise RuntimeError(f"Failed to read scene bounds for {chosen}: {e}") from e
    if bounds is None:
        raise RuntimeError(
            f"Scene bounds is None for {chosen}. "
            "Please rerun stage-2 with --force to regenerate visual assets."
        )
    scene_or_mesh.show()


if __name__ == "__main__":
    main()
