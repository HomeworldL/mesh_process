#!/usr/bin/env python3
"""Validate equivalence of two coacd transform paths.

Path A: transform full coacd mesh, then split into convex pieces.
Path B: split coacd mesh first, then transform each piece.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import trimesh

from process_meshes import compute_principal_transform


def _as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values()] if hasattr(loaded, "geometry") else []
        if not geoms:
            raise RuntimeError(f"empty scene: {path}")
        return trimesh.util.concatenate(geoms)
    return loaded


def _piece_stats(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    extents = []
    for m in meshes:
        b = np.asarray(m.bounds, dtype=float)
        c = 0.5 * (b[0] + b[1])
        e = b[1] - b[0]
        centroids.append(c)
        extents.append(e)
    return np.asarray(centroids, dtype=float), np.asarray(extents, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate coacd transform equivalence")
    p.add_argument("--dataset", required=True)
    p.add_argument("--object-id", required=True)
    p.add_argument("--mass", type=float, default=0.1)
    p.add_argument("--processed-root", type=Path, default=Path("assets/objects/processed"))
    args = p.parse_args()

    obj_dir = args.processed_root / args.dataset / args.object_id
    manifold_obj = obj_dir / "manifold.obj"
    coacd_obj = obj_dir / "coacd.obj"
    if not manifold_obj.exists() or not coacd_obj.exists():
        raise FileNotFoundError(f"missing manifold/coacd under {obj_dir}")

    info = compute_principal_transform(str(manifold_obj), mass_value=float(args.mass), verbose=False)
    T = np.asarray(info["world_to_aligned_principal_T"], dtype=float)

    coacd_mesh = _as_mesh(coacd_obj)
    pieces_raw = list(coacd_mesh.split())

    # Path A: transform full then split
    mesh_a = coacd_mesh.copy()
    mesh_a.apply_transform(T)
    pieces_a = list(mesh_a.split())

    # Path B: split then transform each
    pieces_b = []
    for m in pieces_raw:
        mm = m.copy()
        mm.apply_transform(T)
        pieces_b.append(mm)

    cat_a = trimesh.util.concatenate(pieces_a)
    cat_b = trimesh.util.concatenate(pieces_b)

    bounds_diff = float(np.max(np.abs(np.asarray(cat_a.bounds) - np.asarray(cat_b.bounds))))
    vol_diff = float(abs(float(cat_a.volume) - float(cat_b.volume)))

    ca, ea = _piece_stats(pieces_a)
    cb, eb = _piece_stats(pieces_b)
    n_match = min(len(pieces_a), len(pieces_b))
    centroid_max_diff = float(np.max(np.linalg.norm(ca[:n_match] - cb[:n_match], axis=1))) if n_match > 0 else 0.0
    extent_max_diff = float(np.max(np.abs(ea[:n_match] - eb[:n_match]))) if n_match > 0 else 0.0

    ok = (
        len(pieces_a) == len(pieces_b)
        and bounds_diff < 1e-8
        and vol_diff < 1e-8
        and centroid_max_diff < 1e-8
        and extent_max_diff < 1e-8
    )

    print(f"object={args.object_id}")
    print(f"num_pieces_a={len(pieces_a)} num_pieces_b={len(pieces_b)}")
    print(f"global_bounds_max_abs_diff={bounds_diff:.3e}")
    print(f"global_volume_abs_diff={vol_diff:.3e}")
    print(f"piece_centroid_max_l2_diff={centroid_max_diff:.3e}")
    print(f"piece_extent_max_abs_diff={extent_max_diff:.3e}")
    print(f"equivalent={ok}")


if __name__ == "__main__":
    main()
