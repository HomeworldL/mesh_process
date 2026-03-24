#!/usr/bin/env python3
"""Small debug checks kept for current mesh pipeline maintenance."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_meshes import compute_principal_transform  # noqa: E402


def parse_obj_stats(obj_path: Path) -> dict[str, Any]:
    stats = {
        "v": 0,
        "vt": 0,
        "vn": 0,
        "f": 0,
        "mtllib": [],
        "usemtl": [],
    }
    mtllib_re = re.compile(r"^\s*mtllib\s+(.+?)\s*$")
    usemtl_re = re.compile(r"^\s*usemtl\s+(.+?)\s*$")

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("v "):
                stats["v"] += 1
            elif s.startswith("vt "):
                stats["vt"] += 1
            elif s.startswith("vn "):
                stats["vn"] += 1
            elif s.startswith("f "):
                stats["f"] += 1
            else:
                m = mtllib_re.match(line)
                if m:
                    stats["mtllib"].append(m.group(1).strip())
                    continue
                m = usemtl_re.match(line)
                if m:
                    stats["usemtl"].append(m.group(1).strip())
    return stats


def parse_mtl_maps(mtl_path: Path) -> list[str]:
    maps: list[str] = []
    if not mtl_path.exists():
        return maps
    map_kd_re = re.compile(r"^\s*map_Kd\s+(.+?)\s*$")
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = map_kd_re.match(line)
            if m:
                maps.append(m.group(1).strip())
    return maps


def _as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path))
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values()] if hasattr(loaded, "geometry") else []
        if not geoms:
            raise RuntimeError(f"empty scene: {path}")
        if len(geoms) == 1:
            return geoms[0]
        return trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported loaded type: {type(loaded).__name__}")
    return loaded


def _fmt(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=8, suppress_small=False)
    return str(value)


def inspect_mesh(mesh_path: Path) -> None:
    mesh = _as_mesh(mesh_path)
    raw_stats = parse_obj_stats(mesh_path)

    print("[RAW OBJ STATS]")
    print(f"v={raw_stats['v']} vt={raw_stats['vt']} vn={raw_stats['vn']} f={raw_stats['f']}")
    print(f"mtllib={raw_stats['mtllib'] if raw_stats['mtllib'] else 'none'}")
    print(f"usemtl={raw_stats['usemtl'][:5] if raw_stats['usemtl'] else 'none'}")

    print("\n[GEOMETRY]")
    print(f"type={type(mesh).__name__}")
    print(f"vertices={len(mesh.vertices)} faces={len(mesh.faces)}")
    print(f"is_empty={mesh.is_empty}")
    print(f"is_watertight={mesh.is_watertight}")
    print(f"is_winding_consistent={mesh.is_winding_consistent}")
    print(f"bounds={_fmt(np.asarray(mesh.bounds, dtype=float))}")
    print(f"extents={_fmt(np.asarray(mesh.extents, dtype=float))}")

    print("\n[VISUAL]")
    visual = getattr(mesh, "visual", None)
    if visual is None:
        print("visual=None")
    else:
        print(f"visual_type={type(visual).__name__}")
        print(f"visual_kind={getattr(visual, 'kind', None)}")
        uv = getattr(visual, "uv", None)
        if uv is None:
            print("uv=None")
        else:
            uv = np.asarray(uv)
            print(f"uv_shape={uv.shape}")
            if uv.size > 0:
                print(f"uv_min={_fmt(np.nanmin(uv, axis=0))}")
                print(f"uv_max={_fmt(np.nanmax(uv, axis=0))}")
        material = getattr(visual, "material", None)
        if material is None:
            print("material=None")
        else:
            image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
            if image is None:
                print("texture_image=None")
            else:
                try:
                    print(f"texture_image_size={image.size}")
                except Exception as exc:
                    print(f"texture_image_size=<error {exc}>")

    if raw_stats["mtllib"]:
        print("\n[MTL REFERENCES]")
        for mtl_name in raw_stats["mtllib"]:
            mtl_path = mesh_path.parent / mtl_name
            print(f"mtl={mtl_path} exists={mtl_path.exists()}")
            for rel in parse_mtl_maps(mtl_path):
                tex_path = (mtl_path.parent / rel).resolve()
                print(f"  map_Kd={rel} resolved={tex_path} exists={tex_path.exists()}")

    print("\n[NORMALS]")
    face_normals = np.asarray(mesh.face_normals)
    print(f"face_normals_shape={face_normals.shape}")
    vertex_normals = np.asarray(mesh.vertex_normals)
    print(f"vertex_normals_shape={vertex_normals.shape}")
    if vertex_normals.size > 0:
        norms = np.linalg.norm(vertex_normals, axis=1)
        print(
            "vertex_normals_norm_stats="
            f"min={norms.min():.6f} max={norms.max():.6f} mean={norms.mean():.6f}"
        )


def piece_stats(meshes: list[trimesh.Trimesh]) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    extents = []
    for mesh in meshes:
        bounds = np.asarray(mesh.bounds, dtype=float)
        centroids.append(0.5 * (bounds[0] + bounds[1]))
        extents.append(bounds[1] - bounds[0])
    return np.asarray(centroids, dtype=float), np.asarray(extents, dtype=float)


def validate_coacd_transform_equivalence(
    dataset: str,
    object_id: str,
    mass: float,
    processed_root: Path,
) -> int:
    obj_dir = processed_root / dataset / object_id
    manifold_obj = obj_dir / "manifold.obj"
    coacd_obj = obj_dir / "coacd.obj"
    if not manifold_obj.exists() or not coacd_obj.exists():
        raise FileNotFoundError(f"missing manifold/coacd under {obj_dir}")

    info = compute_principal_transform(str(manifold_obj), mass_value=float(mass), verbose=False)
    transform = np.asarray(info["world_to_aligned_principal_T"], dtype=float)

    coacd_mesh = _as_mesh(coacd_obj)
    pieces_raw = list(coacd_mesh.split())

    mesh_a = coacd_mesh.copy()
    mesh_a.apply_transform(transform)
    pieces_a = list(mesh_a.split())

    pieces_b = []
    for mesh in pieces_raw:
        mm = mesh.copy()
        mm.apply_transform(transform)
        pieces_b.append(mm)

    cat_a = trimesh.util.concatenate(pieces_a)
    cat_b = trimesh.util.concatenate(pieces_b)

    bounds_diff = float(np.max(np.abs(np.asarray(cat_a.bounds) - np.asarray(cat_b.bounds))))
    vol_diff = float(abs(float(cat_a.volume) - float(cat_b.volume)))

    ca, ea = piece_stats(pieces_a)
    cb, eb = piece_stats(pieces_b)
    n_match = min(len(pieces_a), len(pieces_b))
    centroid_max_diff = (
        float(np.max(np.linalg.norm(ca[:n_match] - cb[:n_match], axis=1)))
        if n_match > 0
        else 0.0
    )
    extent_max_diff = float(np.max(np.abs(ea[:n_match] - eb[:n_match]))) if n_match > 0 else 0.0

    ok = (
        len(pieces_a) == len(pieces_b)
        and bounds_diff < 1e-8
        and vol_diff < 1e-8
        and centroid_max_diff < 1e-8
        and extent_max_diff < 1e-8
    )

    print(f"object={object_id}")
    print(f"num_pieces_a={len(pieces_a)} num_pieces_b={len(pieces_b)}")
    print(f"global_bounds_max_abs_diff={bounds_diff:.3e}")
    print(f"global_volume_abs_diff={vol_diff:.3e}")
    print(f"piece_centroid_max_l2_diff={centroid_max_diff:.3e}")
    print(f"piece_extent_max_abs_diff={extent_max_diff:.3e}")
    print(f"equivalent={ok}")
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug checks for the current mesh pipeline.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect-obj", help="Inspect one mesh/OBJ and its visual status.")
    p_inspect.add_argument("mesh", type=Path)

    p_coacd = sub.add_parser(
        "validate-coacd-transform",
        help="Check that transform-before-split equals split-before-transform for CoACD mesh pieces.",
    )
    p_coacd.add_argument("--dataset", required=True)
    p_coacd.add_argument("--object-id", required=True)
    p_coacd.add_argument("--mass", type=float, default=0.1)
    p_coacd.add_argument("--processed-root", type=Path, default=Path("assets/objects/processed"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "inspect-obj":
        inspect_mesh(args.mesh.expanduser().resolve())
        return
    if args.cmd == "validate-coacd-transform":
        raise SystemExit(
            validate_coacd_transform_equivalence(
                dataset=args.dataset,
                object_id=args.object_id,
                mass=float(args.mass),
                processed_root=args.processed_root,
            )
        )


if __name__ == "__main__":
    main()
