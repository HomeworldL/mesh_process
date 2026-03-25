"""Load one glb/gltf/obj file, print mesh stats, and optionally visualize it."""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one mesh file and print basic trimesh statistics."
    )
    parser.add_argument(
        "mesh_path",
        type=Path,
        help="Path to a glb/gltf/obj file.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open trimesh viewer after printing stats.",
    )
    return parser.parse_args()


def _as_mesh(scene_or_mesh: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Scene):
        scene_or_mesh = scene_or_mesh.to_geometry()
        if isinstance(scene_or_mesh, trimesh.Scene):
            geoms = list(getattr(scene_or_mesh, "geometry", {}).values())
            if not geoms:
                raise RuntimeError("loaded scene has no geometry")
            if len(geoms) == 1:
                mesh = geoms[0]
            else:
                mesh = trimesh.util.concatenate(geoms)
        else:
            mesh = scene_or_mesh
    else:
        mesh = scene_or_mesh
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("loaded asset is not a trimesh mesh")
    return mesh


def main() -> None:
    args = parse_args()
    mesh_path = args.mesh_path.resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh file not found: {mesh_path}")

    scene_or_mesh = trimesh.load(mesh_path)
    mesh = _as_mesh(scene_or_mesh)

    print(f"path={mesh_path}")
    print(f"type={type(scene_or_mesh).__name__}")
    print(f"vertices={len(mesh.vertices)}")
    print(f"faces={len(mesh.faces)}")
    print(f"bounds={mesh.bounds.tolist()}")
    print(f"extents={mesh.extents.tolist()}")
    print(f"max_extent={float(mesh.extents.max())}")
    print(f"is_watertight={mesh.is_watertight}")
    print(f"is_volume={mesh.is_volume}")
    print(f"euler_number={mesh.euler_number}")
    print(f"body_count={mesh.body_count}")
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = None
    if material is not None:
        image = (
            getattr(material, "image", None)
            or getattr(material, "baseColorTexture", None)
        )
    print(f"has_uv={uv is not None}")
    print(f"has_texture={uv is not None and image is not None}")

    if args.show:
        scene_or_mesh.show()


if __name__ == "__main__":
    main()
