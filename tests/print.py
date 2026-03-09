#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_mesh.py

给定一个 OBJ/PLY/STL 等网格文件，打印：
- 基本几何信息
- 水密/体信息
- 质量属性
- 各种 inertia / 主惯量信息

示例：
    python inspect_mesh.py path/to/model.obj
    python inspect_mesh.py path/to/model.obj --density 500
    python inspect_mesh.py path/to/model.obj --process
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def fmt_scalar(x: Any) -> str:
    """格式化标量输出。"""
    try:
        if x is None:
            return "None"
        return f"{float(x):.10g}"
    except Exception:
        return str(x)


def fmt_array(x: Any) -> str:
    """格式化数组输出。"""
    if x is None:
        return "None"
    arr = np.asarray(x)
    return np.array2string(arr, precision=8, suppress_small=False)


def print_kv(key: str, value: Any) -> None:
    """统一打印键值。"""
    print(f"{key:<32}: {value}")


def safe_get(name: str, fn):
    """安全获取属性，避免单个属性失败导致整个脚本退出。"""
    try:
        value = fn()
        print_kv(name, fmt_array(value) if isinstance(value, (np.ndarray, list, tuple)) else value)
    except Exception as e:
        print_kv(name, f"<ERROR> {type(e).__name__}: {e}")


def load_as_mesh(path: Path, process: bool) -> trimesh.Trimesh:
    """
    尽量加载为单个 Trimesh。
    如果读出来是 Scene，则把所有几何拼成一个 mesh。
    """
    obj = trimesh.load(path, process=process)

    if isinstance(obj, trimesh.Trimesh):
        return obj

    if isinstance(obj, trimesh.Scene):
        if len(obj.geometry) == 0:
            raise ValueError("Loaded scene contains no geometry.")
        # 尽量保留场景里各几何的当前姿态
        try:
            mesh = obj.to_mesh()
            if isinstance(mesh, trimesh.Trimesh):
                return mesh
        except Exception:
            pass

        # 退化方案：直接拼接所有 geometry
        meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("Scene contains no Trimesh geometry.")
        return trimesh.util.concatenate(meshes)

    raise TypeError(f"Unsupported loaded type: {type(obj)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_path", type=Path, help="Path to obj/ply/stl/... mesh file")
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Density used for mass-related quantities. Default: 1.0",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Let trimesh process the mesh on load",
    )
    args = parser.parse_args()

    mesh_path = args.mesh_path.expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"File not found: {mesh_path}")

    mesh = load_as_mesh(mesh_path, process=args.process)

    # 设置密度，这样 mass / mass_properties 里的质量项有确定含义
    try:
        mesh.density = args.density
    except Exception:
        pass

    np.set_printoptions(precision=8, suppress=False)

    print("=" * 90)
    print("MESH INSPECTION")
    print("=" * 90)

    print_kv("file", str(mesh_path))
    print_kv("type", type(mesh).__name__)
    print_kv("process_on_load", args.process)
    print_kv("density", args.density)

    print("\n" + "-" * 90)
    print("BASIC TOPOLOGY / GEOMETRY")
    print("-" * 90)

    print_kv("vertices.shape", mesh.vertices.shape)
    print_kv("faces.shape", mesh.faces.shape)
    print_kv("dtype(vertices)", mesh.vertices.dtype)
    print_kv("dtype(faces)", mesh.faces.dtype)

    safe_get("bounds", lambda: mesh.bounds)
    safe_get("extents", lambda: mesh.extents)
    safe_get("bounding_box.volume", lambda: mesh.bounding_box.volume)
    safe_get("bounding_box.extents", lambda: mesh.bounding_box.extents)
    safe_get("area", lambda: mesh.area)
    safe_get("euler_number", lambda: mesh.euler_number)
    safe_get("body_count", lambda: mesh.body_count)
    safe_get("is_empty", lambda: mesh.is_empty)
    safe_get("is_convex", lambda: mesh.is_convex)

    print("\n" + "-" * 90)
    print("WATERTIGHT / VOLUME VALIDITY")
    print("-" * 90)

    safe_get("is_watertight", lambda: mesh.is_watertight)
    safe_get("is_winding_consistent", lambda: mesh.is_winding_consistent)
    safe_get("is_volume", lambda: mesh.is_volume)

    print("\n" + "-" * 90)
    print("CENTER / MASS PROPERTIES")
    print("-" * 90)

    safe_get("centroid", lambda: mesh.centroid)
    safe_get("center_mass", lambda: mesh.center_mass)
    safe_get("volume", lambda: mesh.volume)
    safe_get("mass", lambda: mesh.mass)
    safe_get("density(current)", lambda: getattr(mesh, "density", None))
    safe_get("mass_properties", lambda: mesh.mass_properties)

    print("\n" + "-" * 90)
    print("INERTIA")
    print("-" * 90)

    # 1) Trimesh 自带：在 center_mass 处、按当前笛卡尔轴表达
    safe_get("moment_inertia", lambda: mesh.moment_inertia)

    # 2) 在原点处的惯量张量
    safe_get(
        "moment_inertia_frame(I)",
        lambda: mesh.moment_inertia_frame(np.eye(4)),
    )

    # 3) 主惯量 / 主轴
    safe_get("principal_inertia_components", lambda: mesh.principal_inertia_components)
    safe_get("principal_inertia_vectors", lambda: mesh.principal_inertia_vectors)
    safe_get("principal_inertia_transform", lambda: mesh.principal_inertia_transform)

    # 4) 额外手动做一次特征分解，便于和 principal_* 对照
    try:
        I = np.asarray(mesh.moment_inertia, dtype=float)
        evals, evecs = np.linalg.eigh(I)
        print_kv("eigvals(moment_inertia)", fmt_array(evals))
        print_kv("eigvecs(moment_inertia)", fmt_array(evecs))
    except Exception as e:
        print_kv("eig(moment_inertia)", f"<ERROR> {type(e).__name__}: {e}")

    print("\n" + "-" * 90)
    print("CONVEX HULL (REFERENCE)")
    print("-" * 90)

    try:
        hull = mesh.convex_hull
        print_kv("convex_hull.vertices.shape", hull.vertices.shape)
        print_kv("convex_hull.faces.shape", hull.faces.shape)
        print_kv("convex_hull.is_watertight", hull.is_watertight)
        print_kv("convex_hull.volume", fmt_scalar(hull.volume))
        if hasattr(mesh, "volume"):
            try:
                ratio = float(mesh.volume) / float(hull.volume)
                print_kv("volume / convex_hull.volume", fmt_scalar(ratio))
            except Exception:
                pass
    except Exception as e:
        print_kv("convex_hull", f"<ERROR> {type(e).__name__}: {e}")

    print("\n" + "-" * 90)
    print("NOTES")
    print("-" * 90)
    if not mesh.is_watertight:
        print("[WARN] mesh.is_watertight == False")
        print("[WARN] volume / center_mass / mass / moment_inertia 可能不可靠。")
        print("[WARN] centroid 通常仍可参考。")
    if not mesh.is_volume:
        print("[WARN] mesh.is_volume == False")
        print("[WARN] 说明它不满足“有效体”的完整条件，可能是法向、绕序或水密性有问题。")

    print("=" * 90)


if __name__ == "__main__":
    main()