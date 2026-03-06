#!/usr/bin/env python3
"""Stage-4: sample multi-view partial point clouds from built MJCF objects.

Output per object:
  assets/objects/processed/<dataset>/<object_id>/vision_data/pc_views.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def fibonacci_sphere_points(n: int, radius: float, center: np.ndarray) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    idx = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * idx + 1.0) / n
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * idx
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    pts = np.stack([x, y, z], axis=-1) * float(radius) + center[None, :]
    return pts.astype(np.float32)


def build_world_from_camera(pos: np.ndarray, lookat: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = lookat - pos
    nf = np.linalg.norm(forward)
    if nf < 1e-8:
        raise ValueError("camera position too close to lookat")
    forward = forward / nf

    right = np.cross(forward, up_hint)
    nr = np.linalg.norm(right)
    if nr < 1e-8:
        alt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, alt)
        nr = np.linalg.norm(right)
        if nr < 1e-8:
            raise ValueError("invalid camera up configuration")
    right = right / nr
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = right
    T[:3, 1] = up
    T[:3, 2] = forward
    T[:3, 3] = pos
    return T


def get_intrinsic(width: int, height: int, fovy_deg: float) -> tuple[float, float, float, float]:
    fovy = np.deg2rad(float(fovy_deg))
    fy = (height * 0.5) / np.tan(fovy * 0.5)
    fx = fy
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return float(fx), float(fy), float(cx), float(cy)


def depth_to_world_points(
    depth: np.ndarray,
    world_from_camera: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth: float,
) -> np.ndarray:
    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = depth.astype(np.float64)

    valid = np.isfinite(z) & (z > 1e-6) & (z < float(max_depth))
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    u = uu[valid]
    v = vv[valid]
    z = z[valid]

    # Camera coordinates: x right, y up, z forward.
    x_cam = (u - cx) * z / fx
    y_cam = -(v - cy) * z / fy
    cam_pts = np.stack([x_cam, y_cam, z, np.ones_like(z)], axis=-1)
    world_pts = (cam_pts @ world_from_camera.T)[:, :3]
    return world_pts.astype(np.float32)


def fixed_size_downsample(points: np.ndarray, point_num: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    if points.shape[0] == 0:
        return np.zeros((point_num, 3), dtype=np.float32), 0
    if points.shape[0] >= point_num:
        idx = rng.choice(points.shape[0], size=point_num, replace=False)
        out = points[idx]
        return out.astype(np.float32), point_num
    idx = rng.choice(points.shape[0], size=point_num, replace=True)
    out = points[idx]
    return out.astype(np.float32), points.shape[0]


def sample_one_object(
    xml_path: Path,
    out_npz: Path,
    view_num: int,
    point_num: int,
    radius: float,
    height: int,
    width: int,
    max_depth: float,
    seed: int,
) -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lookat = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    camera_pos = fibonacci_sphere_points(view_num, radius=radius, center=lookat)
    camera_ex = np.zeros((view_num, 4, 4), dtype=np.float32)
    camera_lookat = np.repeat(lookat[None, :], view_num, axis=0).astype(np.float32)

    fovy_deg = float(model.vis.global_.fovy)
    fx, fy, cx, cy = get_intrinsic(width, height, fovy_deg=fovy_deg)

    pcs = np.zeros((view_num, point_num, 3), dtype=np.float32)
    valid_num = np.zeros((view_num,), dtype=np.int32)

    rng = np.random.default_rng(seed)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = lookat

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i in range(view_num):
            pos = camera_pos[i].astype(np.float64)
            T_wc = build_world_from_camera(pos=pos, lookat=lookat, up_hint=up_hint)
            camera_ex[i] = T_wc.astype(np.float32)

            delta = pos - lookat
            xy = np.linalg.norm(delta[:2])
            cam.distance = float(np.linalg.norm(delta))
            cam.azimuth = float(np.degrees(np.arctan2(delta[1], delta[0])))
            cam.elevation = float(np.degrees(np.arctan2(delta[2], xy)))
            renderer.update_scene(data, camera=cam)

            renderer.enable_depth_rendering()
            depth = renderer.render()
            renderer.disable_depth_rendering()

            world_pts = depth_to_world_points(
                depth=depth,
                world_from_camera=T_wc,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                max_depth=max_depth,
            )
            pcs[i], valid_num[i] = fixed_size_downsample(world_pts, point_num=point_num, rng=rng)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        object_id=xml_path.stem,
        camera_intrinsic=np.array([fx, fy, cx, cy, float(width), float(height)], dtype=np.float32),
        camera_extrinsic=camera_ex,
        camera_position=camera_pos.astype(np.float32),
        camera_lookat=camera_lookat,
        point_cloud=pcs,
        valid_point_num=valid_num,
        rgb_path=np.array([], dtype=np.str_),
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    processed_root = repo_root / "assets" / "objects" / "processed"
    dataset_choices = sorted([p.name for p in processed_root.iterdir() if p.is_dir()]) if processed_root.exists() else []

    parser = argparse.ArgumentParser(description="Stage-4 multi-view point cloud sampling.")
    parser.add_argument("--dataset", type=str, required=True, choices=dataset_choices)
    parser.add_argument("--processed-root", type=Path, default=processed_root)
    parser.add_argument("--object-id", type=str, default=None, help="Comma-separated object ids.")
    parser.add_argument("--views", type=int, default=25, help="Number of viewpoints per object.")
    parser.add_argument("--points", type=int, default=4096, help="Point number per view.")
    parser.add_argument("--radius", type=float, default=0.8, help="Camera sampling radius around object center.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Overwrite existing pc_views.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.processed_root.resolve() / args.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    selected = None
    if args.object_id:
        selected = {x.strip() for x in args.object_id.split(",") if x.strip()}

    manifest_pm = dataset_dir / "manifest.process_meshes.json"
    if not manifest_pm.exists():
        raise FileNotFoundError(f"Missing {manifest_pm}. Run process_meshes.py first.")

    with open(manifest_pm, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    objects = manifest.get("objects", [])
    if not isinstance(objects, list):
        raise RuntimeError("Invalid manifest.process_meshes.json")

    total = 0
    done = 0
    skipped = 0
    failed = 0

    for rec in objects:
        if not isinstance(rec, dict):
            continue
        object_id = rec.get("object_id")
        if not isinstance(object_id, str):
            continue
        if selected is not None and object_id not in selected:
            continue
        total += 1

        if rec.get("process_status") != "success":
            print(f"[stage4] skip {object_id}: process_status={rec.get('process_status')}")
            skipped += 1
            continue

        obj_dir = dataset_dir / object_id
        xml_path = obj_dir / f"{object_id}.xml"
        if not xml_path.exists():
            print(f"[stage4] skip {object_id}: missing xml {xml_path}")
            skipped += 1
            continue

        out_npz = obj_dir / "vision_data" / "pc_views.npz"
        if out_npz.exists() and not args.force:
            print(f"[stage4] skip {object_id}: exists {out_npz}")
            skipped += 1
            continue

        try:
            sample_one_object(
                xml_path=xml_path,
                out_npz=out_npz,
                view_num=int(args.views),
                point_num=int(args.points),
                radius=float(args.radius),
                width=int(args.width),
                height=int(args.height),
                max_depth=float(args.max_depth),
                seed=int(args.seed),
            )
            done += 1
            print(f"[stage4] done {object_id} -> {out_npz}")
        except Exception as e:
            failed += 1
            print(f"[stage4] fail {object_id}: {e}")

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "total_selected": total,
                "done": done,
                "skipped": skipped,
                "failed": failed,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
