#!/usr/bin/env python3
"""
Stage-2 mesh processing for organized datasets under:
  assets/objects/processed/<dataset>/<object_id>/raw.obj

Pipeline per object:
  1) mesh_manifold_and_convex_decomp -> manifold.obj + coacd.obj
  2) principal-frame transform from manifold.obj (overwrite manifold.obj)
  3) transform raw.obj with same transform -> visual.obj
     - untextured input: simplify geometry with trimesh
     - textured input: keep geometry, only compress texture and rewrite visual sidecars
  4) transform coacd.obj with same transform -> meshes/*.obj
  5) mesh_simplify -> simplified.obj

All outputs are written in the same object folder as raw.obj.
"""

from __future__ import annotations

import argparse
import os
import shutil
import re
import json
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import trimesh
from trimesh.exchange.obj import export_obj
from trimesh.exchange.export import export_mesh

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise ImportError(
        "process_meshes.py requires pillow for visual texture compression. "
        "Install with: pip install pillow"
    ) from e

# -------------------------
# Regex patterns (robust)
# -------------------------
_re_v = re.compile(r"^(\s*)v\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)(.*)$")
_re_vn = re.compile(r"^(\s*)vn\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)(.*)$")
_re_vt = re.compile(r"^(\s*)vt\s+([-\d.eE+]+)\s+([-\d.eE+]+)(?:\s+([-\d.eE+]+))?(.*)$")
_re_f = re.compile(r"^\s*f\s+")
_re_mtllib = re.compile(r"^(\s*)mtllib\s+(.+)$", re.IGNORECASE)
_re_usemtl = re.compile(r"^\s*usemtl\s+(.+)$", re.IGNORECASE)

TEXTURED_INPUT_MTL = "textured.mtl"
TEXTURED_INPUT_PNG = "textured.png"
VISUAL_TEXTURED_MTL = "textured_visual.mtl"
VISUAL_TEXTURED_PNG = "textured_visual.png"


# -------------------------
# Helpers (copied/adapted from your provided code)
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def parse_first_usemtl(obj_path):
    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = _re_usemtl.match(line.strip())
                if m:
                    name = m.group(1).strip()
                    if name:
                        return name
    except Exception:
        pass
    return "material_0"


def parse_first_newmtl(mtl_path):
    if not os.path.exists(mtl_path):
        return None
    try:
        with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split(maxsplit=1)
                if len(parts) == 2 and parts[0].lower() == "newmtl":
                    name = parts[1].strip()
                    if name:
                        return name
    except Exception:
        pass
    return None


def rewrite_visual_mtl_with_texture_map(src_mtl, dst_mtl, remap):
    """Rewrite map_* and bump texture filenames in MTL using remap dict."""
    if not os.path.exists(src_mtl):
        return False
    out_lines = []
    rewritten = False
    with open(src_mtl, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                out_lines.append(line)
                continue
            parts = s.split()
            if len(parts) >= 2:
                key = parts[0].lower()
                if key.startswith("map_") or key == "bump":
                    tex = os.path.basename(parts[-1].strip())
                    if tex in remap:
                        parts[-1] = remap[tex]
                        leading = re.match(r"^\s*", line).group(0)
                        out_lines.append(f"{leading}{' '.join(parts)}\n")
                        rewritten = True
                        continue
            out_lines.append(line)
    with open(dst_mtl, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    return rewritten


def _flip_face_winding(line: str) -> str:
    leading = re.match(r"^\s*", line).group(0)
    stripped = line.strip()
    body = stripped[2:].strip()
    if not body:
        return line
    face_part, sep, comment = body.partition("#")
    verts = face_part.split()
    if len(verts) < 3:
        return line
    verts.reverse()
    rebuilt = f"{leading}f {' '.join(verts)}"
    if sep:
        rebuilt += f" #{comment.rstrip()}"
    return rebuilt.rstrip() + "\n"


def transform_obj_text(src_obj, dst_obj, T, mtllib_map=None, flip_faces=False, flip_normals=False):
    """
    Transform 'v' and 'vn' lines of src_obj into dst_obj using 4x4 transform T.
    Update 'mtllib' lines according to mtllib_map (original->basename).
    Optionally flip OBJ face winding and vertex normals.
    """
    R = T[:3, :3]
    try:
        with open(src_obj, "r", encoding="utf-8", errors="ignore") as fin:
            lines = fin.readlines()
    except Exception as e:
        raise RuntimeError(f"Failed reading OBJ {src_obj}: {e}")

    out_lines = []
    for line in lines:
        m = _re_v.match(line)
        if m:
            leading, sx, sy, sz, rest = m.groups()
            v = np.array([float(sx), float(sy), float(sz), 1.0], dtype=float)
            vt = T @ v
            out_lines.append(f"{leading}v {vt[0]:.9f} {vt[1]:.9f} {vt[2]:.9f}{rest}\n")
            continue
        m2 = _re_vn.match(line)
        if m2:
            leading, snx, sny, snz, rest = m2.groups()
            n = R @ np.array([float(snx), float(sny), float(snz)], dtype=float)
            if flip_normals:
                n = -n
            nn = np.linalg.norm(n)
            if nn > 0:
                n = n / nn
            out_lines.append(f"{leading}vn {n[0]:.9f} {n[1]:.9f} {n[2]:.9f}{rest}\n")
            continue
        if flip_faces and _re_f.match(line):
            out_lines.append(_flip_face_winding(line))
            continue
        m3 = _re_mtllib.match(line)
        if m3 and mtllib_map:
            leading, name = m3.groups()
            name = name.strip()
            newname = mtllib_map.get(name, os.path.basename(name))
            out_lines.append(f"{leading}mtllib {newname}\n")
            continue
        out_lines.append(line)

    ensure_dir(os.path.dirname(dst_obj))
    with open(dst_obj, "w", encoding="utf-8", errors="ignore") as fout:
        fout.writelines(out_lines)


# -------------------------
# Principal inertia frame alignment
# -------------------------


def _best_aligned_principal_frame(principal_axes):
    """
    Choose signed/permuted principal axes that stay close to original XYZ.
    This preserves principal-axis alignment while avoiding abrupt axis relabeling.
    Input `principal_axes` is expected to be a 3x3 matrix whose columns are
    principal axes expressed in WORLD coordinates (principal->world basis vectors).
    Returned `best_frame` keeps the same convention (principal->world).
    """
    base = np.asarray(principal_axes, dtype=float)
    best_frame = None
    best_perm = None
    best_score = -np.inf
    # Permutation without itertools keeps import surface small.
    perms = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    for perm in perms:
        p = base[:, perm]
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    s = np.diag([sx, sy, sz])
                    cand = p @ s
                    if np.linalg.det(cand) <= 0:
                        continue
                    # Equivalent to maximizing axis-wise cosine to original XYZ.
                    score = float(np.trace(cand))
                    if score > best_score:
                        best_score = score
                        best_frame = cand
                        best_perm = perm
    if best_frame is None:
        best_frame = base.copy()
        if np.linalg.det(best_frame) < 0:
            best_frame[:, -1] *= -1.0
        best_perm = (0, 1, 2)
    return best_frame, best_perm


def build_alignment_T(com, principal_axes, principal_moments=None):
    """
    Build homogeneous transform between WORLD and aligned-principal frames.

    Args:
        com: center of mass in WORLD coordinates.
        principal_axes: 3x3 axes matrix (principal->world; columns are principal
            basis vectors expressed in WORLD coordinates).

    Returns dict:
        - `world_to_aligned_principal_T`: 4x4 transform WORLD->aligned-principal.
        - `aligned_axes_principal_to_world`: 3x3 aligned axes (principal->world).
        - `perm`: index permutation applied to principal moments.
    """
    R, perm = _best_aligned_principal_frame(principal_axes)
    T = np.eye(4, dtype=float)
    # R is principal->world, so world->principal rotation is R^T.
    T[:3, :3] = R.T
    # x_principal = R^T * (x_world - com_world)
    T[:3, 3] = -R.T @ np.asarray(com, dtype=float)
    out = {
        "world_to_aligned_principal_T": T,
        "aligned_axes_principal_to_world": R,
        "perm": perm,
    }
    if principal_moments is not None:
        w = np.asarray(principal_moments, dtype=float)
        out["aligned_moments"] = w[list(perm)].tolist()
    return out


# -------------------------
# External tool wrappers (CoACD / ACVD)
# -------------------------
def mesh_manifold_and_convex_decomp(
    input_path,
    manifold_output_path,
    coacd_output_path,
    seed=0,
    quiet=False,
    timeout_sec=None,
):
    cmd = [
        "third_party/CoACD/build/main",
        "-i",
        str(input_path),
        "-o",
        str(coacd_output_path),
        "-ro",
        str(manifold_output_path),
        "-pm",
        "on",
        "--seed",
        str(int(seed)),
    ]
    print("  [CoACD] running:", " ".join(cmd) if not quiet else "coacd (quiet mode)")
    run_kwargs = {}
    if quiet:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.DEVNULL
    try:
        proc = subprocess.run(cmd, timeout=timeout_sec, check=False, **run_kwargs)
        ret = int(proc.returncode)
    except subprocess.TimeoutExpired:
        ret = 124
    return ret


def describe_subprocess_returncode(tool_name: str, ret: int, timeout_sec=None) -> str:
    if int(ret) == 124:
        if timeout_sec:
            return f"{tool_name} timed out after {float(timeout_sec):.1f}s (ret=124)"
        return f"{tool_name} timed out (ret=124)"
    return f"{tool_name} failed (ret={int(ret)})"


def mesh_simplify(input_path, output_path, vert_num=4000, gradation=0.5, quiet=False, timeout_sec=None):
    out_dir = os.path.dirname(output_path)
    out_basename = os.path.basename(output_path)
    cmd = [
        "third_party/ACVD/bin/ACVD",
        str(input_path),
        str(int(vert_num)),
        str(float(gradation)),
        "-o",
        str(out_dir + os.sep),
        "-of",
        str(out_basename),
        "-m",
        "1",
    ]
    print("  [ACVD] running:", " ".join(cmd) if not quiet else "acvd (quiet mode)")
    run_kwargs = {}
    if quiet:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.DEVNULL
    try:
        proc = subprocess.run(cmd, timeout=timeout_sec, check=False, **run_kwargs)
        ret = int(proc.returncode)
    except subprocess.TimeoutExpired:
        ret = 124
    # ACVD may create 'smooth_filename' intermediate; attempt to remove if present
    smooth_p = os.path.join(out_dir, f"smooth_{out_basename}")
    if os.path.exists(smooth_p):
        try:
            os.remove(smooth_p)
        except Exception:
            pass
    return ret


def simplify_visual_obj_with_trimesh(src_obj, dst_obj, target_faces, decimals=6):
    """Simplify untextured visual mesh with trimesh and export OBJ without `vn`/`vt`."""
    try:
        mesh = trimesh.load(str(src_obj))
    except Exception as e:
        raise RuntimeError(f"failed to load visual source OBJ {src_obj}: {e}") from e

    if isinstance(mesh, trimesh.Scene):
        if not getattr(mesh, "geometry", None):
            raise RuntimeError(f"visual source OBJ loaded as empty scene: {src_obj}")
        try:
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        except Exception as e:
            raise RuntimeError(f"failed to concatenate visual scene geometry {src_obj}: {e}") from e

    try:
        face_num = int(len(mesh.faces))
    except Exception:
        face_num = int(target_faces) + 1
    if face_num > int(target_faces):
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=int(target_faces))
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "trimesh visual decimation requires the optional package "
                "`fast_simplification`; install it to process untextured visual meshes"
            ) from e
        except Exception as e:
            raise RuntimeError(f"trimesh visual decimation failed: {e}") from e

    try:
        obj_text = export_obj(
            mesh,
            include_normals=False,
            include_texture=False,
            digits=int(decimals),
        )
    except Exception as e:
        raise RuntimeError(f"failed to export trimesh visual OBJ {dst_obj}: {e}") from e

    try:
        with open(dst_obj, "w", encoding="utf-8", errors="ignore") as fout:
            fout.write(obj_text)
    except Exception as e:
        raise RuntimeError(f"failed writing visual OBJ {dst_obj}: {e}") from e


def mesh_compress_texture(src_tex, dst_tex, max_size=1024):
    with Image.open(src_tex) as img:
        img = img.convert("RGBA") if img.mode in ("P", "LA") else img
        w, h = img.size
        scale = min(1.0, float(max_size) / float(max(w, h)))
        if scale < 1.0:
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            img = img.resize((nw, nh), resample)

        img.save(dst_tex, format="PNG", optimize=True, compress_level=9)
        return {"texture_mode": "png", "texture_path": dst_tex}


def resolve_textured_input(obj_dir: Path) -> tuple[Path, Path] | None:
    mtl = obj_dir / TEXTURED_INPUT_MTL
    tex = obj_dir / TEXTURED_INPUT_PNG
    if (
        mtl.is_file()
        and mtl.stat().st_size > 0
        and tex.is_file()
        and tex.stat().st_size > 0
    ):
        return mtl, tex
    return None


def resolve_visual_texture_output(obj_dir: Path) -> Path | None:
    path = obj_dir / VISUAL_TEXTURED_PNG
    return path if path.is_file() else None


def resolve_visual_mtl_output(obj_dir: Path) -> Path | None:
    path = obj_dir / VISUAL_TEXTURED_MTL
    return path if path.is_file() else None


def build_textured_visual(obj_dir: Path, src_obj: Path, args):
    visual_obj = obj_dir / "visual.obj"
    textured_input = resolve_textured_input(obj_dir)
    if textured_input is None:
        raise RuntimeError(f"missing textured input assets under {obj_dir}")
    src_mtl, src_tex = textured_input
    visual_mtl = obj_dir / VISUAL_TEXTURED_MTL
    visual_tex = obj_dir / VISUAL_TEXTURED_PNG
    visual_mtl.unlink(missing_ok=True)
    visual_tex.unlink(missing_ok=True)

    texture_info = mesh_compress_texture(
        str(src_tex),
        str(visual_tex),
        max_size=int(args.visual_texture_max_size),
    )
    produced_visual_texture = Path(texture_info["texture_path"])
    remap = {src_tex.name: produced_visual_texture.name}

    material_name = parse_first_usemtl(str(src_obj)) or parse_first_newmtl(str(src_mtl)) or "material_0"
    rewritten = rewrite_visual_mtl_with_texture_map(str(src_mtl), str(visual_mtl), remap)
    if rewritten:
        first_newmtl = parse_first_newmtl(str(visual_mtl))
        if first_newmtl:
            material_name = first_newmtl
    else:
        with open(visual_mtl, "w", encoding="utf-8") as f:
            f.write(f"newmtl {material_name}\n")
            f.write("# shader_type beckmann\n")
            f.write(f"map_Kd {produced_visual_texture.name}\n")

    transform_obj_text(
        str(src_obj),
        str(visual_obj),
        np.eye(4, dtype=float),
        mtllib_map={TEXTURED_INPUT_MTL: VISUAL_TEXTURED_MTL},
        flip_faces=False,
        flip_normals=False,
    )

    return {
        "visual_obj": str(visual_obj),
        "visual_mtl": str(visual_mtl),
        "visual_texture": str(produced_visual_texture),
        "decimated_with": "none",
        "decimate_note": None,
        "texture_mode": texture_info.get("texture_mode"),
    }


def build_untextured_visual(obj_dir: Path, src_obj: Path, args):
    visual_obj = obj_dir / "visual.obj"
    simplify_visual_obj_with_trimesh(
        src_obj=str(src_obj),
        dst_obj=str(visual_obj),
        target_faces=int(args.visual_target_faces),
        decimals=int(args.visual_obj_decimals),
    )
    (obj_dir / VISUAL_TEXTURED_MTL).unlink(missing_ok=True)
    (obj_dir / VISUAL_TEXTURED_PNG).unlink(missing_ok=True)
    return {
        "visual_obj": str(visual_obj),
        "visual_mtl": None,
        "visual_texture": None,
        "decimated_with": "trimesh",
        "decimate_note": None,
        "texture_mode": "none",
    }


def mesh_visual(obj_dir, args, src_obj: str | Path | None = None):
    """
    Build compressed visual assets for MuJoCo loading speed:
      source OBJ (default raw.obj) -> visual.obj
      textured source keeps geometry and only rewrites visual sidecars.
    This function never overwrites raw.obj.
    """
    obj_dir = Path(obj_dir)
    if src_obj is None:
        src_obj = obj_dir / "raw.obj"
    else:
        src_obj = Path(src_obj)
    if not src_obj.exists():
        raise RuntimeError(f"source obj missing: {src_obj}")

    if resolve_textured_input(obj_dir) is not None:
        return build_textured_visual(obj_dir=obj_dir, src_obj=src_obj, args=args)
    return build_untextured_visual(obj_dir=obj_dir, src_obj=src_obj, args=args)


def compute_principal_transform(
    src_obj,
    mass_value=0.1,
    verbose=False,
):
    """
    Compute principal-frame transform metadata from input OBJ.

    Coordinate convention:
    - `principal_inertia_transform` from trimesh is WORLD->PRINCIPAL.
    - We extract principal axes in WORLD (principal->world), then apply our
      signed/permuted alignment policy (still principal->world).
    - Final returned transform is WORLD->ALIGNED_PRINCIPAL.
    """
    # load mesh using trimesh (no processing)
    try:
        mesh = trimesh.load(src_obj)
        if isinstance(mesh, trimesh.Scene):
            mesh = (
                trimesh.util.concatenate([g for g in mesh.geometry.values()])
                if hasattr(mesh, "geometry")
                else mesh
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load mesh {src_obj}: {e}")

    mass_value = float(mass_value)
    try:
        vol = float(mesh.volume)
    except Exception:
        vol = np.nan

    # Repair negative orientation before principal inertia computation.
    flip_faces = bool(
        np.isfinite(vol) and vol < 0.0
    )
    if flip_faces:
        try:
            mesh.invert()
        except Exception:
            pass
        try:
            vol = float(mesh.volume)
        except Exception:
            vol = np.nan

    if not np.isfinite(vol) or abs(vol) <= 1e-18:
        raise RuntimeError("invalid mesh volume for principal inertia computation")
    # Keep inertia scale consistent with manifest mass.
    mesh.density = float(mass_value) / abs(float(vol))

    try:
        I_world = np.asarray(mesh.moment_inertia, dtype=float)
    except Exception as e:
        raise RuntimeError(f"failed to read moment_inertia: {e}") from e
    I_world = 0.5 * (I_world + I_world.T)

    try:
        pit = np.asarray(mesh.principal_inertia_transform, dtype=float)
    except Exception as e:
        raise RuntimeError(f"failed to read principal_inertia_transform: {e}") from e
    if pit.shape != (4, 4) or not np.all(np.isfinite(pit)):
        raise RuntimeError("principal_inertia_transform is invalid")

    try:
        principal_components = np.asarray(mesh.principal_inertia_components, dtype=float)
    except Exception as e:
        raise RuntimeError(f"failed to read principal_inertia_components: {e}") from e
    if principal_components.shape != (3,) or not np.all(np.isfinite(principal_components)):
        raise RuntimeError("principal_inertia_components is invalid")

    # Inertia tensor expressed in the principal frame (should be nearly diagonal).
    try:
        I_principal = np.asarray(mesh.moment_inertia_frame(pit), dtype=float)
    except Exception as e:
        raise RuntimeError(f"failed to compute moment_inertia_frame: {e}") from e
    I_principal = 0.5 * (I_principal + I_principal.T)

    try:
        com = np.asarray(mesh.center_mass, dtype=float)
    except Exception as e:
        raise RuntimeError(f"failed to read center_mass: {e}") from e

    # trimesh principal_inertia_transform is WORLD->PRINCIPAL.
    R_world_to_principal = pit[:3, :3]
    # Convert to principal->world axes matrix.
    principal_axes_world = R_world_to_principal.T
    align = build_alignment_T(
        com,
        principal_axes_world,
        principal_moments=principal_components,
    )
    world_to_aligned_principal_T = align["world_to_aligned_principal_T"]

    if verbose:
        print(f"    volume={vol:.6g} mass_used={mass_value:.6g}")
        print(f"    com={np.round(com,6).tolist()}")
        print(
            f"    princ_moments (ordered, alignment-permuted): {np.round(np.asarray(align.get('aligned_moments', principal_components), dtype=float),9).tolist()}"
        )
        print(f"    principal_inertia_transform_rotation_trace={float(np.trace(R_world_to_principal)):.6g}")
        print(f"    moment_inertia_frame_diag={np.round(np.diag(I_principal),9).tolist()}")
        print(f"    orientation_fix_applied={flip_faces}")

    return {
        "world_to_aligned_principal_T": world_to_aligned_principal_T.tolist(),
        "flip_faces": bool(flip_faces),
        "center_of_mass_world": com.tolist(),
        "principal_moments_aligned": align.get("aligned_moments", principal_components.tolist()),
        "principal_axes_principal_to_world_aligned": align["aligned_axes_principal_to_world"].tolist(),
    }


# -------------------------
# Dataset-level processing
# -------------------------
def load_manifest(dataset_root: Path) -> dict:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mass_map(manifest: dict) -> tuple[dict[str, float | None], float]:
    mass_map: dict[str, float | None] = {}
    default_mass = float(manifest.get("summary", {}).get("default_mass_kg", 0.1) or 0.1)
    for obj in manifest.get("objects", []):
        oid = obj.get("object_id")
        mass = obj.get("mass_kg")
        if not isinstance(oid, str):
            continue
        if isinstance(mass, (int, float)):
            mass_map[oid] = float(mass)
        else:
            mass_map[oid] = None
    return mass_map, default_mass


def _convex_pieces_exist(meshes_dir: Path) -> bool:
    return any(meshes_dir.glob("coacd_convex_piece_*.obj"))


def _validate_coacd_convex_piece(
    mesh: trimesh.Trimesh,
    min_volume: float,
    min_extent: float,
) -> tuple[bool, str]:
    if bool(mesh.is_empty):
        return False, "empty mesh"

    try:
        extents = np.asarray(mesh.bounding_box.extents, dtype=float)
    except Exception as e:
        return False, f"failed to read bounding box extents: {e}"
    if extents.shape != (3,) or not np.all(np.isfinite(extents)):
        return False, "invalid bounding box extents"
    if float(np.max(extents)) <= float(min_extent):
        return False, f"bbox max extent too small ({float(np.max(extents)):.6g} <= {float(min_extent):.6g})"

    try:
        vol = float(mesh.volume)
    except Exception as e:
        return False, f"failed to read volume: {e}"
    if not np.isfinite(vol):
        return False, "non-finite volume"
    if abs(vol) <= float(min_volume):
        return False, f"volume too small ({abs(vol):.6g} <= {float(min_volume):.6g})"

    return True, ""


def _visual_outputs_exist(obj_dir: Path) -> bool:
    visual_obj = obj_dir / "visual.obj"
    visual_mtl = resolve_visual_mtl_output(obj_dir)
    if (not visual_obj.exists()) or visual_obj.stat().st_size <= 0:
        return False
    # For textured objects a rewritten visual MTL and compressed visual texture are expected.
    if resolve_textured_input(obj_dir) is not None:
        visual_tex = resolve_visual_texture_output(obj_dir)
        has_visual_tex = visual_tex is not None and visual_tex.stat().st_size > 0
        return visual_mtl is not None and visual_mtl.stat().st_size > 0 and has_visual_tex
    # For non-textured objects only visual.obj is required.
    return True


def validate_manifold_mesh(manifold_obj: Path) -> tuple[bool, str | None]:
    if (not manifold_obj.exists()) or manifold_obj.stat().st_size <= 0:
        return False, f"manifold.obj missing or empty: {manifold_obj}"
    try:
        loaded = trimesh.load(str(manifold_obj))
    except Exception as e:
        return False, f"failed to load manifold.obj: {e}"
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values()] if hasattr(loaded, "geometry") else []
        if not geoms:
            return False, "manifold.obj loaded as empty scene"
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = loaded
    try:
        if not bool(mesh.is_watertight):
            return False, "manifold.obj is not watertight"
    except Exception as e:
        return False, f"failed to evaluate watertight: {e}"
    try:
        if not bool(mesh.is_volume):
            return False, "manifold.obj is not volume"
    except Exception as e:
        return False, f"failed to evaluate is_volume: {e}"
    return True, None


def validate_visual_obj_loadable(obj_dir: Path) -> tuple[bool, str | None]:
    visual_obj = obj_dir / "visual.obj"
    if (not visual_obj.exists()) or visual_obj.stat().st_size <= 0:
        return False, f"visual.obj missing or empty: {visual_obj}"
    try:
        loaded = trimesh.load(str(visual_obj))
    except Exception as e:
        return False, f"failed to load visual.obj: {e}"
    if isinstance(loaded, trimesh.Scene):
        geoms = getattr(loaded, "geometry", {})
        if not geoms:
            return False, "visual.obj loaded as empty scene (no geometry)"
        scene = loaded
    else:
        if getattr(loaded, "vertices", None) is None or len(loaded.vertices) == 0:
            return False, "visual.obj loaded as empty mesh"
        scene = loaded
    try:
        bounds = scene.bounds
    except Exception as e:
        return False, f"failed to read visual bounds: {e}"
    if bounds is None:
        return False, "visual scene bounds is None"
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or (not np.all(np.isfinite(bounds))):
        return False, f"visual scene bounds invalid: shape={bounds.shape}"
    return True, None


def detect_visual_needs_flip(
    manifold_obj: Path,
    visual_obj: Path,
    *,
    max_face_samples: int = 2048,
) -> tuple[bool, str]:
    """
    Compare visual face normals against nearest manifold face normals.
    Returns (needs_flip, debug_reason).
    """
    try:
        manifold_mesh = trimesh.load(str(manifold_obj))
        visual_mesh = trimesh.load(str(visual_obj))
    except Exception as e:
        return False, f"load failed: {e}"

    if isinstance(manifold_mesh, trimesh.Scene):
        if not getattr(manifold_mesh, "geometry", None):
            return False, "manifold loaded as empty scene"
        manifold_mesh = trimesh.util.concatenate(tuple(manifold_mesh.geometry.values()))
    if isinstance(visual_mesh, trimesh.Scene):
        if not getattr(visual_mesh, "geometry", None):
            return False, "visual loaded as empty scene"
        visual_mesh = trimesh.util.concatenate(tuple(visual_mesh.geometry.values()))

    if len(manifold_mesh.faces) == 0 or len(visual_mesh.faces) == 0:
        return False, "empty face set"

    try:
        visual_centers = np.asarray(visual_mesh.triangles_center, dtype=float)
        visual_normals = np.asarray(visual_mesh.face_normals, dtype=float)
        if len(visual_centers) > int(max_face_samples):
            sample_idx = np.linspace(
                0,
                len(visual_centers) - 1,
                num=int(max_face_samples),
                dtype=int,
            )
            visual_centers = visual_centers[sample_idx]
            visual_normals = visual_normals[sample_idx]
        _, _, triangle_id = trimesh.proximity.closest_point(manifold_mesh, visual_centers)
        ref_normals = np.asarray(manifold_mesh.face_normals, dtype=float)[np.asarray(triangle_id, dtype=int)]
    except Exception as e:
        return False, f"closest-point comparison failed: {e}"

    valid = (
        np.all(np.isfinite(visual_normals), axis=1)
        & np.all(np.isfinite(ref_normals), axis=1)
    )
    if not np.any(valid):
        return False, "no valid normal pairs"

    dots = np.einsum("ij,ij->i", visual_normals[valid], ref_normals[valid])
    negative_ratio = float((dots < 0.0).mean())
    positive_ratio = float((dots > 0.0).mean())
    needs_flip = negative_ratio > 0.5 and negative_ratio > positive_ratio
    reason = (
        f"normal agreement: compared={len(dots)}, sampled_from={len(visual_mesh.faces)}, "
        f"positive_ratio={positive_ratio:.3f}, negative_ratio={negative_ratio:.3f}"
    )
    return needs_flip, reason


def process_object(
    idx: int,
    total: int,
    object_id: str,
    obj_dir: Path,
    mass_value: float,
    process_cfg: dict,
) -> dict:
    print(f"[{idx}/{total}] {object_id}")
    src_obj = obj_dir / "raw.obj"
    dst_manifold_obj = obj_dir / "manifold.obj"
    dst_coacd_obj = obj_dir / "coacd.obj"
    dst_simplified_obj = obj_dir / "simplified.obj"
    tmp_raw_aligned_obj = obj_dir / "raw.aligned.tmp.obj"
    tmp_manifold_aligned_obj = obj_dir / "manifold.aligned.tmp.obj"
    tmp_coacd_aligned_obj = obj_dir / "coacd.aligned.tmp.obj"
    tmp_visual_flipped_obj = obj_dir / "visual.flipped.tmp.obj"
    meshes_dir = obj_dir / "meshes"
    ensure_dir(str(meshes_dir))

    rec = {
        "status": "failed",
        "error": None,
        "center_of_mass": None,
        "principal_moments": None,
        "principal_axes": None,
        "visual_obj_path": None,
        "visual_mtl_path": None,
        "visual_texture_path": None,
    }

    if not src_obj.exists():
        rec["error"] = f"{object_id}: raw.obj missing: {src_obj}"
        print(f"    ERROR [{object_id}]: {rec['error']}")
        return rec

    # Step 1: run CoACD once to produce manifold.obj and coacd.obj
    ran_joint_coacd = False
    try:
        need_joint_coacd = (
            bool(process_cfg["force"])
            or (not dst_manifold_obj.exists())
            or (not dst_coacd_obj.exists())
        )
        if need_joint_coacd:
            print("  running mesh_manifold_and_convex_decomp...")
            coacd_timeout_sec = (
                float(process_cfg["coacd_timeout_sec"])
                if process_cfg["coacd_timeout_sec"]
                else None
            )
            ret = mesh_manifold_and_convex_decomp(
                str(src_obj),
                str(dst_manifold_obj),
                str(dst_coacd_obj),
                seed=int(process_cfg["coacd_seed"]),
                quiet=bool(process_cfg["coacd_quiet"]),
                timeout_sec=coacd_timeout_sec,
            )
            if ret != 0 or (not dst_manifold_obj.exists()) or (not dst_coacd_obj.exists()):
                detail = describe_subprocess_returncode(
                    "CoACD manifold/convex",
                    ret,
                    timeout_sec=coacd_timeout_sec,
                )
                raise RuntimeError(
                    f"{detail}; manifold_exists={dst_manifold_obj.exists()} "
                    f"coacd_exists={dst_coacd_obj.exists()}"
                )
            ran_joint_coacd = True
        else:
            print("  manifold/coacd exist -> skip CoACD")
        ok, reason = validate_manifold_mesh(dst_manifold_obj)
        if not ok:
            raise RuntimeError(f"invalid manifold: {reason}")
    except Exception as e:
        rec["error"] = f"{object_id}: manifold failed: {e}"
        print(f"    ERROR [{object_id}]: {rec['error']}")
        return rec

    # Step 2: principal-frame transform from manifold, overwrite manifold.obj
    try:
        print("  computing principal frame from manifold...")
        info = compute_principal_transform(
            str(dst_manifold_obj),
            mass_value=mass_value,
            verbose=bool(process_cfg["verbose"]),
        )
        rec["center_of_mass"] = info["center_of_mass_world"]
        rec["principal_moments"] = info["principal_moments_aligned"]
        rec["principal_axes"] = info["principal_axes_principal_to_world_aligned"]
        world_to_aligned_principal_T = np.asarray(
            info["world_to_aligned_principal_T"], dtype=float
        )
        flip_faces = bool(info["flip_faces"])
        transform_obj_text(
            str(dst_manifold_obj),
            str(tmp_manifold_aligned_obj),
            world_to_aligned_principal_T,
            mtllib_map=None,
            flip_faces=flip_faces,
            flip_normals=flip_faces,
        )
        os.replace(str(tmp_manifold_aligned_obj), str(dst_manifold_obj))
        ok, reason = validate_manifold_mesh(dst_manifold_obj)
        if not ok:
            raise RuntimeError(f"transformed manifold invalid: {reason}")
    except Exception as e:
        rec["error"] = f"{object_id}: inertia failed: {e}"
        print(f"    ERROR [{object_id}]: {rec['error']}")
        return rec
    finally:
        tmp_manifold_aligned_obj.unlink(missing_ok=True)

    # Step 3: build visual.obj from transformed raw.obj.
    try:
        if _visual_outputs_exist(obj_dir) and not bool(process_cfg["force"]):
            print("  visual outputs exist -> skip")
        else:
            print("  transforming raw.obj and running mesh_visual...")
            transform_obj_text(
                str(src_obj),
                str(tmp_raw_aligned_obj),
                world_to_aligned_principal_T,
                mtllib_map=None,
                flip_faces=flip_faces,
                flip_normals=flip_faces,
            )
            mesh_visual(
                obj_dir=str(obj_dir),
                args=argparse.Namespace(**process_cfg),
                src_obj=str(tmp_raw_aligned_obj),
            )
        visual_obj = obj_dir / "visual.obj"
        visual_mtl = resolve_visual_mtl_output(obj_dir)
        visual_tex = resolve_visual_texture_output(obj_dir)
        needs_visual_flip, flip_reason = detect_visual_needs_flip(dst_manifold_obj, visual_obj)
        print(f"  visual orientation check: {flip_reason}")
        if needs_visual_flip:
            print("  flipping visual.obj orientation to match manifold...")
            transform_obj_text(
                str(visual_obj),
                str(tmp_visual_flipped_obj),
                np.eye(4, dtype=float),
                mtllib_map=None,
                flip_faces=True,
                flip_normals=True,
            )
            os.replace(str(tmp_visual_flipped_obj), str(visual_obj))
        rec["visual_obj_path"] = str(visual_obj) if visual_obj.exists() else None
        rec["visual_mtl_path"] = (
            str(visual_mtl)
            if (visual_mtl is not None and visual_mtl.exists())
            else None
        )
        rec["visual_texture_path"] = str(visual_tex) if visual_tex is not None else None
        ok, reason = validate_visual_obj_loadable(obj_dir)
        if not ok:
            raise RuntimeError(f"visual validation failed: {reason}")
    except Exception as e:
        rec["error"] = f"{object_id}: visual mesh failed: {e}"
        print(f"    ERROR [{object_id}]: {rec['error']}")
        return rec
    finally:
        tmp_raw_aligned_obj.unlink(missing_ok=True)
        tmp_visual_flipped_obj.unlink(missing_ok=True)

    # Step 4: transform coacd.obj to aligned principal frame and export pieces
    try:
        need_convex_export = bool(process_cfg["force"]) or (not _convex_pieces_exist(meshes_dir))
        if ran_joint_coacd or bool(process_cfg["force"]):
            print("  transforming coacd.obj to aligned principal frame...")
            transform_obj_text(
                str(dst_coacd_obj),
                str(tmp_coacd_aligned_obj),
                world_to_aligned_principal_T,
                mtllib_map=None,
                flip_faces=flip_faces,
                flip_normals=flip_faces,
            )
            os.replace(str(tmp_coacd_aligned_obj), str(dst_coacd_obj))
        if need_convex_export:
            print("  exporting convex pieces...")
            for old_piece in meshes_dir.glob("coacd_convex_piece_*.obj"):
                old_piece.unlink(missing_ok=True)
            convex_pieces = list(trimesh.load(str(dst_coacd_obj)).split())
            exported_piece_count = 0
            skipped_piece_count = 0
            for raw_idx, piece in enumerate(convex_pieces):
                ok_piece, piece_reason = _validate_coacd_convex_piece(
                    piece,
                    min_volume=float(process_cfg["coacd_piece_min_volume"]),
                    min_extent=float(process_cfg["coacd_piece_min_extent"]),
                )
                if not ok_piece:
                    skipped_piece_count += 1
                    print(f"    skip convex piece {raw_idx}: {piece_reason}")
                    continue
                piece_filepath = meshes_dir / f"coacd_convex_piece_{exported_piece_count}.obj"
                export_mesh(piece, str(piece_filepath))
                exported_piece_count += 1
            if exported_piece_count <= 0:
                raise RuntimeError(
                    "all convex pieces were filtered out by mesh validity checks"
                )
            if skipped_piece_count > 0:
                print(
                    f"    exported {exported_piece_count} convex pieces, "
                    f"skipped {skipped_piece_count} invalid pieces"
                )
        else:
            print("  convex pieces exist -> skip")
        wrls = obj_dir / "coacd.wrl"
        wrls.unlink(missing_ok=True)
    except Exception as e:
        rec["error"] = f"{object_id}: convex export failed: {e}"
        print(f"    ERROR [{object_id}]: {rec['error']}")
        return rec
    finally:
        tmp_coacd_aligned_obj.unlink(missing_ok=True)

    # Step 5: simplify (ACVD)
    try:
        if dst_simplified_obj.exists() and not bool(process_cfg["force"]):
            print("  simplified exists -> skip")
        else:
            print("  running mesh_simplify...")
            ret = mesh_simplify(
                str(dst_manifold_obj),
                str(dst_simplified_obj),
                vert_num=int(process_cfg["acvd_vertnum"]),
                gradation=float(process_cfg["acvd_gradation"]),
                quiet=bool(process_cfg["acvd_quiet"]),
                timeout_sec=float(process_cfg["acvd_timeout_sec"]) if process_cfg["acvd_timeout_sec"] else None,
            )
            if ret != 0 or (not dst_simplified_obj.exists()):
                raise RuntimeError(f"ACVD failed (ret={ret}); simplified.obj missing")
    except Exception as e:
        rec["error"] = f"{object_id}: ACVD failed: {e}"
        print(f"    ERROR [{object_id}]: {rec['error']}")
        return rec

    rec["status"] = "success"
    return rec


def process_object_task(task: tuple[int, int, str, str, float, dict]) -> tuple[str, dict]:
    idx, total, object_id, obj_dir_str, mass_value, process_cfg = task
    rec = process_object(
        idx=idx,
        total=total,
        object_id=object_id,
        obj_dir=Path(obj_dir_str),
        mass_value=mass_value,
        process_cfg=process_cfg,
    )
    return object_id, rec


def write_process_report(
    dataset_root: Path,
    manifest: dict,
    per_object: dict[str, dict],
    elapsed_seconds: float | None = None,
    processed_objects: int | None = None,
) -> Path:
    report = deepcopy(manifest)
    now = datetime.now(timezone.utc).isoformat()
    repo_root = Path(__file__).resolve().parents[1]

    num_success = 0
    num_failed = 0
    num_skipped = 0

    for obj in report.get("objects", []):
        oid = obj.get("object_id")
        rec = per_object.get(oid, {"status": "missing", "error": "object folder not processed"})
        status = rec.get("status", "missing")
        obj["process_status"] = status
        obj["process_error"] = rec.get("error")
        obj["center_of_mass"] = rec.get("center_of_mass")
        obj["principal_moments"] = rec.get("principal_moments")
        obj["principal_axes"] = rec.get("principal_axes")
        visual_obj_path = rec.get("visual_obj_path")
        visual_mtl_path = rec.get("visual_mtl_path")
        visual_texture_path = rec.get("visual_texture_path")
        if isinstance(visual_obj_path, str):
            visual_obj_path = os.path.relpath(visual_obj_path, str(repo_root)).replace(os.sep, "/")
        if isinstance(visual_mtl_path, str):
            visual_mtl_path = os.path.relpath(visual_mtl_path, str(repo_root)).replace(os.sep, "/")
        if isinstance(visual_texture_path, str):
            visual_texture_path = os.path.relpath(visual_texture_path, str(repo_root)).replace(os.sep, "/")
        obj["visual_obj_path"] = visual_obj_path
        obj["visual_mtl_path"] = visual_mtl_path
        obj["visual_texture_path"] = visual_texture_path

        if status == "success":
            num_success += 1
        else:
            num_failed += 1

    report["process_meshes"] = {
        "generated_at": now,
        "script": "src/process_meshes.py",
        "num_success": num_success,
        "num_failed": num_failed,
        "num_skipped": num_skipped,
        "num_total": len(report.get("objects", [])),
        "elapsed_seconds": (float(elapsed_seconds) if elapsed_seconds is not None else None),
        "avg_seconds_per_object": (
            float(elapsed_seconds) / float(processed_objects)
            if (elapsed_seconds is not None and processed_objects is not None and processed_objects > 0)
            else None
        ),
        "num_processed_objects": (int(processed_objects) if processed_objects is not None else None),
    }

    out_path = dataset_root / "manifest.process_meshes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return out_path


def process_dataset(dataset_root: Path, args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    manifest = load_manifest(dataset_root)
    mass_map, default_mass = build_mass_map(manifest)

    objects = manifest.get("objects", [])
    if not isinstance(objects, list) or not objects:
        print(f"No objects in manifest: {dataset_root / 'manifest.json'}")
        return

    print(f"Found {len(objects)} objects in manifest for {dataset_root.name}")
    fs_dirs = {p.name for p in dataset_root.iterdir() if p.is_dir()}
    manifest_ids = {
        o.get("object_id") for o in objects if isinstance(o, dict) and isinstance(o.get("object_id"), str)
    }
    extra_dirs = sorted([d for d in fs_dirs if d not in manifest_ids])
    if extra_dirs:
        print(f"Warning: {len(extra_dirs)} folders are not in manifest and will be ignored.")

    per_object: dict[str, dict] = {}
    process_cfg = {
        "force": bool(args.force),
        "verbose": bool(args.verbose),
        "coacd_quiet": bool(args.coacd_quiet),
        "acvd_quiet": bool(args.acvd_quiet),
        "coacd_seed": int(args.coacd_seed),
        "coacd_timeout_sec": (float(args.coacd_timeout_sec) if float(args.coacd_timeout_sec) > 0 else None),
        "acvd_timeout_sec": (float(args.acvd_timeout_sec) if float(args.acvd_timeout_sec) > 0 else None),
        "acvd_vertnum": int(args.acvd_vertnum),
        "acvd_gradation": float(args.acvd_gradation),
        "visual_target_faces": int(args.visual_target_faces),
        "visual_obj_decimals": int(args.visual_obj_decimals),
        "visual_texture_max_size": int(args.visual_texture_max_size),
        "coacd_piece_min_volume": float(args.coacd_piece_min_volume),
        "coacd_piece_min_extent": float(args.coacd_piece_min_extent),
    }
    tasks: list[tuple[int, int, str, str, float, dict]] = []

    for idx, obj in enumerate(objects, start=1):
        if not isinstance(obj, dict):
            print(f"[{idx}/{len(objects)}] invalid manifest entry -> skip")
            continue
        object_id = obj.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            print(f"[{idx}/{len(objects)}] invalid object_id -> skip")
            continue

        mesh_path = obj.get("mesh_path")
        obj_dir = dataset_root / object_id
        if isinstance(mesh_path, str) and mesh_path:
            mesh_p = Path(mesh_path)
            if not mesh_p.is_absolute():
                mesh_p = (Path(__file__).resolve().parents[1] / mesh_p).resolve()
            if mesh_p.name == "raw.obj":
                obj_dir = mesh_p.parent

        mass_value = mass_map.get(object_id)
        if mass_value is None:
            mass_value = default_mass

        tasks.append(
            (
                idx,
                len(objects),
                object_id,
                str(obj_dir),
                float(mass_value),
                process_cfg,
            )
        )

    workers = int(args.workers)
    if workers <= 1:
        for task in tasks:
            oid, rec = process_object_task(task)
            per_object[oid] = rec
    else:
        print(f"Running in parallel with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(process_object_task, t): t[2] for t in tasks}
            pool_broken = False
            for fut in as_completed(future_map):
                oid = future_map[fut]
                try:
                    ret_oid, rec = fut.result()
                    per_object[ret_oid] = rec
                except BrokenProcessPool as e:
                    print(
                        "ERROR: worker crashed while processing "
                        f"{oid}; likely native tool crash (CoACD/ACVD): {e}"
                    )
                    per_object[oid] = {
                        "status": "failed",
                        "error": f"worker crashed while processing {oid}; likely native tool crash (CoACD/ACVD): {e}",
                    }
                    pool_broken = True
                    break
                except Exception as e:
                    print(f"ERROR: worker failed on {oid}: {e}")
                    per_object[oid] = {"status": "failed", "error": f"worker crash: {e}"}
            if pool_broken:
                for pending_fut, pending_oid in future_map.items():
                    if pending_oid in per_object:
                        continue
                    per_object[pending_oid] = {
                        "status": "failed",
                        "error": (
                            "not processed because ProcessPoolExecutor became broken after "
                            f"worker crash near {oid}; likely native tool crash (CoACD/ACVD)"
                        ),
                    }

    elapsed = time.perf_counter() - t0
    report_path = write_process_report(
        dataset_root,
        manifest,
        per_object,
        elapsed_seconds=elapsed,
        processed_objects=len(tasks),
    )
    avg = elapsed / len(tasks) if len(tasks) > 0 else 0.0
    print(f"Elapsed: {elapsed:.3f}s total, {avg:.3f}s/object over {len(tasks)} objects")
    print(f"Wrote process report: {report_path}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    root_dir = Path(__file__).resolve().parents[1]
    processed_root = root_dir / "assets" / "objects" / "processed"

    try:
        from asset_ingest.registry import ADAPTERS

        dataset_choices = set(ADAPTERS.keys())
    except Exception:
        dataset_choices = set([
            "DexNet",
            "GraspNet",
            "HOPE",
            "KIT",
            "MSO",
            "Objaverse",
            "RealDex",
            "ShapeNetCore",
            "ShapeNetSem",
            "YCB",
        ])
    if processed_root.exists():
        dataset_choices.update([p.name for p in processed_root.iterdir() if p.is_dir()])
    dataset_choices = sorted(dataset_choices)

    p = argparse.ArgumentParser(
        description="Process meshes in-place for assets/objects/processed/<dataset>."
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=dataset_choices,
        required=True,
        help="Dataset name under assets/objects/processed/<dataset>",
    )
    p.add_argument("--force", action="store_true", help="Overwrite outputs")
    default_workers = min(8, max(1, (os.cpu_count() or 1) // 2))
    p.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Parallel workers (default: {default_workers}).",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    p.add_argument(
        "--coacd-quiet",
        dest="coacd_quiet",
        action="store_true",
        default=True,
        help="Run CoACD quietly (default: enabled).",
    )
    p.add_argument(
        "--acvd-quiet",
        dest="acvd_quiet",
        action="store_true",
        default=True,
        help="Run ACVD quietly (default: enabled).",
    )
    p.add_argument(
        "--coacd-seed",
        type=int,
        default=0,
        help="Fixed random seed passed to CoACD for reproducible decomposition (default: 0).",
    )
    p.add_argument(
        "--coacd-timeout-sec",
        type=float,
        default=180.0,
        help="Timeout for CoACD per object in seconds (default: 180). Set <=0 to disable.",
    )
    p.add_argument(
        "--acvd-timeout-sec",
        type=float,
        default=180.0,
        help="Timeout for ACVD per object in seconds (default: 180). Set <=0 to disable.",
    )
    p.add_argument(
        "--acvd-vertnum", type=int, default=2000, help="ACVD: target vertex number"
    )
    p.add_argument(
        "--acvd-gradation", type=float, default=1.5, help="ACVD: gradation parameter"
    )
    p.add_argument(
        "--visual-target-faces",
        type=int,
        default=20000,
        help="Visual mesh decimation target face count.",
    )
    p.add_argument(
        "--visual-obj-decimals",
        type=int,
        default=6,
        help="Decimal precision for visual.obj text slimming.",
    )
    p.add_argument(
        "--visual-texture-max-size",
        type=int,
        default=1024,
        help="Max texture width/height for visual texture compression.",
    )
    p.add_argument(
        "--coacd-piece-min-volume",
        type=float,
        default=1e-12,
        help="Skip exported CoACD convex pieces whose absolute volume is <= this threshold.",
    )
    p.add_argument(
        "--coacd-piece-min-extent",
        type=float,
        default=1e-6,
        help="Skip exported CoACD convex pieces whose max bbox extent is <= this threshold.",
    )
    args = p.parse_args()

    args.dataset_root = processed_root / args.dataset
    return args


def main() -> None:
    args = parse_args()
    process_dataset(args.dataset_root, args)


if __name__ == "__main__":
    main()
