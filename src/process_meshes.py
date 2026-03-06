#!/usr/bin/env python3
"""
Stage-2 mesh processing for organized datasets under:
  assets/objects/processed/<dataset>/<object_id>/raw.obj

Pipeline per object:
  1) mesh_transform -> inertia.obj
  2) mesh_manifold_and_convex_decomp -> manifold.obj + coacd.obj + meshes/*.obj
  3) mesh_simplify -> simplified.obj
  4) mesh_visual -> visual.obj + visual.mtl + visual_texture_map.(jpg|png)

All outputs are written in the same object folder as raw.obj.
"""

from __future__ import annotations

import argparse
import os
import shutil
import re
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import trimesh
from trimesh.exchange.export import export_mesh
try:
    import pymeshlab  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "process_meshes.py requires pymeshlab. Install with: pip install pymeshlab"
    ) from e

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
_re_mtllib = re.compile(r"^(\s*)mtllib\s+(.+)$", re.IGNORECASE)
_re_usemtl = re.compile(r"^\s*usemtl\s+(.+)$", re.IGNORECASE)


# -------------------------
# Helpers (copied/adapted from your provided code)
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def parse_mtllibs(obj_path):
    libs = []
    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = _re_mtllib.match(line)
                if m:
                    libs.append(m.group(2).strip())
    except Exception:
        pass
    return libs


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


def parse_first_texture_from_mtl(mtl_path):
    if not os.path.exists(mtl_path):
        return None
    try:
        with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                lower = s.lower()
                if lower.startswith("map_kd "):
                    parts = s.split(maxsplit=1)
                    if len(parts) == 2:
                        return parts[1].strip()
    except Exception:
        pass
    return None


def copy_and_patch_mtl(src_obj, dst_obj_dir):
    """
    Copy mtls referenced by src_obj into dst_obj_dir.
    For each mtl, rewrite texture lines to use basenames and copy textures.
    Return mapping: original_mtllib_token_in_obj -> basename_written_in_dst
    """
    mapping = {}
    src_dir = os.path.dirname(src_obj)
    mtllibs = parse_mtllibs(src_obj)
    for lib in mtllibs:
        src_mtl = os.path.join(src_dir, lib)
        if not os.path.exists(src_mtl):
            src_mtl = os.path.join(src_dir, os.path.basename(lib))
            if not os.path.exists(src_mtl):
                continue

        dst_mtl_basename = os.path.basename(src_mtl)
        dst_mtl = os.path.join(dst_obj_dir, dst_mtl_basename)

        out_lines = []
        try:
            with open(src_mtl, "r", encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    stripped = line.strip()
                    # texture tokens to handle
                    if stripped.lower().startswith(
                        ("map_kd ", "map_ka ", "map_d ", "map_bump ", "bump ")
                    ):
                        parts = stripped.split(maxsplit=1)
                        if len(parts) > 1:
                            tex_rel = parts[1].strip()
                            src_tex = os.path.join(os.path.dirname(src_mtl), tex_rel)
                            if not os.path.exists(src_tex):
                                src_tex = os.path.join(src_dir, tex_rel)
                            if os.path.exists(src_tex):
                                dst_tex = os.path.join(
                                    dst_obj_dir, os.path.basename(src_tex)
                                )
                                try:
                                    shutil.copy2(src_tex, dst_tex)
                                except Exception:
                                    pass
                                out_lines.append(
                                    f"{parts[0]} {os.path.basename(src_tex)}\n"
                                )
                                continue
                            else:
                                out_lines.append(line)
                                continue
                        else:
                            out_lines.append(line)
                    else:
                        out_lines.append(line)
        except Exception:
            continue

        try:
            ensure_dir(dst_obj_dir)
            with open(dst_mtl, "w", encoding="utf-8", errors="ignore") as fout:
                fout.writelines(out_lines)
            mapping[lib] = dst_mtl_basename
        except Exception:
            pass

    return mapping


def transform_obj_text(src_obj, dst_obj, T, mtllib_map=None):
    """
    Transform 'v' and 'vn' lines of src_obj into dst_obj using 4x4 transform T.
    Update 'mtllib' lines according to mtllib_map (original->basename).
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
            nn = np.linalg.norm(n)
            if nn > 0:
                n = n / nn
            out_lines.append(f"{leading}vn {n[0]:.9f} {n[1]:.9f} {n[2]:.9f}{rest}\n")
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
# Principal inertia computation (adapted)
# -------------------------
def compute_principal(mesh, mass_override=None):
    """
    Return com (3,), principal_moments (3,), principal_axes (3x3 columns),
    mass (used), volume, and inertia matrix about COM.
    """
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values()] if hasattr(mesh, "geometry") else []
        if len(geoms) == 0:
            raise RuntimeError("Scene has no geometry")
        mesh = trimesh.util.concatenate(geoms)

    vol = getattr(mesh, "volume", None)
    if vol is None or not np.isfinite(vol) or vol == 0:
        ext = mesh.bounds[1] - mesh.bounds[0]
        vol = float(ext[0] * ext[1] * ext[2]) if np.all(ext > 0) else 1.0

    if mass_override is not None:
        mass = float(mass_override)
    else:
        mass = float(getattr(mesh, "mass", vol) or vol)

    try:
        if vol > 0:
            mesh.density = float(mass) / float(vol)
        else:
            mesh.density = 1.0
    except Exception:
        pass

    try:
        com = mesh.center_mass
    except Exception:
        com = mesh.centroid

    try:
        I = np.asarray(mesh.moment_inertia, dtype=float)
    except Exception:
        I = np.eye(3) * 1e-8
    I = 0.5 * (I + I.T)

    w, v = np.linalg.eigh(I)
    w = np.real_if_close(w)
    w[w < 0] = np.maximum(w[w >= 0].min() if np.any(w >= 0) else 0.0, 0.0)

    largest_idx = int(np.argmax(w))
    if largest_idx != 2:
        others = [i for i in [0, 1, 2] if i != largest_idx]
        new_order = others + [largest_idx]
        w = w[new_order]
        v = v[:, new_order]

    if np.linalg.det(v) < 0:
        v[:, -1] *= -1.0

    return com, w, v, mass, vol, I


def build_alignment_T(com, principal_axes):
    R = np.asarray(principal_axes, dtype=float)
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1.0
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ np.asarray(com, dtype=float)
    return T


# -------------------------
# External tool wrappers (CoACD / ACVD)
# -------------------------
def mesh_manifold_and_convex_decomp(
    input_path,
    manifold_output_path,
    coacd_output_path,
    quiet=False,
):
    # uses your provided command template
    cmd = f"third_party/CoACD/build/main -i {input_path} -o {coacd_output_path} -ro {manifold_output_path} -pm on"
    if quiet:
        cmd += " > /dev/null 2>&1"
    print("  [CoACD] running:", cmd if not quiet else "coacd (quiet mode)")

    return os.system(cmd)


def mesh_simplify(input_path, output_path, vert_num=4000, gradation=0.5, quiet=False):
    out_dir = os.path.dirname(output_path)
    out_basename = os.path.basename(output_path)
    cmd = f"third_party/ACVD/bin/ACVD {input_path} {vert_num} {gradation} -o {out_dir+os.sep} -of {out_basename} -m 1"
    if quiet:
        cmd += " > /dev/null 2>&1"
    print("  [ACVD] running:", cmd if not quiet else "acvd (quiet mode)")
    os.system(cmd)
    # ACVD may create 'smooth_filename' intermediate; attempt to remove if present
    smooth_p = os.path.join(out_dir, f"smooth_{out_basename}")
    if os.path.exists(smooth_p):
        try:
            os.remove(smooth_p)
        except Exception:
            pass
    return 0


def mesh_decimate_with_pymeshlab(src_obj, dst_obj, target_faces):
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(src_obj))
        try:
            face_num = int(ms.current_mesh().face_number())
        except Exception:
            face_num = target_faces + 1
        if face_num > int(target_faces):
            filter_candidates = [
                (
                    "meshing_decimation_quadric_edge_collapse_with_texture",
                    dict(targetfacenum=int(target_faces), preservenormal=True, preservetopology=True),
                ),
                (
                    "meshing_decimation_quadric_edge_collapse",
                    dict(targetfacenum=int(target_faces), preservenormal=True, preservetopology=True),
                ),
            ]
            applied = False
            for filter_name, kwargs in filter_candidates:
                try:
                    ms.apply_filter(filter_name, **kwargs)
                    applied = True
                    break
                except Exception:
                    continue
            if not applied:
                raise RuntimeError("pymeshlab decimation filters unavailable")
        ms.save_current_mesh(str(dst_obj))
        return None
    except Exception as e:
        raise RuntimeError(f"pymeshlab decimation failed: {e}") from e


def mesh_slim_obj_text(src_obj, dst_obj, decimals=6, force_mtllib=None, force_usemtl=None):
    fmt = "{:." + str(int(decimals)) + "f}"
    out_lines = []
    saw_mtllib = False
    saw_usemtl = False
    try:
        with open(src_obj, "r", encoding="utf-8", errors="ignore") as fin:
            for line in fin:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                m = _re_mtllib.match(line)
                if m:
                    if force_mtllib is not None:
                        if not saw_mtllib:
                            out_lines.append(f"mtllib {force_mtllib}\n")
                            saw_mtllib = True
                    else:
                        out_lines.append(f"mtllib {os.path.basename(m.group(2).strip())}\n")
                        saw_mtllib = True
                    continue
                m = _re_usemtl.match(line)
                if m:
                    saw_usemtl = True
                    if force_usemtl is not None:
                        out_lines.append(f"usemtl {force_usemtl}\n")
                    else:
                        out_lines.append(f"usemtl {m.group(1).strip()}\n")
                    continue
                m = _re_v.match(line)
                if m:
                    _, sx, sy, sz, rest = m.groups()
                    out_lines.append(
                        f"v {fmt.format(float(sx))} {fmt.format(float(sy))} {fmt.format(float(sz))}{rest}\n"
                    )
                    continue
                m = _re_vn.match(line)
                if m:
                    _, sx, sy, sz, rest = m.groups()
                    out_lines.append(
                        f"vn {fmt.format(float(sx))} {fmt.format(float(sy))} {fmt.format(float(sz))}{rest}\n"
                    )
                    continue
                m = _re_vt.match(line)
                if m:
                    _, su, sv, sw, rest = m.groups()
                    if sw is None:
                        out_lines.append(f"vt {fmt.format(float(su))} {fmt.format(float(sv))}{rest}\n")
                    else:
                        out_lines.append(
                            f"vt {fmt.format(float(su))} {fmt.format(float(sv))} {fmt.format(float(sw))}{rest}\n"
                        )
                    continue
                out_lines.append(s + "\n")
    except Exception as e:
        raise RuntimeError(f"OBJ slim failed: {e}") from e

    if force_mtllib is not None and not saw_mtllib:
        out_lines.insert(0, f"mtllib {force_mtllib}\n")
    if force_usemtl is not None and not saw_usemtl:
        insert_at = 1 if out_lines and out_lines[0].lower().startswith("mtllib ") else 0
        out_lines.insert(insert_at, f"usemtl {force_usemtl}\n")

    with open(dst_obj, "w", encoding="utf-8", errors="ignore") as fout:
        fout.writelines(out_lines)


def mesh_compress_texture(src_tex, dst_tex, max_size=1024, jpg_quality=85):
    with Image.open(src_tex) as img:
        img = img.convert("RGBA") if img.mode in ("P", "LA") else img
        w, h = img.size
        scale = min(1.0, float(max_size) / float(max(w, h)))
        if scale < 1.0:
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
            img = img.resize((nw, nh), resample)

        ext = os.path.splitext(dst_tex)[1].lower()
        if ext in (".jpg", ".jpeg"):
            # Keep alpha-safe fallback.
            if "A" in img.getbands():
                dst_png = os.path.splitext(dst_tex)[0] + ".png"
                img.save(dst_png, format="PNG", optimize=True, compress_level=9)
                return {"texture_mode": "png_fallback_alpha", "texture_path": dst_png}
            img = img.convert("RGB")
            img.save(dst_tex, format="JPEG", quality=int(jpg_quality), optimize=True, progressive=True)
            return {"texture_mode": "jpeg", "texture_path": dst_tex}

        img.save(dst_tex, format="PNG", optimize=True, compress_level=9)
        return {"texture_mode": "png", "texture_path": dst_tex}


def mesh_visual(obj_dir, args):
    """
    Build compressed visual assets for MuJoCo loading speed:
      raw.obj -> visual.obj
      textured.mtl/texture -> visual.mtl + visual_texture_map.(jpg|png)
    This function never overwrites raw.obj.
    """
    obj_dir = Path(obj_dir)
    src_obj = obj_dir / "raw.obj"
    if not src_obj.exists():
        raise RuntimeError(f"raw.obj missing: {src_obj}")

    visual_obj = obj_dir / "visual.obj"
    tmp_decimated = obj_dir / "visual.decimated.tmp.obj"
    visual_mtl = obj_dir / "visual.mtl"
    for stale in [obj_dir / "visual_texture_map.jpg", obj_dir / "visual_texture_map.png"]:
        stale.unlink(missing_ok=True)

    # Step A: geometry compression (prefer texture-aware pymeshlab).
    mesh_decimate_with_pymeshlab(
        src_obj=src_obj,
        dst_obj=tmp_decimated,
        target_faces=int(args.visual_target_faces),
    )

    # Step B: texture discovery from existing canonical files.
    src_mtl = obj_dir / "textured.mtl"
    src_tex = obj_dir / "texture_map.png"
    if src_mtl.exists():
        tex_in_mtl = parse_first_texture_from_mtl(str(src_mtl))
        if tex_in_mtl:
            candidate = obj_dir / os.path.basename(tex_in_mtl)
            if candidate.exists():
                src_tex = candidate

    # Step C: texture compression to visual texture.
    dst_tex_ext = ".jpg" if args.visual_texture_format == "jpg" else ".png"
    dst_tex = obj_dir / f"visual_texture_map{dst_tex_ext}"
    texture_info = {"texture_mode": "none", "texture_path": None}
    if src_tex.exists():
        texture_info = mesh_compress_texture(
            str(src_tex),
            str(dst_tex),
            max_size=int(args.visual_texture_max_size),
            jpg_quality=int(args.visual_jpeg_quality),
        )
        tex_path = texture_info.get("texture_path")
        if tex_path:
            dst_tex = Path(tex_path)

    # Step D: write visual.mtl when texture exists.
    material_name = parse_first_usemtl(str(tmp_decimated))
    if not material_name:
        material_name = "material_0"
    if dst_tex.exists():
        tex_name = dst_tex.name
        with open(visual_mtl, "w", encoding="utf-8") as f:
            f.write(f"newmtl {material_name}\n")
            f.write("# shader_type beckmann\n")
            f.write(f"map_Kd {tex_name}\n")
        force_mtllib = visual_mtl.name
    else:
        force_mtllib = None
        if visual_mtl.exists():
            visual_mtl.unlink(missing_ok=True)

    # Step E: text slimming.
    mesh_slim_obj_text(
        src_obj=str(tmp_decimated),
        dst_obj=str(visual_obj),
        decimals=int(args.visual_obj_decimals),
        force_mtllib=force_mtllib,
        force_usemtl=material_name if force_mtllib is not None else None,
    )
    tmp_decimated.unlink(missing_ok=True)

    return {
        "visual_obj": str(visual_obj),
        "visual_mtl": str(visual_mtl) if visual_mtl.exists() else None,
        "visual_texture": str(dst_tex) if dst_tex.exists() else None,
        "decimated_with": "pymeshlab",
        "decimate_note": None,
        "texture_mode": texture_info.get("texture_mode"),
    }


# -------------------------
# High-level pipeline steps
# -------------------------
def preview_obj(dst_obj):
    try:
        scene = trimesh.load(dst_obj, force="scene")
        print("  Preview (close viewer to continue)...")
        scene.show()
    except Exception as e:
        print("  Preview failed (likely headless):", e)


def mesh_transform(
    src_obj,
    dst_inertia_obj,
    dst_inertia_dir,
    mass_value=0.1,
    verbose=False,
):
    """
    Compute principal axes/com, write transformed .obj into dst_inertia_obj.
    Copy/patch mtls & textures into dst_inertia_dir.
    """
    # load mesh using trimesh (no processing)
    try:
        mesh = trimesh.load(src_obj, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = (
                trimesh.util.concatenate([g for g in mesh.geometry.values()])
                if hasattr(mesh, "geometry")
                else mesh
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load mesh {src_obj}: {e}")

    mass_value = float(mass_value)

    com, princ_w, princ_v, mass_used, vol, I = compute_principal(
        mesh, mass_override=mass_value
    )
    if verbose:
        print(f"    volume={vol:.6g} mass_used={mass_used:.6g}")
        print(f"    com={np.round(com,6).tolist()}")
        print(
            f"    princ_moments (ordered, largest->Z): {np.round(princ_w,9).tolist()}"
        )

    T = build_alignment_T(com, princ_v)

    # copy and patch mtl/textures into dst_inertia_dir
    mtllib_map = copy_and_patch_mtl(src_obj, dst_inertia_dir)
    if verbose and mtllib_map:
        print(f"    copied mtls/textures: {mtllib_map}")

    # write transformed obj
    transform_obj_text(src_obj, dst_inertia_obj, T, mtllib_map)

    return {
        "com": com.tolist(),
        "princ_w": princ_w.tolist(),
        "princ_v": princ_v.tolist(),
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


def _visual_outputs_exist(obj_dir: Path) -> bool:
    visual_obj = obj_dir / "visual.obj"
    visual_mtl = obj_dir / "visual.mtl"
    visual_tex_jpg = obj_dir / "visual_texture_map.jpg"
    visual_tex_png = obj_dir / "visual_texture_map.png"
    if not visual_obj.exists():
        return False
    # For textured objects visual.mtl + visual_texture_map are expected.
    if (obj_dir / "texture_map.png").exists():
        return visual_mtl.exists() and (visual_tex_jpg.exists() or visual_tex_png.exists())
    # For non-textured objects only visual.obj is required.
    return True


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
    dst_inertia_obj = obj_dir / "inertia.obj"
    dst_manifold_obj = obj_dir / "manifold.obj"
    dst_coacd_obj = obj_dir / "coacd.obj"
    dst_simplified_obj = obj_dir / "simplified.obj"
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
        rec["error"] = f"raw.obj missing: {src_obj}"
        print(f"    ERROR: {rec['error']}")
        return rec

    # Step 1: inertia transform
    try:
        print("  running mesh_transform...")
        info = mesh_transform(
            str(src_obj),
            str(dst_inertia_obj),
            str(obj_dir),
            mass_value=mass_value,
            verbose=bool(process_cfg["verbose"]),
        )
        rec["center_of_mass"] = info["com"]
        rec["principal_moments"] = info["princ_w"]
        rec["principal_axes"] = info["princ_v"]
        if bool(process_cfg["preview"]):
            preview_obj(str(dst_inertia_obj))
    except Exception as e:
        rec["error"] = f"inertia failed: {e}"
        print(f"    ERROR: {rec['error']}")
        return rec

    # Step 2: manifold + convex decomposition (CoACD)
    try:
        need_coacd = bool(process_cfg["force"]) or (not dst_manifold_obj.exists()) or (not dst_coacd_obj.exists())
        if need_coacd:
            print("  running mesh_manifold_and_convex_decomp...")
            in_for_coacd = str(dst_inertia_obj if dst_inertia_obj.exists() else src_obj)
            ret = mesh_manifold_and_convex_decomp(
                in_for_coacd,
                str(dst_manifold_obj),
                str(dst_coacd_obj),
                quiet=bool(process_cfg["coacd_quiet"]),
            )
            if ret != 0 or (not dst_coacd_obj.exists()):
                raise RuntimeError(f"CoACD failed (ret={ret})")
        else:
            print("  manifold/coacd exists -> skip")
    except Exception as e:
        rec["error"] = str(e)
        print(f"    ERROR: {rec['error']}")
        return rec

    # Step 3: export convex pieces
    try:
        need_convex_export = bool(process_cfg["force"]) or (not _convex_pieces_exist(meshes_dir))
        if need_convex_export:
            print("  exporting convex pieces...")
            for old_piece in meshes_dir.glob("coacd_convex_piece_*.obj"):
                old_piece.unlink(missing_ok=True)
            convex_pieces = list(trimesh.load(str(dst_coacd_obj), process=False).split())
            for i, piece in enumerate(convex_pieces):
                piece_filepath = meshes_dir / f"coacd_convex_piece_{i}.obj"
                export_mesh(piece, str(piece_filepath))
        else:
            print("  convex pieces exist -> skip")
        wrls = obj_dir / "coacd.wrl"
        wrls.unlink(missing_ok=True)
    except Exception as e:
        rec["error"] = f"convex export failed: {e}"
        print(f"    ERROR: {rec['error']}")
        return rec

    # Step 4: simplify (ACVD)
    try:
        if dst_simplified_obj.exists() and not bool(process_cfg["force"]):
            print("  simplified exists -> skip")
        else:
            print("  running mesh_simplify...")
            in_for_acvd = str(
                dst_manifold_obj
                if dst_manifold_obj.exists()
                else (dst_inertia_obj if dst_inertia_obj.exists() else src_obj)
            )
            mesh_simplify(
                in_for_acvd,
                str(dst_simplified_obj),
                vert_num=int(process_cfg["acvd_vertnum"]),
                gradation=float(process_cfg["acvd_gradation"]),
                quiet=bool(process_cfg["acvd_quiet"]),
            )
            if not dst_simplified_obj.exists():
                raise RuntimeError("ACVD finished but simplified.obj missing")
    except Exception as e:
        rec["error"] = f"ACVD failed: {e}"
        print(f"    ERROR: {rec['error']}")
        return rec

    # Step 5: visual compression mesh
    try:
        if _visual_outputs_exist(obj_dir) and not bool(process_cfg["force"]):
            print("  visual outputs exist -> skip")
        else:
            print("  running mesh_visual...")
            mesh_visual(obj_dir=str(obj_dir), args=argparse.Namespace(**process_cfg))
        visual_obj = obj_dir / "visual.obj"
        visual_mtl = obj_dir / "visual.mtl"
        visual_tex = next(
            (p for p in [obj_dir / "visual_texture_map.jpg", obj_dir / "visual_texture_map.png"] if p.exists()),
            None,
        )
        rec["visual_obj_path"] = str(visual_obj) if visual_obj.exists() else None
        rec["visual_mtl_path"] = str(visual_mtl) if visual_mtl.exists() else None
        rec["visual_texture_path"] = str(visual_tex) if visual_tex is not None else None
    except Exception as e:
        rec["error"] = f"visual mesh failed: {e}"
        print(f"    ERROR: {rec['error']}")
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

    if args.preview and args.workers > 1:
        raise ValueError("--preview requires --workers 1")

    per_object: dict[str, dict] = {}
    process_cfg = {
        "force": bool(args.force),
        "preview": bool(args.preview),
        "verbose": bool(args.verbose),
        "coacd_quiet": bool(args.coacd_quiet),
        "acvd_quiet": bool(args.acvd_quiet),
        "acvd_vertnum": int(args.acvd_vertnum),
        "acvd_gradation": float(args.acvd_gradation),
        "visual_target_faces": int(args.visual_target_faces),
        "visual_obj_decimals": int(args.visual_obj_decimals),
        "visual_texture_max_size": int(args.visual_texture_max_size),
        "visual_texture_format": str(args.visual_texture_format),
        "visual_jpeg_quality": int(args.visual_jpeg_quality),
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
            for fut in as_completed(future_map):
                oid = future_map[fut]
                try:
                    ret_oid, rec = fut.result()
                    per_object[ret_oid] = rec
                except Exception as e:
                    per_object[oid] = {"status": "failed", "error": f"worker crash: {e}"}

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
    p.add_argument(
        "--preview",
        action="store_true",
        help="Show trimesh preview for inertia step (requires GUI)",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    p.add_argument("--coacd-quiet", action="store_true", help="Run CoACD quietly")
    p.add_argument("--acvd-quiet", action="store_true", help="Run ACVD quietly")
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
        help="Visual mesh decimation target face count (from raw.obj).",
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
        "--visual-texture-format",
        type=str,
        choices=["jpg", "png"],
        default="jpg",
        help="Preferred visual texture format.",
    )
    p.add_argument(
        "--visual-jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for visual texture (when format=jpg and no alpha).",
    )
    args = p.parse_args()

    args.dataset_root = processed_root / args.dataset
    return args


def main() -> None:
    args = parse_args()
    process_dataset(args.dataset_root, args)


if __name__ == "__main__":
    main()
