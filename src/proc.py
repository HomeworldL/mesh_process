#!/usr/bin/env python3
"""
src/proc.py

Batch-preprocess YCB models into dataset structure with steps:
  1) copy raw obj/mtl/textures -> dst/raw/
  2) transform to COM + principal axes -> dst/inertia.obj
  3) run CoACD to produce watertight mesh -> dst/manifold.obj
  4) run CoACD to produce convex pieces -> dst/meshes/
  5) run ACVD to simplify mesh -> dst/simplified.obj

Usage:
  python src/proc.py --src assets/YCB/ycb --dst assets/YCB/ycb_datasets
Options:
  --skip-existing   skip objects that already have output files
  --force           overwrite outputs
  --acvd-vertnum    target vertex number for ACVD (default 4000)
  --acvd-gradation  ACVD gradation parameter (default 0.5)
  --coacd-quiet / --acvd-quiet  silence external tools
  --preview         show trimesh preview after inertia transform (requires GUI)
"""

import os
import sys
import argparse
import shutil
import re
import json
from pathlib import Path
import subprocess
import numpy as np
import trimesh
from trimesh.exchange.export import export_mesh

# -------------------------
# Regex patterns (robust)
# -------------------------
_re_v = re.compile(r"^(\s*)v\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)(.*)$")
_re_vn = re.compile(r"^(\s*)vn\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)(.*)$")
_re_mtllib = re.compile(r"^(\s*)mtllib\s+(.+)$", re.IGNORECASE)


# -------------------------
# Helpers (copied/adapted from your provided code)
# -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_obj_files(src_root):
    for root, _, files in os.walk(src_root):
        for fn in files:
            if fn.lower().endswith(".obj"):
                yield os.path.join(root, fn)


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


# -------------------------
# High-level pipeline steps
# -------------------------
def copy_raw_files(src_obj, dst_raw_dir, verbose=False):
    """
    Copy the .obj plus its mtl and referenced textures into dst_raw_dir.
    """
    ensure_dir(dst_raw_dir)
    src_dir = os.path.dirname(src_obj)

    # copy all files in the source folder (common for YCB textured set)
    try:
        for fn in os.listdir(src_dir):
            src_f = os.path.join(src_dir, fn)
            if os.path.isfile(src_f):
                dst_f = os.path.join(dst_raw_dir, fn)
                try:
                    shutil.copy2(src_f, dst_f)
                except Exception as e:
                    if verbose:
                        print(f"    warn: failed copying {src_f}: {e}")
    except Exception as e:
        if verbose:
            print("    warn: couldn't enumerate src dir:", e)


def try_find_mass(mass_map, folder_key):
    # direct key
    if folder_key in mass_map:
        return mass_map[folder_key]
    # remove leading digits + underscore: '002_master_chef_can' -> 'master_chef_can'
    m = re.match(r"^\d+_(.+)$", folder_key)
    if m:
        key2 = m.group(1)
        if key2 in mass_map:
            return mass_map[key2]
    # fallback: try part after first underscore
    if "_" in folder_key:
        parts = folder_key.split("_", 1)
        if len(parts) == 2 and parts[1] in mass_map:
            return mass_map[parts[1]]
    return None


def preview_obj(dst_obj):
    try:
        scene = trimesh.load(dst_obj, force="scene")
        print("  Preview (close viewer to continue)...")
        scene.show()
    except Exception as e:
        print("  Preview failed (likely headless):", e)


def do_inertia_transform(
    src_obj, dst_inertia_obj, dst_inertia_dir, mass_map=None, force=False, verbose=False
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

    # determine mass override if provided
    folder_key = os.path.basename(os.path.dirname(src_obj))
    mass_value = try_find_mass(mass_map or {}, folder_key)

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
        "mass_used": float(mass_used),
    }


# -------------------------
# Orchestrator
# -------------------------
def process_all(src_root, dst_root, args):
    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)
    ensure_dir(dst_root)

    # load mass map if provided
    mass_map = {}
    if args.mass and os.path.exists(args.mass):
        try:
            with open(args.mass, "r", encoding="utf-8") as f:
                mass_map = json.load(f)
        except Exception:
            print("Warning: failed to read mass json; proceeding without it")

    metadata = {}
    obj_list = sorted(list(find_obj_files(src_root)))
    if not obj_list:
        print("No .obj files found under", src_root)
        return

    print(f"Found {len(obj_list)} .obj files under {src_root}")

    for idx, src_obj in enumerate(obj_list, start=1):
        # object folder key: first-level folder name (e.g. '002_master_chef_can')
        rel = os.path.relpath(src_obj, src_root).replace("\\", "/")
        parts = rel.split("/")
        folder_key = parts[0] if len(parts) > 1 else os.path.dirname(rel) or rel
        print(f"[{idx}/{len(obj_list)}] {folder_key}")

        # build per-object base dir
        obj_base_dir = os.path.join(dst_root, folder_key)
        ensure_dir(obj_base_dir)

        # raw.obj placed under obj_raw_dir
        obj_raw_dir = os.path.join(obj_base_dir, "raw")
        ensure_dir(obj_raw_dir)

        # target filenames placed directly under obj_base_dir
        dst_raw_obj = os.path.join(obj_raw_dir, "raw.obj")
        dst_inertia_obj = os.path.join(obj_base_dir, "inertia.obj")
        dst_manifold_obj = os.path.join(obj_base_dir, "manifold.obj")
        dst_coacd_obj = os.path.join(obj_base_dir, "coacd.obj")
        dst_simplified_obj = os.path.join(obj_base_dir, "simplified.obj")

        # coacd part obj under
        part_output_folder = os.path.join(obj_base_dir, "meshes")
        ensure_dir(part_output_folder)
        # part_output_filename = folder_key + "_part"

        # Step 1: copy raw (if not exists or force)
        if os.path.exists(dst_raw_obj) and args.skip_existing and not args.force:
            print("  raw exists -> skip")
        else:
            print("  copying raw files...")
            copy_raw_files(src_obj, obj_raw_dir, verbose=args.verbose)

        if args.preview:
            preview_obj(dst_raw_obj)

        # Step 2: inertia transform
        if os.path.exists(dst_inertia_obj) and args.skip_existing and not args.force:
            print("  inertia exists -> skip")
            # try to read metadata if present
        else:
            print("  computing inertia & writing transformed obj...")
            try:
                info = do_inertia_transform(
                    src_obj,
                    dst_inertia_obj,
                    obj_base_dir,
                    mass_map,
                    force=args.force,
                    verbose=args.verbose,
                )
                metadata[folder_key] = {
                    "src_obj": os.path.relpath(src_obj, src_root),
                    "raw_obj": os.path.relpath(dst_raw_obj, dst_root),
                    "inertia_obj": os.path.relpath(dst_inertia_obj, dst_root),
                    "mass_used": info["mass_used"],
                    "center_of_mass": info["com"],
                    "principal_moments": info["princ_w"],
                    "principal_axes": info["princ_v"],
                }
                print(
                    f"    wrote inertia obj: {os.path.relpath(dst_inertia_obj, dst_root)}"
                )

                # write dataset metadata
                meta_path = os.path.join(obj_base_dir, "metadata.json")
                try:
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    print("Wrote metadata:", meta_path)
                except Exception as e:
                    print("Failed to write metadata file:", e)

            except Exception as e:
                print(f"    ERROR computing inertia for {folder_key}: {e}")
                continue

        if args.preview:
            preview_obj(dst_inertia_obj)

        # Step 3: manifold + convex decomposition (CoACD)
        if os.path.exists(dst_manifold_obj) and args.skip_existing and not args.force:
            print("  manifold exists -> skip")
        else:
            print("  running CoACD to create manifold...")
            in_for_coacd = (
                dst_inertia_obj if os.path.exists(dst_inertia_obj) else dst_raw_obj
            )
            ret = mesh_manifold_and_convex_decomp(
                in_for_coacd,
                dst_manifold_obj,
                dst_coacd_obj,
                quiet=args.coacd_quiet,
            )
            if ret != 0:
                print(
                    f"    CoACD returned code {ret} (object {folder_key}), continuing."
                )

        # save part
        convex_pieces = list(trimesh.load(dst_coacd_obj, process=False).split())
        for i, piece in enumerate(convex_pieces):

            # Save each nearly convex mesh out to a file
            piece_name = "{}_convex_piece_{}".format("coacd", i)
            piece_filename = "{}.obj".format(piece_name)
            piece_filepath = os.path.join(part_output_folder, piece_filename)
            export_mesh(piece, piece_filepath)

        # delete .wrl
        try:
            wrls = os.path.join(obj_base_dir, "coacd.wrl")
            os.remove(wrls)
        except Exception:
            pass

        # Step 4: simplify (ACVD)
        if os.path.exists(dst_simplified_obj) and args.skip_existing and not args.force:
            print("  simplified exists -> skip")
        else:
            print("  running ACVD to simplify...")
            in_for_acvd = (
                dst_manifold_obj
                if os.path.exists(dst_manifold_obj)
                else (
                    dst_inertia_obj if os.path.exists(dst_inertia_obj) else dst_raw_obj
                )
            )
            mesh_simplify(
                in_for_acvd,
                dst_simplified_obj,
                vert_num=args.acvd_vertnum,
                gradation=args.acvd_gradation,
                quiet=args.acvd_quiet,
            )

        break

    print("Done processing all objects.")


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess YCB meshes into raw/inertia/manifold/simplified folders."
    )
    p.add_argument(
        "--src",
        type=str,
        default="assets/YCB/ycb",
        help="Source folder containing YCB object subfolders",
    )
    p.add_argument(
        "--dst",
        type=str,
        default="assets/YCB/ycb_datasets",
        help="Destination dataset root",
    )
    p.add_argument(
        "--mass",
        type=str,
        default="assets/YCB/ycb/ycb_mass.json",
        help="Optional JSON map of masses for objects",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip if target exists")
    p.add_argument("--force", action="store_true", help="Overwrite outputs")
    p.add_argument(
        "--preview",
        action="store_true",
        help="Show trimesh preview for inertia step (requires GUI)",
    )
    p.add_argument("--verbose", action="store_false", help="Verbose output")
    p.add_argument("--coacd-quiet", action="store_true", help="Run CoACD quietly")
    p.add_argument("--acvd-quiet", action="store_true", help="Run ACVD quietly")
    p.add_argument(
        "--acvd-vertnum", type=int, default=2000, help="ACVD: target vertex number"
    )
    p.add_argument(
        "--acvd-gradation", type=float, default=1.5, help="ACVD: gradation parameter"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all(args.src, args.dst, args)
