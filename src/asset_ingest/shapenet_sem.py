"""ShapeNetSem adapter."""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath
import os
import shutil
import zipfile
import re

from .base import (
    CANONICAL_MTL_NAME,
    CANONICAL_RAW_OBJ_NAME,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
    canonicalize_texture_assets,
    relative_to_repo,
    rewrite_obj_material_refs,
    sanitize_object_id,
    tqdm,
)
from .filter_lists import SHAPENET_SEM_BAD_CATEGORIES, SHAPENET_SEM_BAD_INSTANCE_IDS
from .shapenet_common import _ShapeNetHFBaseAdapter


class ShapeNetSemAdapter(_ShapeNetHFBaseAdapter):
    source_name = "ShapeNetSem"
    version = None
    hf_repo_id = "ShapeNet/ShapeNetSem-archive"
    homepage = "https://huggingface.co/datasets/ShapeNet/ShapeNetSem-archive"
    required_files = [
        ".gitattributes",
        "DATA.md",
        "README.md",
        "ShapeNetSem.zip",
    ]
    _PREPARE_MARKER = ".prepare_sem_v2_complete"
    _TEXTURE_LINE_PREFIXES = ("map_kd", "map_ka", "map_d", "map_bump", "bump")

    @staticmethod
    def _export_textures_enabled() -> bool:
        # default OFF; enable with SHAPENET_EXPORT_TEXTURES=1 or SHAPENET_SEM_EXPORT_TEXTURES=1
        v = os.environ.get("SHAPENET_SEM_EXPORT_TEXTURES", "").strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        return _ShapeNetHFBaseAdapter._truthy_env("SHAPENET_EXPORT_TEXTURES")

    @staticmethod
    def _sem_instance_id(full_id: str) -> str:
        val = full_id.strip()
        if val.startswith("wss."):
            return val[4:]
        if "." in val:
            return val.split(".", 1)[1]
        return val

    @staticmethod
    def _select_sem_instances(rows: list[dict[str, str]]) -> dict[str, str]:
        selected: dict[str, str] = {}
        for row in rows:
            category_text = row.get("category") or ""
            if category_text == "":
                continue

            full_id = row.get("fullId") or ""
            instance_id = ShapeNetSemAdapter._sem_instance_id(full_id)
            if not instance_id:
                continue
            if instance_id in SHAPENET_SEM_BAD_INSTANCE_IDS:
                continue

            chosen_cat: str | None = None
            for cat in category_text.split(","):
                if cat and cat[0] == "_":
                    continue
                if cat in SHAPENET_SEM_BAD_CATEGORIES:
                    continue
                chosen_cat = cat
                break

            if chosen_cat is None:
                continue
            selected[instance_id] = chosen_cat
        return selected

    def _find_sem_meta_member(self, zip_path: Path) -> str | None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if PurePosixPath(info.filename).name == "metadata.csv":
                    return info.filename
        return None

    def _read_sem_metadata(self, metadata_path: Path) -> list[dict[str, str]]:
        with metadata_path.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            return [dict(row) for row in reader]

    @staticmethod
    def _find_sem_root(snapshot_dir: Path) -> Path | None:
        for metadata_csv in sorted(snapshot_dir.rglob("metadata.csv")):
            root = metadata_csv.parent
            if (root / "models-OBJ" / "models").is_dir():
                return root
        return None

    @staticmethod
    def _parse_mtl_texture_refs(mtl_path: Path) -> set[str]:
        refs: set[str] = set()
        if not mtl_path.exists():
            return refs
        try:
            for raw in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                if parts[0].lower() not in ShapeNetSemAdapter._TEXTURE_LINE_PREFIXES:
                    continue
                # Skip MTL option flags (e.g. -blendu on -s 1 1 1), last token is texture path.
                tex_token = parts[-1]
                tex_name = Path(tex_token).name
                if tex_name:
                    refs.add(tex_name)
        except Exception:
            return refs
        return refs

    @staticmethod
    def _copy_texture_to_png(src_path: Path, dst_png_path: Path) -> bool:
        """Copy image to PNG format; return True on success."""
        try:
            from PIL import Image  # type: ignore

            with Image.open(src_path) as img:
                # Keep alpha when present.
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
                img.save(dst_png_path, format="PNG")
            return True
        except Exception:
            return False

    @staticmethod
    def _rewrite_mtl_with_local_textures(
        src_mtl: Path, dst_mtl: Path, texture_rename_map: dict[str, str]
    ) -> None:
        out_lines: list[str] = []
        try:
            lines = src_mtl.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            lines = []

        for raw in lines:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                out_lines.append(line)
                continue

            parts = stripped.split()
            if len(parts) >= 2 and parts[0].lower() in ShapeNetSemAdapter._TEXTURE_LINE_PREFIXES:
                tex_token = parts[-1]
                tex_name = Path(tex_token).name
                if tex_name in texture_rename_map:
                    parts[-1] = texture_rename_map[tex_name]
                    prefix = re.match(r"^\s*", line).group(0) if line else ""
                    out_lines.append(prefix + " ".join(parts))
                    continue
            out_lines.append(line)

        if not out_lines:
            out_lines = ["newmtl material_0"]
        dst_mtl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    def _prepare_after_download(self, cfg: IngestConfig, snapshot_dir: Path, report: DownloadReport) -> None:
        zip_path = snapshot_dir / "ShapeNetSem.zip"
        if not zip_path.exists():
            report.notes.append(f"missing archive: {relative_to_repo(cfg.repo_root, zip_path)}")
            return

        metadata_member = self._find_sem_meta_member(zip_path)
        if metadata_member is None:
            report.notes.append("cannot find metadata.csv inside ShapeNetSem.zip")
            return

        meta_root = str(PurePosixPath(metadata_member).parent)
        sem_root_local = snapshot_dir / Path(*PurePosixPath(meta_root).parts)
        metadata_local_path = snapshot_dir / Path(*PurePosixPath(metadata_member).parts)

        meta_files = {
            "metadata.csv",
            "categories.synset.csv",
            "densities.csv",
            "materials.csv",
            "taxonomy.txt",
        }

        def include_meta(member: str) -> bool:
            p = PurePosixPath(member)
            if not member.startswith(meta_root + "/"):
                return False
            return p.name in meta_files

        selected_meta, extracted_meta = self._extract_selected_members(
            zip_path=zip_path,
            output_dir=snapshot_dir,
            include_predicate=include_meta,
            force=cfg.force,
            desc="ShapeNetSem prepare metadata",
        )

        if not metadata_local_path.exists():
            report.notes.append("metadata.csv not extracted; skip selective model extraction")
            return

        rows = self._read_sem_metadata(metadata_local_path)
        selected_instances = self._select_sem_instances(rows)
        selected_ids = set(selected_instances.keys())

        def include_models(member: str) -> bool:
            if not member.startswith(meta_root + "/"):
                return False
            rel = PurePosixPath(member).relative_to(PurePosixPath(meta_root))
            parts = rel.parts
            if len(parts) < 3:
                return False
            if parts[0] != "models-OBJ" or "models" not in parts[1:]:
                return False

            suffix = PurePosixPath(parts[-1]).suffix.lower()
            stem = PurePosixPath(parts[-1]).stem
            return suffix in {".obj", ".mtl"} and stem in selected_ids

        selected_models, extracted_models = self._extract_selected_members(
            zip_path=zip_path,
            output_dir=snapshot_dir,
            include_predicate=include_models,
            force=cfg.force,
            desc="ShapeNetSem prepare obj+mtl",
        )

        selected_textures = 0
        extracted_textures = 0
        def include_textures(member: str) -> bool:
            if not member.startswith(meta_root + "/"):
                return False
            rel = PurePosixPath(member).relative_to(PurePosixPath(meta_root))
            parts = rel.parts
            if len(parts) < 3:
                return False
            if parts[0] != "models-textures" or "textures" not in parts[1:]:
                return False
            return PurePosixPath(parts[-1]).suffix.lower() in {".jpg", ".jpeg", ".png"}

        selected_textures, extracted_textures = self._extract_selected_members(
            zip_path=zip_path,
            output_dir=snapshot_dir,
            include_predicate=include_textures,
            force=cfg.force,
            desc="ShapeNetSem prepare textures(all)",
        )

        report.notes.append(
            f"ShapeNetSem extracted metadata files: selected={selected_meta}, newly_extracted={extracted_meta}"
        )
        report.notes.append(
            "ShapeNetSem extracted obj+mtl: "
            f"selected={selected_models}, newly_extracted={extracted_models}, instances={len(selected_instances)}"
        )
        report.notes.append(
            "ShapeNetSem extracted textures: "
            f"selected={selected_textures}, newly_extracted={extracted_textures}"
        )
        (sem_root_local / self._PREPARE_MARKER).write_text("ok\n", encoding="utf-8")

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        snapshot_dir = self._snapshot_dir(cfg)
        processed_dir = cfg.source_processed_dir
        if cfg.force and processed_dir.exists():
            shutil.rmtree(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not snapshot_dir.exists():
            report.notes.append(f"missing source dir: {snapshot_dir}")
            return report

        sem_root = self._find_sem_root(snapshot_dir)
        tex_root_probe = sem_root / "models-textures" / "textures" if sem_root else None
        marker_probe = sem_root / self._PREPARE_MARKER if sem_root else None
        needs_prepare = sem_root is None or not tex_root_probe.exists() or not marker_probe.exists()
        if needs_prepare:
            prep_report = DownloadReport(source=self.source_name)
            self._prepare_after_download(cfg, snapshot_dir, prep_report)
            report.notes.extend(prep_report.notes)
            sem_root = self._find_sem_root(snapshot_dir)
            if sem_root is None:
                report.notes.append(f"cannot locate extracted ShapeNetSem root under: {snapshot_dir}")
                return report

        metadata_csv = sem_root / "metadata.csv"
        if not metadata_csv.exists():
            report.notes.append(f"missing metadata csv: {metadata_csv}")
            return report

        rows = self._read_sem_metadata(metadata_csv)
        selected_instances = self._select_sem_instances(rows)

        subset = self._subset_filter(cfg)
        seen_ids = {p.name for p in processed_dir.iterdir() if p.is_dir()}

        obj_root = sem_root / "models-OBJ" / "models"
        tex_root = sem_root / "models-textures" / "textures"
        texture_by_name: dict[str, Path] = {}
        export_textures = self._export_textures_enabled()
        if export_textures and tex_root.is_dir():
            for p in tex_root.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                texture_by_name.setdefault(p.name, p)

        iterable = list(selected_instances.items())
        if tqdm is not None:
            iterable = tqdm(iterable, desc="ShapeNetSem organize", unit="obj")

        baked_single_texture = 0
        normalized_count = 0
        dropped_bad_scale = 0
        for instance_id, category in iterable:
            if subset is not None and category not in subset:
                continue

            src_obj = obj_root / f"{instance_id}.obj"
            src_mtl = obj_root / f"{instance_id}.mtl"

            if not src_obj.is_file():
                report.failed_items.append(f"missing obj for {instance_id}")
                continue

            base_id = sanitize_object_id(f"{self.source_name}_{category}_{instance_id}")
            if base_id in seen_ids:
                existing_base_dir = processed_dir / base_id
                if existing_base_dir.exists():
                    if cfg.force:
                        shutil.rmtree(existing_base_dir)
                    else:
                        continue
                object_id = base_id
            else:
                object_id = base_id
                seen_ids.add(object_id)

            dst_dir = processed_dir / object_id
            if dst_dir.exists() and cfg.force:
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst_obj = dst_dir / CANONICAL_RAW_OBJ_NAME
            if dst_obj.exists() and not cfg.force:
                continue

            ok_norm, norm_reason, norm_metrics = self._normalize_obj_center_and_scale(
                src_obj_path=src_obj,
                dst_obj_path=dst_obj,
            )
            if not ok_norm:
                dropped_bad_scale += 1
                report.failed_items.append(
                    f"drop {instance_id}: normalize failed ({norm_reason}); metrics={norm_metrics}"
                )
                shutil.rmtree(dst_dir, ignore_errors=True)
                continue
            normalized_count += 1

            if export_textures:
                texture_src: Path | None = None
                texture_refs = self._parse_mtl_texture_refs(src_mtl) if src_mtl.is_file() else set()
                has_jpg_in_source_mtl = any(Path(x).suffix.lower() in {".jpg", ".jpeg"} for x in texture_refs)
                texture_rename_map: dict[str, str] = {}
                copied_textures: list[tuple[str, Path]] = []
                for tex_name in sorted(texture_refs):
                    candidate = texture_by_name.get(tex_name)
                    if candidate is None or not candidate.is_file():
                        continue
                    copied_textures.append((tex_name, candidate))
                    if texture_src is None:
                        texture_src = candidate

                if not copied_textures and tex_root.is_dir():
                    for ext in [".jpg", ".jpeg", ".png"]:
                        candidate = texture_by_name.get(f"{instance_id}{ext}")
                        if candidate is not None and candidate.is_file():
                            copied_textures.append((candidate.name, candidate))
                            texture_src = candidate
                            break

                for idx_tex, (orig_name, candidate) in enumerate(copied_textures):
                    renamed = f"texture_map_{idx_tex}.png"
                    dst_png = dst_dir / renamed
                    ok = self._copy_texture_to_png(candidate, dst_png)
                    if not ok:
                        continue
                    texture_rename_map[orig_name] = renamed

                canonical_mtl = dst_dir / CANONICAL_MTL_NAME
                if src_mtl.is_file():
                    self._rewrite_mtl_with_local_textures(
                        src_mtl=src_mtl,
                        dst_mtl=canonical_mtl,
                        texture_rename_map=texture_rename_map,
                    )
                    rewrite_obj_material_refs(obj_path=dst_obj, mtl_name=CANONICAL_MTL_NAME)
                elif texture_src is not None:
                    fallback_png = dst_dir / "texture_map_0.png"
                    if self._copy_texture_to_png(texture_src, fallback_png):
                        texture_src = fallback_png
                    else:
                        texture_src = None
                    canonicalize_texture_assets(
                        dst_dir=dst_dir,
                        raw_obj_path=dst_obj,
                        texture_src=texture_src,
                        mtl_src=None,
                        create_mtl_if_texture=True,
                    )

                if has_jpg_in_source_mtl and canonical_mtl.is_file():
                    baked_ok, _ = self._bake_single_texture_png(
                        dst_dir=dst_dir,
                        raw_obj_path=dst_obj,
                        src_mtl_path=canonical_mtl,
                    )
                    if baked_ok:
                        baked_single_texture += 1

                # Enforce final canonical output: raw.obj + optional single texture_map.png/textured.mtl.
                tex_candidates = sorted(p for p in dst_dir.glob("*.png") if p.is_file())
                tex_pick = tex_candidates[0] if tex_candidates else None
                mtl_pick = canonical_mtl if canonical_mtl.is_file() else None
                canonicalize_texture_assets(
                    dst_dir=dst_dir,
                    raw_obj_path=dst_obj,
                    texture_src=tex_pick,
                    mtl_src=mtl_pick,
                    create_mtl_if_texture=True,
                )

            report.organized_objects += 1

        report.notes.append(f"organized from extracted root: {relative_to_repo(cfg.repo_root, sem_root)}")
        report.notes.append(f"ShapeNetSem metadata: {relative_to_repo(cfg.repo_root, metadata_csv)}")
        report.notes.append("ShapeNetSem filter: metadata first-valid category + denylist categories + denylist ids")
        report.notes.append(
            "ShapeNet normalize policy: center=AABB center to origin, scale=max_extent to 1.0; "
            "drop if pre_max_extent<=1e-9 or pre_aspect_ratio>1e5 or post_max_extent not in [0.95,1.05] or post_center_norm>5e-3"
        )
        report.notes.append(
            f"ShapeNetSem normalize stats: normalized={normalized_count}, dropped_bad_scale={dropped_bad_scale}"
        )
        if export_textures:
            report.notes.append(f"ShapeNetSem texture export: enabled, baked_single_texture={baked_single_texture}")
        else:
            report.notes.append("ShapeNetSem texture export: disabled (default), organized OBJ-only")
        self.write_manifest_for_organize(cfg, report)
        return report
