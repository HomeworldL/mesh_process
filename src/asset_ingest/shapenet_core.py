"""ShapeNetCore adapter."""

from __future__ import annotations

import os
import shutil
from pathlib import Path, PurePosixPath

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
from .filter_lists import SHAPENET_CORE_BAD_INSTANCES, SHAPENET_CORE_CATEGORIES
from .shapenet_common import _ShapeNetHFBaseAdapter


class ShapeNetCoreAdapter(_ShapeNetHFBaseAdapter):
    source_name = "ShapeNetCore"
    version = None
    hf_repo_id = "ShapeNet/ShapeNetCore-archive"
    homepage = "https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive"
    required_files = [
        ".gitattributes",
        "DATA.md",
        "README.md",
        "ShapeNetCore.v2.zip",
    ]
    _TEXTURE_LINE_PREFIXES = ("map_kd", "map_ka", "map_d", "map_bump", "bump")
    _PREPARE_MARKER = ".prepare_core_v2_complete"

    @staticmethod
    def _export_textures_enabled() -> bool:
        # default OFF; enable with SHAPENET_EXPORT_TEXTURES=1 or SHAPENET_CORE_EXPORT_TEXTURES=1
        v = os.environ.get("SHAPENET_CORE_EXPORT_TEXTURES", "").strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        return _ShapeNetHFBaseAdapter._truthy_env("SHAPENET_EXPORT_TEXTURES")

    @staticmethod
    def _find_core_root(snapshot_dir: Path) -> Path | None:
        synset_codes = list(SHAPENET_CORE_CATEGORIES.values())
        stack: list[Path] = [snapshot_dir]
        best_root: Path | None = None
        best_score = -1
        while stack:
            cur = stack.pop()
            try:
                children = [p for p in cur.iterdir() if p.is_dir()]
            except Exception:
                children = []
            score = sum(1 for code in synset_codes if (cur / code).is_dir())
            if score > best_score:
                best_score = score
                best_root = cur
            stack.extend(children)
        if best_score <= 0:
            return None
        return best_root

    def _prepare_after_download(self, cfg: IngestConfig, snapshot_dir: Path, report: DownloadReport) -> None:
        zip_path = snapshot_dir / "ShapeNetCore.v2.zip"
        if not zip_path.exists():
            report.notes.append(f"missing archive: {relative_to_repo(cfg.repo_root, zip_path)}")
            return

        code_to_category = {v: k for k, v in SHAPENET_CORE_CATEGORIES.items()}
        def include_member(member: str) -> bool:
            p = PurePosixPath(member)
            parts = p.parts
            code_idx = -1
            code = ""
            for idx, token in enumerate(parts):
                if token in code_to_category:
                    code_idx = idx
                    code = token
                    break
            if code_idx < 0:
                return False
            if len(parts) <= code_idx + 1:
                return False

            instance_id = parts[code_idx + 1]
            category = code_to_category[code]
            if f"{category}-{instance_id}" in SHAPENET_CORE_BAD_INSTANCES:
                return False

            rel = "/".join(parts[code_idx + 2 :])
            if rel in {
                "models/model_normalized.obj",
                "models/model_normalized.mtl",
                "models/model_normalized.json",
            }:
                return True

            if len(parts) > code_idx + 3 and parts[code_idx + 2] == "images":
                return p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            return False

        selected, extracted = self._extract_selected_members(
            zip_path=zip_path,
            output_dir=snapshot_dir,
            include_predicate=include_member,
            force=cfg.force,
            desc="ShapeNetCore selective extract",
        )
        report.notes.append(
            f"selective extracted from zip: selected={selected}, newly_extracted={extracted}"
        )
        marker = snapshot_dir / self._PREPARE_MARKER
        marker.write_text("ok\n", encoding="utf-8")

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

        core_root = self._find_core_root(snapshot_dir)
        marker = snapshot_dir / self._PREPARE_MARKER
        needs_prepare = core_root is None or (not marker.exists())
        if needs_prepare:
            prep_report = DownloadReport(source=self.source_name)
            self._prepare_after_download(cfg, snapshot_dir, prep_report)
            report.notes.extend(prep_report.notes)
            core_root = self._find_core_root(snapshot_dir)
            if core_root is None:
                report.notes.append(f"cannot locate extracted ShapeNetCore root under: {snapshot_dir}")
                return report

        seen_ids = {p.name for p in processed_dir.iterdir() if p.is_dir()}
        baked_single_texture = 0
        normalized_count = 0
        dropped_bad_scale = 0
        export_textures = self._export_textures_enabled()

        candidates: list[tuple[str, Path]] = []
        for category, synset_code in SHAPENET_CORE_CATEGORIES.items():
            cat_dir = core_root / synset_code
            if not cat_dir.is_dir():
                continue
            for instance_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
                candidates.append((category, instance_dir))

        iterable = candidates
        if tqdm is not None:
            iterable = tqdm(candidates, desc="ShapeNetCore organize", unit="obj")

        for category, instance_dir in iterable:
            instance_id = instance_dir.name
            if f"{category}-{instance_id}" in SHAPENET_CORE_BAD_INSTANCES:
                continue

            model_dir = instance_dir / "models"
            src_obj = model_dir / "model_normalized.obj"
            src_mtl = model_dir / "model_normalized.mtl"
            src_json = model_dir / "model_normalized.json"
            images_dir = instance_dir / "images"
            if not src_obj.is_file():
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
            ok_norm, norm_reason, norm_metrics = self.normalize_obj_center_and_scale(
                src_obj_path=src_obj,
                dst_obj_path=dst_obj,
            )
            if not ok_norm:
                dropped_bad_scale += 1
                report.failed_items.append(
                    f"drop {category}-{instance_id}: normalize failed ({norm_reason}); metrics={norm_metrics}"
                )
                shutil.rmtree(dst_dir, ignore_errors=True)
                continue
            normalized_count += 1

            if export_textures:
                texture_rename_map: dict[str, str] = {}
                copied_png_count = 0
                texture_refs = (
                    self._parse_mtl_texture_refs(
                        src_mtl,
                        texture_line_prefixes=self._TEXTURE_LINE_PREFIXES,
                    )
                    if src_mtl.is_file()
                    else []
                )
                has_jpg_in_source_mtl = any(Path(x).suffix.lower() in {".jpg", ".jpeg"} for x in texture_refs)
                for idx_tex, tex_name in enumerate(texture_refs):
                    src_tex = images_dir / tex_name
                    if not src_tex.is_file():
                        src_tex = images_dir / Path(tex_name).name
                    if not src_tex.is_file():
                        continue
                    renamed = f"texture_map_{idx_tex}.png"
                    if self._copy_texture_to_png(src_tex, dst_dir / renamed):
                        texture_rename_map[Path(tex_name).name] = renamed
                        copied_png_count += 1

                if src_mtl.is_file():
                    self._rewrite_mtl_with_local_textures(
                        src_mtl=src_mtl,
                        dst_mtl=dst_dir / CANONICAL_MTL_NAME,
                        texture_rename_map=texture_rename_map,
                        texture_line_prefixes=self._TEXTURE_LINE_PREFIXES,
                    )
                    rewrite_obj_material_refs(obj_path=dst_obj, mtl_name=CANONICAL_MTL_NAME)
                elif copied_png_count > 0:
                    canonicalize_texture_assets(
                        dst_dir=dst_dir,
                        raw_obj_path=dst_obj,
                        texture_src=(dst_dir / "texture_map_0.png"),
                        mtl_src=None,
                        create_mtl_if_texture=True,
                    )

                canonical_mtl = dst_dir / CANONICAL_MTL_NAME
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

        report.notes.append(f"organized from extracted root: {relative_to_repo(cfg.repo_root, core_root)}")
        report.notes.append(
            f"ShapeNetCore filter: {len(SHAPENET_CORE_CATEGORIES)} categories + bad instance denylist"
        )
        report.notes.append(
            "ShapeNet normalize policy: center=AABB center to origin, scale=max_extent to 1.0; "
            "drop if pre_max_extent<=1e-9 or pre_aspect_ratio>1e5 or post_max_extent not in [0.95,1.05] or post_center_norm>5e-3"
        )
        report.notes.append(
            f"ShapeNetCore normalize stats: normalized={normalized_count}, dropped_bad_scale={dropped_bad_scale}"
        )
        if export_textures:
            report.notes.append(f"ShapeNetCore texture export: enabled, baked_single_texture={baked_single_texture}")
        else:
            report.notes.append("ShapeNetCore texture export: disabled (default), organized OBJ-only")
        self.write_manifest_for_organize(cfg, report)
        return report
