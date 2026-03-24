"""ShapeNetSem adapter."""

from __future__ import annotations

import shutil
import zipfile
import csv
from pathlib import Path, PurePosixPath

from .base import (
    CANONICAL_RAW_OBJ_NAME,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
    relative_to_repo,
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
    def _sem_instance_id(self, full_id: str) -> str:
        val = full_id.strip()
        if val.startswith("wss."):
            return val[4:]
        if "." in val:
            return val.split(".", 1)[1]
        return val

    def _select_sem_instances(self, rows: list[dict[str, str]]) -> dict[str, str]:
        selected: dict[str, str] = {}
        for row in rows:
            category_text = row.get("category") or ""
            if category_text == "":
                continue

            full_id = row.get("fullId") or ""
            instance_id = self._sem_instance_id(full_id)
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

    def _find_sem_root(self, snapshot_dir: Path) -> Path | None:
        for metadata_csv in sorted(snapshot_dir.rglob("metadata.csv")):
            root = metadata_csv.parent
            if (root / "models-OBJ" / "models").is_dir():
                return root
        return None

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

        seen_ids = {p.name for p in processed_dir.iterdir() if p.is_dir()}

        obj_root = sem_root / "models-OBJ" / "models"
        iterable = list(selected_instances.items())
        if tqdm is not None:
            iterable = tqdm(iterable, desc="ShapeNetSem organize", unit="obj")

        normalized_count = 0
        dropped_bad_scale = 0
        for instance_id, category in iterable:
            src_obj = obj_root / f"{instance_id}.obj"

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

            ok_norm, norm_reason, norm_metrics = self.normalize_obj_center_and_scale(
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
        report.notes.append("ShapeNetSem texture export: disabled, organized OBJ-only")
        self.write_manifest_for_organize(cfg, report)
        return report
