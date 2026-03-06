"""RealDex source adapter: download archive, organize object meshes, build manifest."""

from __future__ import annotations

from pathlib import Path
import shutil

from .base import (
    DEFAULT_MASS_KG,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
    download_google_drive_file,
    extract_zip,
    relative_to_repo,
    sanitize_object_id,
)
from .manifest import IngestManifest, ManifestSource, ManifestSummary, ObjectRecord


class RealDexAdapter(BaseIngestAdapter):
    source_name = "RealDex"
    version = None

    zip_file_id = "1u4q9N_q-pgEYfyzr94vPk4YWkcx8vggh"
    zip_name = "RealDex-objmodels.zip"

    def download(self, cfg: IngestConfig) -> DownloadReport:
        out_dir = cfg.source_download_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report = DownloadReport(source=self.source_name)

        zip_path = out_dir / self.zip_name
        if not zip_path.exists() or cfg.force:
            download_google_drive_file(self.zip_file_id, zip_path)
            report.downloaded_files.append(relative_to_repo(cfg.repo_root, zip_path))
        else:
            report.notes.append(f"skip existing: {zip_path.name}")

        extract_dir = out_dir / "RealDex-objmodels"
        if not extract_dir.exists() or cfg.force:
            extract_zip(zip_path, extract_dir, force=cfg.force)
            report.notes.append(f"extracted: {relative_to_repo(cfg.repo_root, extract_dir)}")

        return report

    @staticmethod
    def _resolve_models_root(extract_root: Path) -> Path:
        # Expected RealDex layout:
        # assets/objects/raw/RealDex/RealDex-objmodels/models/*.obj
        direct = extract_root / "models"
        if direct.is_dir():
            return direct
        candidates = [p for p in extract_root.rglob("models") if p.is_dir()]
        if candidates:
            return sorted(candidates)[0]
        return extract_root

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        extract_root = cfg.source_download_dir / "RealDex-objmodels"
        src_root = self._resolve_models_root(extract_root)
        processed_dir = cfg.source_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not extract_root.exists():
            report.notes.append(f"missing extracted source dir: {extract_root}")
            return report

        obj_files = sorted(p for p in src_root.rglob("*.obj") if p.is_file())
        if not obj_files:
            report.notes.append(f"no .obj found under: {src_root}")
            return report

        seen_ids: set[str] = set()
        for src_obj in obj_files:
            base_name = sanitize_object_id(src_obj.stem)
            base_id = sanitize_object_id(f"{self.source_name}_{base_name}")
            object_id = base_id
            suffix = 1
            while object_id in seen_ids:
                suffix += 1
                object_id = f"{base_id}_{suffix}"
            seen_ids.add(object_id)

            dst_dir = processed_dir / object_id
            if dst_dir.exists() and cfg.force:
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst_obj = dst_dir / "raw.obj"
            if dst_obj.exists() and not cfg.force:
                continue

            # RealDex object models are white-mesh OBJ files without texture.
            # Keep only a normalized object mesh path in raw.
            shutil.copy2(src_obj, dst_obj)
            report.organized_objects += 1

        report.notes.append("RealDex organize mode: OBJ-only (no texture assets copied)")
        self.write_manifest_for_organize(cfg, report)
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        raw_dir = cfg.source_processed_dir
        manifest = IngestManifest.create(dataset=self.source_name, version=self.version)
        manifest.source = ManifestSource(
            homepage="https://github.com/4DVLab/RealDex",
            download_method="google_drive",
            notes="Archive from RealDex-objmodels.zip",
        )

        if not raw_dir.exists():
            manifest.summary = ManifestSummary(
                num_objects=0,
                num_categories=0,
                has_texture_policy="unknown",
                default_mass_kg=DEFAULT_MASS_KG,
            )
            return manifest

        objects: list[ObjectRecord] = []
        texture_true_count = 0
        texture_false_count = 0

        for obj_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
            object_id = obj_dir.name
            mesh_path = obj_dir / "raw.obj"
            if not mesh_path.exists():
                continue

            mtl_candidates = list(obj_dir.glob("*.mtl"))
            texture_files = [p.name for p in obj_dir.glob("*.png")]
            has_texture = "true" if texture_files else "false"
            if has_texture == "true":
                texture_true_count += 1
            else:
                texture_false_count += 1

            mtl_path = mtl_candidates[0] if mtl_candidates else None
            objects.append(
                ObjectRecord(
                    object_id=object_id,
                    name=object_id,
                    category=None,
                    mesh_path=relative_to_repo(cfg.repo_root, mesh_path),
                    mesh_format="obj",
                    mass_kg=DEFAULT_MASS_KG,
                    has_texture=has_texture,
                    mtl_path=(relative_to_repo(cfg.repo_root, mtl_path) if mtl_path else None),
                    texture_files=texture_files,
                )
            )

        if texture_true_count and texture_false_count:
            texture_policy = "mixed"
        elif texture_true_count:
            texture_policy = "all"
        elif texture_false_count:
            texture_policy = "none"
        else:
            texture_policy = "unknown"

        manifest.objects = objects
        manifest.summary = ManifestSummary(
            num_objects=len(objects),
            num_categories=0,
            has_texture_policy=texture_policy,
            default_mass_kg=DEFAULT_MASS_KG,
        )
        return manifest
