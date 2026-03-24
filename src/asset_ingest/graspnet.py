"""GraspNet source adapter: download object models archive and normalize meshes."""

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
from .manifest import IngestManifest


class GraspNetAdapter(BaseIngestAdapter):
    source_name = "GraspNet"
    version = None

    zip_file_id = "1Gxwu2C5wRQ0QwjdA8CbMXx-bYf_wwPT5"
    zip_name = "models.zip"

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

        extract_dir = out_dir / "models"
        if not extract_dir.exists() or cfg.force:
            extract_zip(zip_path, extract_dir, force=cfg.force)
            report.notes.append(f"extracted: {relative_to_repo(cfg.repo_root, extract_dir)}")

        return report

    def _choose_obj(self, obj_paths: list[Path]) -> Path:
        names = {p.name.lower(): p for p in obj_paths}
        for key in ["textured.obj", "nontextured.obj", "model.obj", "mesh.obj"]:
            if key in names:
                return names[key]
        return sorted(obj_paths)[0]

    def _resolve_models_root(self, source_root: Path) -> Path:
        # Expected layout after extraction:
        # raw/GraspNet/models/models/<object_id>/
        direct = source_root / "models" / "models"
        if direct.is_dir():
            return direct
        fallback = source_root / "models"
        if fallback.is_dir():
            return fallback
        candidates = [p for p in source_root.rglob("models") if p.is_dir()]
        if candidates:
            return sorted(candidates, key=lambda p: len(p.parts), reverse=True)[0]
        return source_root

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        source_root = cfg.source_download_dir
        src_root = self._resolve_models_root(source_root)
        processed_dir = cfg.source_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not source_root.exists():
            report.notes.append(f"missing extracted source dir: {source_root}")
            return report

        seen_ids: set[str] = set()
        for folder in sorted(p for p in src_root.iterdir() if p.is_dir()):
            objs = [x for x in folder.iterdir() if x.is_file() and x.suffix.lower() == ".obj"]
            if not objs:
                continue

            chosen = self._choose_obj(objs)
            base_name = sanitize_object_id(folder.name)
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

            ok_load, load_reason, mesh = self.load_obj_mesh(
                chosen,
                remove_unreferenced_vertices=False,
            )
            if not ok_load or mesh is None:
                report.failed_items.append(f"load failed for {object_id}: {load_reason}")
                shutil.rmtree(dst_dir, ignore_errors=True)
                continue
            try:
                export_texture = self.mesh_has_texture(mesh)
                if not export_texture:
                    mesh = mesh.copy()
                    mesh.remove_unreferenced_vertices()
                self.export_trimesh_obj_assets(
                    mesh,
                    dst_dir,
                    export_texture=export_texture,
                )
            except Exception as exc:
                report.failed_items.append(f"export failed for {object_id}: {exc}")
                shutil.rmtree(dst_dir, ignore_errors=True)
                continue
            report.organized_objects += 1

        report.notes.append(f"organized from {src_root}")
        self.write_manifest_for_organize(cfg, report)
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage="https://graspnet.net/datasets.html",
            download_method="google_drive",
            notes="objects archive models.zip",
            default_mass_kg=DEFAULT_MASS_KG,
        )
