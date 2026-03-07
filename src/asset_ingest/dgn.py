"""DGN adapter: download/extract archive, organize into DGN/DDG processed datasets."""

from __future__ import annotations

from pathlib import Path
import shutil

from .base import (
    CANONICAL_RAW_OBJ_NAME,
    DEFAULT_MASS_KG,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
    download_url_file,
    extract_zip,
    relative_to_repo,
    sanitize_object_id,
)
from .manifest import IngestManifest, ManifestSource, ManifestSummary, ObjectRecord

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None


class DGNAdapter(BaseIngestAdapter):
    source_name = "DGN"
    version = None

    default_url = (
        "https://huggingface.co/datasets/JiayiChenPKU/BODex/resolve/main/"
        "object_assets/DGN_obj_raw.zip"
    )
    archive_name = "DGN_obj_raw.zip"
    extract_dirname = "DGN_obj_raw"

    @classmethod
    def _resolve_extract_root(cls, source_download_dir: Path) -> Path:
        root = source_download_dir / cls.extract_dirname
        if not root.exists():
            return root
        top_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if len(top_dirs) == 1:
            # Common zip layout: DGN_obj_raw/<single_root>/...
            maybe = top_dirs[0]
            if any(x.is_file() and x.suffix.lower() == ".obj" for x in maybe.rglob("*.obj")):
                return maybe
        return root

    def download(self, cfg: IngestConfig) -> DownloadReport:
        out_dir = cfg.source_download_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report = DownloadReport(source=self.source_name)

        url = self.default_url
        archive_path = out_dir / self.archive_name
        if not archive_path.exists() or cfg.force:
            download_url_file(url, archive_path, desc=f"download {self.source_name} archive")
            report.downloaded_files.append(relative_to_repo(cfg.repo_root, archive_path))
        else:
            report.notes.append(f"skip existing: {archive_path.name}")

        extract_dir = out_dir / self.extract_dirname
        if not extract_dir.exists() or cfg.force:
            extract_zip(archive_path, extract_dir, force=cfg.force, show_progress=True)
            report.notes.append(f"extracted: {relative_to_repo(cfg.repo_root, extract_dir)}")
        else:
            report.notes.append(f"skip existing extract dir: {relative_to_repo(cfg.repo_root, extract_dir)}")

        return report

    def _build_manifest_for_dataset(self, cfg: IngestConfig, dataset_name: str) -> IngestManifest:
        processed_dir = cfg.processed_root / dataset_name
        manifest = IngestManifest.create(dataset=dataset_name, version=self.version)
        manifest.source = ManifestSource(
            homepage="https://huggingface.co/datasets/JiayiChenPKU/BODex",
            download_method="url_zip",
            notes="Derived from DGN_obj_raw.zip; object split by prefix into DGN / DDG",
        )
        if not processed_dir.exists():
            manifest.summary = ManifestSummary(
                num_objects=0,
                num_categories=0,
                has_texture_policy="none",
                default_mass_kg=DEFAULT_MASS_KG,
            )
            return manifest

        objects: list[ObjectRecord] = []
        for obj_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
            mesh_path = obj_dir / CANONICAL_RAW_OBJ_NAME
            if not mesh_path.exists():
                continue
            objects.append(
                ObjectRecord(
                    object_id=obj_dir.name,
                    name=obj_dir.name,
                    category=None,
                    mesh_path=relative_to_repo(cfg.repo_root, mesh_path),
                    mesh_format="obj",
                    mass_kg=DEFAULT_MASS_KG,
                    has_texture="false",
                    mtl_path=None,
                    texture_files=[],
                )
            )
        manifest.objects = objects
        manifest.summary = ManifestSummary(
            num_objects=len(objects),
            num_categories=0,
            has_texture_policy="none",
            default_mass_kg=DEFAULT_MASS_KG,
        )
        return manifest

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        src_root = self._resolve_extract_root(cfg.source_download_dir)
        dgn_processed_dir = cfg.processed_root / "DGN"
        ddg_processed_dir = cfg.processed_root / "DDG"
        if cfg.force:
            if dgn_processed_dir.exists():
                shutil.rmtree(dgn_processed_dir)
            if ddg_processed_dir.exists():
                shutil.rmtree(ddg_processed_dir)
        dgn_processed_dir.mkdir(parents=True, exist_ok=True)
        ddg_processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not src_root.exists():
            report.notes.append(f"missing extracted source dir: {src_root}")
            return report

        obj_files = sorted([p for p in src_root.rglob("*.obj") if p.is_file()])
        if not obj_files:
            report.notes.append(f"no .obj found under: {src_root}")
            return report
        if tqdm is not None:
            obj_iter = tqdm(obj_files, desc=f"{self.source_name} organize", unit="obj")
        else:
            obj_iter = obj_files

        seen_dgn: set[str] = set()
        seen_ddg: set[str] = set()
        ddg_count = 0
        dgn_count = 0

        for src_obj in obj_iter:
            raw_name = sanitize_object_id(src_obj.stem)
            if not raw_name:
                continue

            # 1) Always keep full DGN set.
            base_id_dgn = sanitize_object_id(f"DGN_{raw_name}")
            object_id_dgn = base_id_dgn
            suffix = 1
            while object_id_dgn in seen_dgn:
                suffix += 1
                object_id_dgn = f"{base_id_dgn}_{suffix}"
            seen_dgn.add(object_id_dgn)

            dgn_dst_dir = dgn_processed_dir / object_id_dgn
            if dgn_dst_dir.exists() and cfg.force:
                shutil.rmtree(dgn_dst_dir)
            dgn_dst_dir.mkdir(parents=True, exist_ok=True)
            dgn_dst_obj = dgn_dst_dir / CANONICAL_RAW_OBJ_NAME
            if not dgn_dst_obj.exists() or cfg.force:
                shutil.copy2(src_obj, dgn_dst_obj)
                dgn_count += 1
                report.organized_objects += 1

            # 2) Additionally export DDG subset (prefix ddg*).
            if raw_name.lower().startswith("ddg"):
                ddg_suffix = raw_name
                raw_lower = raw_name.lower()
                if raw_lower.startswith("ddg_"):
                    ddg_suffix = raw_name[4:]
                elif raw_lower.startswith("ddg-"):
                    ddg_suffix = raw_name[4:]
                elif raw_lower.startswith("ddg"):
                    ddg_suffix = raw_name[3:]
                ddg_suffix = sanitize_object_id(ddg_suffix).lstrip("_-")
                if not ddg_suffix:
                    ddg_suffix = raw_name
                base_id_ddg = sanitize_object_id(f"DDG_{ddg_suffix}")
                object_id_ddg = base_id_ddg
                suffix_ddg = 1
                while object_id_ddg in seen_ddg:
                    suffix_ddg += 1
                    object_id_ddg = f"{base_id_ddg}_{suffix_ddg}"
                seen_ddg.add(object_id_ddg)

                ddg_dst_dir = ddg_processed_dir / object_id_ddg
                if ddg_dst_dir.exists() and cfg.force:
                    shutil.rmtree(ddg_dst_dir)
                ddg_dst_dir.mkdir(parents=True, exist_ok=True)
                ddg_dst_obj = ddg_dst_dir / CANONICAL_RAW_OBJ_NAME
                if not ddg_dst_obj.exists() or cfg.force:
                    shutil.copy2(src_obj, ddg_dst_obj)
                    ddg_count += 1

        # Keep ingest CLI compatibility: DGN source manifest at processed/DGN/manifest.json
        self.write_manifest_for_organize(cfg, report)
        # Also emit derived DDG manifest to make downstream pipeline consistent.
        ddg_manifest = self._build_manifest_for_dataset(cfg, "DDG")
        ddg_manifest_path = cfg.processed_root / "DDG" / "manifest.json"
        ddg_manifest.save(ddg_manifest_path)

        report.notes.append(f"organized from: {relative_to_repo(cfg.repo_root, src_root)}")
        report.notes.append(f"DGN objects (full set): {dgn_count}")
        report.notes.append(f"DDG objects (subset, prefix ddg*): {ddg_count}")
        report.notes.append(f"DDG manifest: {relative_to_repo(cfg.repo_root, ddg_manifest_path)}")
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        # By design this adapter's source manifest points to processed/DGN.
        return self._build_manifest_for_dataset(cfg, "DGN")
