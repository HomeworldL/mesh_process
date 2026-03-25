"""DGN adapter: download/extract archive, organize into the DGN processed dataset."""

from __future__ import annotations

from pathlib import Path
import shutil

from .base import (
    CANONICAL_RAW_OBJ_NAME,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
    download_url_file,
    extract_zip,
    relative_to_repo,
    sanitize_object_id,
)
from .manifest import IngestManifest

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm optional
    tqdm = None


class DGNAdapter(BaseIngestAdapter):
    source_name = "DGN"
    version = None
    normalized_default_mass_kg = 50.0

    default_url = (
        "https://huggingface.co/datasets/JiayiChenPKU/BODex/resolve/main/"
        "object_assets/DGN_obj_raw.zip"
    )
    archive_name = "DGN_obj_raw.zip"
    extract_dirname = "DGN_obj_raw"

    def _skip_zip_member(self, member_name: str) -> bool:
        # EN: Drop macOS metadata files during unzip so they never appear as
        # candidate stage-1 meshes.
        # ZH: 解压时跳过 macOS 元数据文件，避免它们进入一阶段 mesh 候选集。
        normalized = member_name.replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
        if any(p == "__MACOSX" for p in parts):
            return True
        basename = parts[-1] if parts else normalized
        return basename.startswith("._")

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
            extract_zip(
                archive_path,
                extract_dir,
                force=cfg.force,
                show_progress=True,
                skip_member=self._skip_zip_member,
            )
            report.notes.append(f"extracted: {relative_to_repo(cfg.repo_root, extract_dir)}")
        else:
            report.notes.append(f"skip existing extract dir: {relative_to_repo(cfg.repo_root, extract_dir)}")

        return report

    def _build_manifest_for_dataset(self, cfg: IngestConfig, dataset_name: str) -> IngestManifest:
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage="https://huggingface.co/datasets/JiayiChenPKU/BODex",
            download_method="url_zip",
            notes="Derived from DGN_obj_raw.zip",
            default_mass_kg=self.normalized_default_mass_kg,
            dataset_name=dataset_name,
            processed_dir=cfg.processed_root / dataset_name,
            missing_has_texture_policy="none",
            mtl_resolver=lambda _obj_dir: None,
        )

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        src_root = self._resolve_extract_root(cfg.source_download_dir)
        dgn_processed_dir = cfg.processed_root / "DGN"
        if cfg.force:
            if dgn_processed_dir.exists():
                shutil.rmtree(dgn_processed_dir)
        dgn_processed_dir.mkdir(parents=True, exist_ok=True)
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
        dgn_count = 0
        normalized_count = 0
        dropped_bad_scale = 0

        for src_obj in obj_iter:
            raw_name = sanitize_object_id(src_obj.stem)
            if not raw_name:
                continue

            # EN: Always export the full source object into the DGN dataset.
            # Preserve source naming (after sanitization) instead of adding a
            # dataset prefix, so DGN ids stay close to the original archive ids.
            # ZH: 每个源对象都先进入完整 DGN 集合。对象 id 仅做 sanitize，
            # 不额外添加数据集前缀，以保持与源压缩包命名尽量接近。
            base_id_dgn = raw_name
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
                ok_norm, norm_reason, norm_mesh, norm_metrics = self.load_normalized_obj_mesh(
                    src_obj_path=src_obj,
                )
                if not ok_norm or norm_mesh is None:
                    dropped_bad_scale += 1
                    report.failed_items.append(
                        f"drop {src_obj.name}: normalize failed ({norm_reason}); metrics={norm_metrics}"
                    )
                    shutil.rmtree(dgn_dst_dir, ignore_errors=True)
                    continue
                try:
                    self.export_trimesh_obj_assets(
                        norm_mesh,
                        dgn_dst_dir,
                        export_texture=False,
                    )
                except Exception as exc:
                    dropped_bad_scale += 1
                    report.failed_items.append(
                        f"drop {src_obj.name}: export failed ({exc}); metrics={norm_metrics}"
                    )
                    shutil.rmtree(dgn_dst_dir, ignore_errors=True)
                    continue
                ok_post, post_reason = self.validate_normalized_obj_export(
                    dgn_dst_obj,
                    target_max_extent=1.0,
                    metrics=norm_metrics,
                )
                if not ok_post:
                    dropped_bad_scale += 1
                    report.failed_items.append(
                        f"drop {src_obj.name}: normalize export invalid ({post_reason}); metrics={norm_metrics}"
                    )
                    shutil.rmtree(dgn_dst_dir, ignore_errors=True)
                    continue
                normalized_count += 1
                dgn_count += 1
                report.organized_objects += 1

        self.write_manifest_for_organize(cfg, report)

        report.notes.append(f"organized from: {relative_to_repo(cfg.repo_root, src_root)}")
        report.notes.append(
            "DGN normalize policy: center=AABB center to origin, scale=max_extent to 1.0; "
            "drop if pre_max_extent<=1e-9 or pre_aspect_ratio>1e5 or post_max_extent not in [0.95,1.05] or post_center_norm>5e-3"
        )
        report.notes.append(
            f"DGN normalize stats: normalized={normalized_count}, dropped_bad_scale={dropped_bad_scale}"
        )
        report.notes.append(f"DGN objects: {dgn_count}")
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        # By design this adapter's source manifest points to processed/DGN.
        return self._build_manifest_for_dataset(cfg, "DGN")
