"""DexNet adapter: download/extract, organize canonical assets, build manifest."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import (
    CANONICAL_MTL_NAME,
    CANONICAL_RAW_OBJ_NAME,
    DEFAULT_MASS_KG,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
    canonicalize_texture_assets,
    copy_file_with_progress,
    download_google_drive_file,
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


class DexNetAdapter(BaseIngestAdapter):
    source_name = "DexNet"
    version = None

    default_drive_file_id = "1dwzHMGI_bqekBoqpFgDRQSXsDIZj--v8"
    default_drive_file_url = (
        "https://drive.google.com/file/d/1dwzHMGI_bqekBoqpFgDRQSXsDIZj--v8/view"
    )
    default_archive_name = "DexNet_Object_Mesh_Dataset.zip"
    archive_env_name = "DEXNET_ARCHIVE"
    source_url_env_name = "DEXNET_URL"
    source_dir_env_name = "DEXNET_SOURCE_DIR"

    @staticmethod
    def _choose_obj(obj_paths: list[Path]) -> Path:
        names = {p.name.lower(): p for p in obj_paths}
        for key in ["textured.obj", "model.obj", "mesh.obj"]:
            if key in names:
                return names[key]
        return sorted(obj_paths)[0]

    @staticmethod
    def _iter_candidate_dirs(root: Path) -> list[Path]:
        """
        Candidate object folders are directories that contain at least one OBJ and
        whose direct children do not contain another OBJ directory layer.
        """
        all_dirs = sorted([p for p in root.rglob("*") if p.is_dir()])
        leaf_like: list[Path] = []
        for d in all_dirs:
            has_obj = any(x.is_file() and x.suffix.lower() == ".obj" for x in d.iterdir())
            if not has_obj:
                continue
            child_has_obj_dir = any(
                c.is_dir() and any(y.is_file() and y.suffix.lower() == ".obj" for y in c.iterdir())
                for c in d.iterdir()
            )
            if not child_has_obj_dir:
                leaf_like.append(d)
        return leaf_like

    def _archive_path(self, cfg: IngestConfig) -> Path:
        return cfg.source_download_dir / self.default_archive_name

    def _extract_archive(self, archive_path: Path, extract_dir: Path, force: bool) -> None:
        extract_zip(archive_path, extract_dir, force=force, show_progress=True)

    def download(self, cfg: IngestConfig) -> DownloadReport:
        out_dir = cfg.source_download_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report = DownloadReport(source=self.source_name)

        source_dir = os.environ.get(self.source_dir_env_name)
        if source_dir:
            src = Path(source_dir).expanduser().resolve()
            if not src.exists() or not src.is_dir():
                raise FileNotFoundError(f"{self.source_dir_env_name} invalid: {src}")
            link_dst = out_dir / "source"
            if link_dst.exists() and cfg.force:
                if link_dst.is_symlink() or link_dst.is_file():
                    link_dst.unlink()
                else:
                    shutil.rmtree(link_dst)
            if not link_dst.exists():
                try:
                    link_dst.symlink_to(src)
                    report.notes.append(f"linked source dir from {self.source_dir_env_name}")
                except OSError:
                    shutil.copytree(src, link_dst, dirs_exist_ok=True)
                    report.notes.append(f"copied source dir from {self.source_dir_env_name}")
            report.downloaded_files.append(relative_to_repo(cfg.repo_root, link_dst))
            return report

        archive_path = self._archive_path(cfg)

        local_archive = os.environ.get(self.archive_env_name)
        if local_archive and (not archive_path.exists() or cfg.force):
            src = Path(local_archive).expanduser().resolve()
            if not src.exists() or not src.is_file():
                raise FileNotFoundError(f"{self.archive_env_name} invalid: {src}")
            copy_file_with_progress(src, archive_path, desc=f"copy {self.source_name} archive")
            report.downloaded_files.append(relative_to_repo(cfg.repo_root, archive_path))

        if not archive_path.exists() or cfg.force:
            custom_url = os.environ.get(self.source_url_env_name)
            if custom_url:
                download_url_file(
                    custom_url,
                    archive_path,
                    desc=f"download {self.source_name} archive",
                )
                report.downloaded_files.append(relative_to_repo(cfg.repo_root, archive_path))
            else:
                download_google_drive_file(self.default_drive_file_id, archive_path)
                report.downloaded_files.append(relative_to_repo(cfg.repo_root, archive_path))
                report.notes.append(
                    "downloaded from default Google Drive source "
                    f"({self.default_drive_file_url})"
                )

        if not archive_path.exists():
            raise RuntimeError(
                f"{self.source_name} requires manual/authenticated data source. "
                f"Set {self.source_dir_env_name}=<dir> OR "
                f"{self.archive_env_name}=<archive> OR {self.source_url_env_name}=<url>, "
                f"or place archive at {archive_path}."
            )

        extract_dir = out_dir / "extracted"
        if not extract_dir.exists() or cfg.force:
            self._extract_archive(archive_path, extract_dir, force=cfg.force)
            report.notes.append(f"extracted: {relative_to_repo(cfg.repo_root, extract_dir)}")

        return report

    @staticmethod
    def _resolve_source_root(source_download_dir: Path) -> Path:
        source_link = source_download_dir / "source"
        if source_link.exists():
            return source_link
        extract_dir = source_download_dir / "extracted"
        if not extract_dir.exists():
            return source_download_dir
        top_dirs = sorted(p for p in extract_dir.iterdir() if p.is_dir())
        if len(top_dirs) == 1:
            return top_dirs[0]
        return extract_dir

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        src_root = self._resolve_source_root(cfg.source_download_dir)
        processed_dir = cfg.source_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not src_root.exists():
            report.notes.append(f"missing source dir: {src_root}")
            return report

        seen_ids: set[str] = set()

        candidate_dirs = self._iter_candidate_dirs(src_root)
        if tqdm is not None:
            candidate_iter = tqdm(candidate_dirs, desc=f"{self.source_name} organize", unit="obj")
        else:
            candidate_iter = candidate_dirs

        for folder in candidate_iter:
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

            dst_obj = dst_dir / CANONICAL_RAW_OBJ_NAME
            if dst_obj.exists() and not cfg.force:
                continue

            shutil.copy2(chosen, dst_obj)
            # Search texture/material from current folder, then one level below.
            mtl_src = next((p for p in sorted(folder.glob("*.mtl")) if p.is_file()), None)
            tex_src = next((p for p in sorted(folder.glob("*.png")) if p.is_file()), None)
            if mtl_src is None:
                mtl_src = next((p for p in sorted(folder.glob("*/*.mtl")) if p.is_file()), None)
            if tex_src is None:
                tex_src = next((p for p in sorted(folder.glob("*/*.png")) if p.is_file()), None)
            canonicalize_texture_assets(
                dst_dir=dst_dir,
                raw_obj_path=dst_obj,
                texture_src=tex_src,
                mtl_src=mtl_src,
                create_mtl_if_texture=False,
            )
            report.organized_objects += 1

        report.notes.append(f"organized from {relative_to_repo(cfg.repo_root, src_root)}")
        self.write_manifest_for_organize(cfg, report)
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage="https://berkeleyautomation.github.io/dex-net/",
            download_method="google_drive_or_manual",
            notes=(
                "Dex-Net object mesh dataset archive "
                "(default Google Drive or manual source)"
            ),
            default_mass_kg=DEFAULT_MASS_KG,
            name_resolver=lambda object_id, _obj_dir: (
                object_id[len(f"{self.source_name}_") :]
                if object_id.startswith(f"{self.source_name}_")
                else object_id
            ),
        )
