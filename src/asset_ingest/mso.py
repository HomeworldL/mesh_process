"""MuJoCo Scanned Objects (MSO) adapter via GitHub mirror repository."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess

from .base import (
    DEFAULT_MASS_KG,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
    canonicalize_texture_assets,
    relative_to_repo,
    sanitize_object_id,
)
from .filter_lists import MSO_BAD_INSTANCES
from .manifest import IngestManifest


class MSOAdapter(BaseIngestAdapter):
    source_name = "MSO"
    version = None

    repo_url = "https://github.com/kevinzakka/mujoco_scanned_objects.git"
    repo_dirname = "mujoco_scanned_objects"

    @staticmethod
    def _run_git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
        cmd = ["git", *args]
        where = str(cwd) if cwd is not None else os.getcwd()
        print(f"[MSO] ({where}) $ {' '.join(cmd)}", flush=True)
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
        )

    @classmethod
    def _clone_sparse_models_only(cls, repo_dir: Path) -> None:
        cls._run_git(
            [
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--progress",
                cls.repo_url,
                str(repo_dir),
            ]
        )
        cls._run_git(["sparse-checkout", "set", "models"], cwd=repo_dir)

    @classmethod
    def _update_existing_repo(cls, repo_dir: Path) -> None:
        # Keep updates conservative to avoid local corruption; only fast-forward pull.
        cls._run_git(["sparse-checkout", "set", "models"], cwd=repo_dir)
        cls._run_git(["pull", "--ff-only", "--depth", "1", "--progress"], cwd=repo_dir)

    def download(self, cfg: IngestConfig) -> DownloadReport:
        out_dir = cfg.source_download_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report = DownloadReport(source=self.source_name)

        if shutil.which("git") is None:
            raise RuntimeError("MSO download requires git. Please install git first.")

        repo_dir = out_dir / self.repo_dirname
        max_retries = 3
        last_error: str | None = None

        for attempt in range(1, max_retries + 1):
            try:
                if repo_dir.exists() and cfg.force:
                    shutil.rmtree(repo_dir)

                if not repo_dir.exists():
                    self._clone_sparse_models_only(repo_dir)
                    report.notes.append("download mode: git clone (sparse models)")
                else:
                    self._update_existing_repo(repo_dir)
                    report.notes.append("download mode: git pull --ff-only")

                report.downloaded_files.append(relative_to_repo(cfg.repo_root, repo_dir))
                models_dir = repo_dir / "models"
                if not models_dir.exists():
                    raise RuntimeError("download finished but models/ is missing")
                report.notes.append(f"models root ready: {relative_to_repo(cfg.repo_root, models_dir)}")
                return report
            except Exception as exc:
                if isinstance(exc, subprocess.CalledProcessError):
                    last_error = f"git exited with code {exc.returncode}"
                else:
                    last_error = str(exc)
                report.notes.append(f"attempt {attempt}/{max_retries} failed: {last_error}")
                if repo_dir.exists() and attempt < max_retries:
                    # If repo is left in a broken state, clean and retry fresh clone.
                    shutil.rmtree(repo_dir, ignore_errors=True)

        raise RuntimeError(f"MSO download failed after {max_retries} attempts: {last_error}")

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        repo_dir = cfg.source_download_dir / self.repo_dirname
        src_root = repo_dir / "models"
        processed_dir = cfg.source_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not src_root.exists():
            report.notes.append(f"missing models dir: {src_root}")
            return report

        seen_ids: set[str] = set()
        for folder in sorted(p for p in src_root.iterdir() if p.is_dir()):
            if folder.name in MSO_BAD_INSTANCES:
                continue
            src_obj = folder / "model.obj"
            if not src_obj.is_file():
                continue

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

            # Keep only normalized mesh path + canonical texture bundle.
            shutil.copy2(src_obj, dst_obj)
            src_tex = folder / "texture.png"
            canonicalize_texture_assets(
                dst_dir=dst_dir,
                raw_obj_path=dst_obj,
                texture_src=(src_tex if src_tex.is_file() else None),
                mtl_src=None,
                create_mtl_if_texture=True,
                default_material_name="material_0",
            )

            report.organized_objects += 1

        report.notes.append(
            "organized from mujoco_scanned_objects/models; normalized to raw.obj + "
            "texture_map.png + textured.mtl (when texture exists)"
        )
        self.write_manifest_for_organize(cfg, report)
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage="https://github.com/kevinzakka/mujoco_scanned_objects",
            download_method="git",
            notes="Sparse clone of models/ from mujoco_scanned_objects",
            default_mass_kg=DEFAULT_MASS_KG,
        )
