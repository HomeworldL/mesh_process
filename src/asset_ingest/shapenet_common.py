"""Shared utilities for ShapeNet HF archive adapters."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import os
import shutil
import zipfile

from .base import (
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    relative_to_repo,
    tqdm,
)
from .manifest import IngestManifest


class _ShapeNetHFBaseAdapter(BaseIngestAdapter):
    hf_repo_id: str = ""
    homepage: str = "https://huggingface.co/datasets/ShapeNet"
    required_files: list[str] = []
    normalized_default_mass_kg: float = 50.0

    def _truthy_env(self, name: str) -> bool:
        v = os.environ.get(name, "").strip().lower()
        return v in {"1", "true", "yes", "on"}

    def _snapshot_dir(self, cfg: IngestConfig) -> Path:
        return cfg.source_download_dir / "hf_snapshot"

    def _required_files(self) -> list[str]:
        if self.required_files:
            return list(self.required_files)
        env_key = f"{self.source_name.upper()}_HF_REQUIRED_FILES"
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            return [x.strip() for x in env_val.split(",") if x.strip()]
        raise RuntimeError(
            f"No required file list configured for {self.source_name}. "
            f"Set class required_files or env {env_key}."
        )

    def _is_complete(self, snapshot_dir: Path, required_files: list[str]) -> bool:
        for rel in required_files:
            p = snapshot_dir / rel
            if not p.exists() or not p.is_file() or p.stat().st_size <= 0:
                return False
        return True

    def _safe_extract_member(self, zf: zipfile.ZipFile, info: zipfile.ZipInfo, output_dir: Path) -> None:
        member_path = PurePosixPath(info.filename)
        if ".." in member_path.parts:
            return
        out_path = output_dir / Path(*member_path.parts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info, "r") as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    def _extract_selected_members(
        self,
        zip_path: Path,
        output_dir: Path,
        include_predicate,
        force: bool,
        desc: str,
    ) -> tuple[int, int]:
        selected = 0
        extracted = 0
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = [i for i in zf.infolist() if not i.is_dir() and include_predicate(i.filename)]
            selected = len(infos)

            bar = None
            iterable = infos
            if tqdm is not None:
                bar = tqdm(infos, desc=desc, unit="file")
                iterable = bar

            for info in iterable:
                out_path = output_dir / Path(*PurePosixPath(info.filename).parts)
                if out_path.exists() and not force:
                    continue
                self._safe_extract_member(zf, info, output_dir)
                extracted += 1

            if bar is not None:
                bar.close()

        return selected, extracted

    def _prepare_after_download(self, cfg: IngestConfig, snapshot_dir: Path, report: DownloadReport) -> None:
        """Dataset-specific extraction prep; subclasses override."""

    def download(self, cfg: IngestConfig) -> DownloadReport:
        out_dir = cfg.source_download_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report = DownloadReport(source=self.source_name)

        snapshot_dir = self._snapshot_dir(cfg)
        required_files = self._required_files()

        if snapshot_dir.exists() and not cfg.force and self._is_complete(snapshot_dir, required_files):
            report.notes.append(f"reuse existing snapshot: {relative_to_repo(cfg.repo_root, snapshot_dir)}")
            report.notes.append(f"required files already present: {len(required_files)}")
            self._prepare_after_download(cfg, snapshot_dir, report)
            report.downloaded_files.append(relative_to_repo(cfg.repo_root, snapshot_dir))
            return report

        if snapshot_dir.exists() and cfg.force:
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import get_token, hf_hub_download
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required for ShapeNet download. "
                "Install with: pip install huggingface_hub"
            ) from e

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or get_token()
        filtered_files = required_files

        print(
            f"[{self.source_name}] preparing download: {len(filtered_files)} files "
            f"(token={'yes' if token else 'no'})",
            flush=True,
        )

        pbar = (
            tqdm(total=len(filtered_files), desc=f"{self.source_name} download", unit="file")
            if tqdm is not None
            else None
        )
        downloaded_count = 0
        reused_count = 0
        failed_files: list[str] = []

        for rel_path in filtered_files:
            local_path = snapshot_dir / rel_path
            if local_path.exists() and not cfg.force:
                reused_count += 1
                if pbar is not None:
                    pbar.update(1)
                continue
            try:
                print(f"[{self.source_name}] downloading {rel_path}", flush=True)
                hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename=rel_path,
                    repo_type="dataset",
                    token=token,
                    local_dir=str(snapshot_dir),
                    force_download=bool(cfg.force),
                )
                downloaded_count += 1
            except Exception as e:
                failed_files.append(f"{rel_path}: {e}")
            finally:
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        if failed_files:
            preview = "\n".join(failed_files[:10])
            raise RuntimeError(
                f"Failed to download {len(failed_files)} files from {self.hf_repo_id}. "
                f"First failures:\n{preview}"
            )
        if not self._is_complete(snapshot_dir, required_files):
            raise RuntimeError(
                f"Download finished but required files are incomplete under {snapshot_dir}."
            )

        self._prepare_after_download(cfg, snapshot_dir, report)

        report.downloaded_files.append(relative_to_repo(cfg.repo_root, snapshot_dir))
        report.notes.append(f"downloaded from HF dataset: {self.hf_repo_id}")
        report.notes.append(f"files downloaded: {downloaded_count}, reused: {reused_count}")
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage=self.homepage,
            download_method="huggingface_snapshot_download",
            notes=f"HF dataset repo: {self.hf_repo_id}",
            default_mass_kg=self.normalized_default_mass_kg,
            category_resolver=lambda object_id, _obj_dir: (
                object_id.split("_")[1] if len(object_id.split("_")) >= 3 else None
            ),
        )
