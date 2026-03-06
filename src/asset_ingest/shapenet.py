"""ShapeNet adapters (Hugging Face only).

Datasets:
- ShapeNet/ShapeNetCore-archive
- ShapeNet/ShapeNetSem-archive

Downloaded files are stored under:
  assets/objects/raw/<source>/hf_snapshot/
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil

from .base import (
    CANONICAL_MTL_NAME,
    CANONICAL_RAW_OBJ_NAME,
    DEFAULT_MASS_KG,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
    canonicalize_texture_assets,
    relative_to_repo,
    sanitize_object_id,
    tqdm,
)
from .manifest import IngestManifest, ManifestSource, ManifestSummary, ObjectRecord


class _ShapeNetHFBaseAdapter(BaseIngestAdapter):
    hf_repo_id: str = ""
    homepage: str = "https://huggingface.co/datasets/ShapeNet"
    required_files: list[str] = []

    @staticmethod
    def _subset_filter(cfg: IngestConfig) -> set[str] | None:
        if not cfg.subset:
            return None
        vals = {x.strip() for x in cfg.subset.split(",") if x.strip()}
        return vals or None

    @staticmethod
    def _choose_obj(obj_paths: list[Path]) -> Path:
        names = {p.name.lower(): p for p in obj_paths}
        priority = [
            "model_normalized.obj",
            "textured.obj",
            "model.obj",
            "mesh.obj",
        ]
        for key in priority:
            if key in names:
                return names[key]
        return sorted(obj_paths)[0]

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

    @staticmethod
    def _is_complete(snapshot_dir: Path, required_files: list[str]) -> bool:
        for rel in required_files:
            p = snapshot_dir / rel
            if not p.exists() or not p.is_file() or p.stat().st_size <= 0:
                return False
        return True

    def download(self, cfg: IngestConfig) -> DownloadReport:
        out_dir = cfg.source_download_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report = DownloadReport(source=self.source_name)

        snapshot_dir = self._snapshot_dir(cfg)
        required_files = self._required_files()
        if snapshot_dir.exists() and not cfg.force and self._is_complete(snapshot_dir, required_files):
            report.notes.append(f"reuse existing snapshot: {relative_to_repo(cfg.repo_root, snapshot_dir)}")
            report.notes.append(f"required files already present: {len(required_files)}")
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

        report.downloaded_files.append(relative_to_repo(cfg.repo_root, snapshot_dir))
        report.notes.append(f"downloaded from HF dataset: {self.hf_repo_id}")
        report.notes.append(f"files downloaded: {downloaded_count}, reused: {reused_count}")
        return report

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        src_root = self._snapshot_dir(cfg)
        processed_dir = cfg.source_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not src_root.exists():
            report.notes.append(f"missing source dir: {src_root}")
            return report

        subset = self._subset_filter(cfg)
        seen_ids = {p.name for p in processed_dir.iterdir() if p.is_dir()}

        candidate_folders: list[Path] = []
        for folder in sorted(p for p in src_root.rglob("*") if p.is_dir()):
            objs = [x for x in folder.iterdir() if x.is_file() and x.suffix.lower() == ".obj"]
            if objs:
                candidate_folders.append(folder)

        for folder in candidate_folders:
            rel = folder.relative_to(src_root).as_posix()
            rel_parts = rel.split("/")
            if subset is not None and not any(part in subset for part in rel_parts):
                continue

            objs = [x for x in folder.iterdir() if x.is_file() and x.suffix.lower() == ".obj"]
            if not objs:
                continue

            chosen = self._choose_obj(objs)
            rel_name = rel.replace("/", "_")
            base_name = sanitize_object_id(rel_name or folder.name)
            base_id = sanitize_object_id(f"{self.source_name}_{base_name}")
            object_id = base_id
            suffix = 1
            while object_id in seen_ids:
                suffix += 1
                object_id = f"{base_id}_{suffix}"
            seen_ids.add(object_id)

            out_dir = processed_dir / object_id
            if out_dir.exists() and cfg.force:
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            out_obj = out_dir / CANONICAL_RAW_OBJ_NAME
            if out_obj.exists() and not cfg.force:
                continue
            shutil.copy2(chosen, out_obj)

            # Texture/material lookup in current folder or parent.
            mtl_src = next((p for p in sorted(folder.glob("*.mtl")) if p.is_file()), None)
            tex_src = next((p for p in sorted(folder.glob("*.png")) if p.is_file()), None)
            if mtl_src is None:
                mtl_src = next((p for p in sorted(folder.parent.glob("*.mtl")) if p.is_file()), None)
            if tex_src is None:
                tex_src = next((p for p in sorted(folder.parent.glob("*.png")) if p.is_file()), None)

            canonicalize_texture_assets(
                dst_dir=out_dir,
                raw_obj_path=out_obj,
                texture_src=tex_src,
                mtl_src=mtl_src,
                create_mtl_if_texture=False,
            )
            report.organized_objects += 1

        report.notes.append(f"organized from HF snapshot: {relative_to_repo(cfg.repo_root, src_root)}")
        self.write_manifest_for_organize(cfg, report)
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        processed_dir = cfg.source_processed_dir
        manifest = IngestManifest.create(dataset=self.source_name, version=self.version)
        manifest.source = ManifestSource(
            homepage=self.homepage,
            download_method="huggingface_snapshot_download",
            notes=f"HF dataset repo: {self.hf_repo_id}",
        )

        if not processed_dir.exists():
            manifest.summary = ManifestSummary(
                num_objects=0,
                num_categories=0,
                has_texture_policy="unknown",
                default_mass_kg=DEFAULT_MASS_KG,
            )
            return manifest

        objects: list[ObjectRecord] = []
        categories = set()
        tex_true = 0
        tex_false = 0

        for obj_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
            object_id = obj_dir.name
            mesh_path = obj_dir / CANONICAL_RAW_OBJ_NAME
            if not mesh_path.exists():
                continue

            # Category heuristic: <Source>_<synset>_...
            category = None
            parts = object_id.split("_")
            if len(parts) >= 3:
                category = parts[1]
                categories.add(category)

            mtl_path = obj_dir / CANONICAL_MTL_NAME
            texture_files = [p.name for p in obj_dir.glob("*.png")]
            has_texture = "true" if texture_files else "false"
            if has_texture == "true":
                tex_true += 1
            else:
                tex_false += 1

            objects.append(
                ObjectRecord(
                    object_id=object_id,
                    name=object_id,
                    category=category,
                    mesh_path=relative_to_repo(cfg.repo_root, mesh_path),
                    mesh_format="obj",
                    mass_kg=DEFAULT_MASS_KG,
                    has_texture=has_texture,
                    mtl_path=(relative_to_repo(cfg.repo_root, mtl_path) if mtl_path.exists() else None),
                    texture_files=texture_files,
                )
            )

        if tex_true and tex_false:
            texture_policy = "mixed"
        elif tex_true:
            texture_policy = "all"
        elif tex_false:
            texture_policy = "none"
        else:
            texture_policy = "unknown"

        manifest.objects = objects
        manifest.summary = ManifestSummary(
            num_objects=len(objects),
            num_categories=len(categories),
            has_texture_policy=texture_policy,
            default_mass_kg=DEFAULT_MASS_KG,
        )
        return manifest


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
