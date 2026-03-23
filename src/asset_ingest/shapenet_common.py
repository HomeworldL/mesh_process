"""Shared utilities for ShapeNet HF archive adapters."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import os
import re
import shutil
import zipfile

from .base import (
    CANONICAL_MTL_NAME,
    CANONICAL_RAW_OBJ_NAME,
    CANONICAL_TEXTURE_NAME,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    canonicalize_texture_assets,
    relative_to_repo,
    tqdm,
)
from .manifest import IngestManifest


class _ShapeNetHFBaseAdapter(BaseIngestAdapter):
    hf_repo_id: str = ""
    homepage: str = "https://huggingface.co/datasets/ShapeNet"
    required_files: list[str] = []
    normalized_default_mass_kg: float = 50.0

    @staticmethod
    def _subset_filter(cfg: IngestConfig) -> set[str] | None:
        if not cfg.subset:
            return None
        vals = {x.strip() for x in cfg.subset.split(",") if x.strip()}
        return vals or None

    @staticmethod
    def _truthy_env(name: str) -> bool:
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

    @staticmethod
    def _is_complete(snapshot_dir: Path, required_files: list[str]) -> bool:
        for rel in required_files:
            p = snapshot_dir / rel
            if not p.exists() or not p.is_file() or p.stat().st_size <= 0:
                return False
        return True

    @staticmethod
    def _safe_extract_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, output_dir: Path) -> None:
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

    @staticmethod
    def _parse_mtl_texture_refs(
        mtl_path: Path,
        *,
        texture_line_prefixes: tuple[str, ...],
    ) -> list[str]:
        refs: list[str] = []
        if not mtl_path.exists():
            return refs
        try:
            for raw in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                if parts[0].lower() not in texture_line_prefixes:
                    continue
                tex_name = Path(parts[-1]).name
                if tex_name and tex_name not in refs:
                    refs.append(tex_name)
        except Exception:
            return refs
        return refs

    @staticmethod
    def _copy_texture_to_png(src_path: Path, dst_png_path: Path) -> bool:
        try:
            from PIL import Image  # type: ignore

            with Image.open(src_path) as img:
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
                img.save(dst_png_path, format="PNG")
            return True
        except Exception:
            return False

    @staticmethod
    def _rewrite_mtl_with_local_textures(
        src_mtl: Path,
        dst_mtl: Path,
        texture_rename_map: dict[str, str],
        *,
        texture_line_prefixes: tuple[str, ...],
    ) -> None:
        out_lines: list[str] = []
        try:
            lines = src_mtl.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            lines = []

        for raw in lines:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                out_lines.append(line)
                continue
            parts = stripped.split()
            if len(parts) >= 2 and parts[0].lower() in texture_line_prefixes:
                tex_name = Path(parts[-1]).name
                if tex_name in texture_rename_map:
                    parts[-1] = texture_rename_map[tex_name]
                    prefix = re.match(r"^\s*", line).group(0) if line else ""
                    out_lines.append(prefix + " ".join(parts))
                    continue
            out_lines.append(line)

        if not out_lines:
            out_lines = ["newmtl material_0"]
        dst_mtl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    @staticmethod
    def _bake_single_texture_png(
        dst_dir: Path,
        raw_obj_path: Path,
        src_mtl_path: Path,
    ) -> tuple[bool, str]:
        """
        Bake potentially multi-material/multi-texture OBJ into one texture atlas PNG.
        Returns (success, message).
        """
        if not raw_obj_path.is_file() or not src_mtl_path.is_file():
            return False, "missing raw.obj or textured.mtl for bake"

        try:
            import trimesh  # type: ignore
        except Exception as e:
            return False, f"trimesh unavailable: {e}"

        bake_tmp = dst_dir / ".bake_tmp"
        if bake_tmp.exists():
            shutil.rmtree(bake_tmp, ignore_errors=True)
        bake_tmp.mkdir(parents=True, exist_ok=True)

        try:
            scene_or_mesh = trimesh.load(
                str(raw_obj_path),
                force="scene",
                process=False,
                maintain_order=True,
            )
            if isinstance(scene_or_mesh, trimesh.Scene):
                meshes = []
                for geom in scene_or_mesh.geometry.values():
                    if not isinstance(geom, trimesh.Trimesh):
                        continue
                    if geom.faces is None or len(geom.faces) == 0:
                        continue
                    meshes.append(geom)
                if not meshes:
                    return False, "no mesh geometry loaded for bake"
                mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
            elif isinstance(scene_or_mesh, trimesh.Trimesh):
                mesh = scene_or_mesh
            else:
                return False, f"unsupported trimesh type: {type(scene_or_mesh).__name__}"

            # Reduce duplicated OBJ-expanded vertices while keeping UV seams by default.
            mesh.merge_vertices()

            baked_obj = bake_tmp / CANONICAL_RAW_OBJ_NAME
            mesh.export(baked_obj)

            baked_mtl = next((p for p in sorted(bake_tmp.glob("*.mtl")) if p.is_file()), None)
            baked_tex = next((p for p in sorted(bake_tmp.glob("*.png")) if p.is_file()), None)
            if baked_tex is None:
                baked_tex = next(
                    (p for p in sorted(bake_tmp.iterdir()) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}),
                    None,
                )

            if baked_mtl is None or baked_tex is None or not baked_obj.exists():
                return False, "bake export missing obj/mtl/texture outputs"

            shutil.copy2(baked_obj, raw_obj_path)
            canonicalize_texture_assets(
                dst_dir=dst_dir,
                raw_obj_path=raw_obj_path,
                texture_src=baked_tex,
                mtl_src=baked_mtl,
                create_mtl_if_texture=True,
            )

            # Enforce PNG-only policy in organized outputs.
            for p in dst_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}:
                    p.unlink(missing_ok=True)

            return True, "baked to single texture_map.png"
        except Exception as e:
            return False, f"bake failed: {e}"
        finally:
            shutil.rmtree(bake_tmp, ignore_errors=True)

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
