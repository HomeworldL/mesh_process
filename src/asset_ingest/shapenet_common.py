"""Shared utilities for ShapeNet HF archive adapters."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import math
import os
import shutil
import zipfile

from .base import (
    CANONICAL_MTL_NAME,
    CANONICAL_RAW_OBJ_NAME,
    CANONICAL_TEXTURE_NAME,
    DEFAULT_MASS_KG,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    canonicalize_texture_assets,
    relative_to_repo,
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
    def _normalize_obj_center_and_scale(
        src_obj_path: Path,
        dst_obj_path: Path,
        *,
        target_max_extent: float = 1.0,
        min_extent_eps: float = 1e-9,
        max_aspect_ratio: float = 1e5,
    ) -> tuple[bool, str, dict[str, float]]:
        """
        Load OBJ for validation, then normalize geometry:
        - center to origin by AABB center
        - uniform scale so max AABB extent == target_max_extent

        Returns:
          (success, reason, metrics)
        metrics keys:
          pre_min_extent, pre_max_extent, pre_aspect_ratio, post_max_extent, post_center_norm
        """
        metrics: dict[str, float] = {}
        try:
            import trimesh  # type: ignore
        except Exception as e:
            return False, f"trimesh unavailable: {e}", metrics

        try:
            loaded = trimesh.load(
                str(src_obj_path),
                force="mesh",
                process=False,
                maintain_order=True,
            )
        except Exception as e:
            return False, f"failed to load obj: {e}", metrics

        if not isinstance(loaded, trimesh.Trimesh) or loaded.vertices is None or len(loaded.vertices) == 0:
            return False, "loaded object has no valid mesh vertices", metrics

        bounds = loaded.bounds
        if bounds is None or len(bounds) != 2:
            return False, "loaded object has invalid bounds", metrics

        extent = bounds[1] - bounds[0]
        if not all(math.isfinite(float(x)) for x in extent):
            return False, "non-finite bounds extent", metrics

        pre_max_extent = float(max(extent))
        pre_min_extent = float(min(extent))
        pre_aspect_ratio = float(pre_max_extent / max(pre_min_extent, min_extent_eps))
        metrics["pre_max_extent"] = pre_max_extent
        metrics["pre_min_extent"] = pre_min_extent
        metrics["pre_aspect_ratio"] = pre_aspect_ratio

        if pre_max_extent <= min_extent_eps:
            return False, f"degenerate mesh extent: max_extent={pre_max_extent:.3e}", metrics
        if pre_aspect_ratio > max_aspect_ratio:
            return (
                False,
                (
                    "extreme aspect ratio before normalization: "
                    f"{pre_aspect_ratio:.3e} > {max_aspect_ratio:.3e}"
                ),
                metrics,
            )

        center = (bounds[0] + bounds[1]) * 0.5
        scale = float(target_max_extent / pre_max_extent)

        try:
            lines = src_obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as e:
            return False, f"failed reading obj text: {e}", metrics

        out_lines = list(lines)
        vertex_count = 0
        for idx, line in enumerate(lines):
            if not line.startswith("v "):
                continue
            toks = line.split()
            if len(toks) < 4:
                continue
            try:
                vx = (float(toks[1]) - float(center[0])) * scale
                vy = (float(toks[2]) - float(center[1])) * scale
                vz = (float(toks[3]) - float(center[2])) * scale
            except Exception:
                continue
            out_lines[idx] = f"v {vx:.9f} {vy:.9f} {vz:.9f}"
            vertex_count += 1

        if vertex_count == 0:
            return False, "no valid 'v' records found in obj text", metrics

        dst_obj_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

        try:
            post_mesh = trimesh.load(
                str(dst_obj_path),
                force="mesh",
                process=False,
                maintain_order=True,
            )
        except Exception as e:
            return False, f"failed to reload normalized obj: {e}", metrics

        if not isinstance(post_mesh, trimesh.Trimesh) or post_mesh.bounds is None:
            return False, "normalized obj has invalid bounds", metrics

        post_extent = post_mesh.bounds[1] - post_mesh.bounds[0]
        if not all(math.isfinite(float(x)) for x in post_extent):
            return False, "normalized obj extent is non-finite", metrics
        post_max_extent = float(max(post_extent))
        post_center = (post_mesh.bounds[0] + post_mesh.bounds[1]) * 0.5
        post_center_norm = float(math.sqrt(float(post_center[0]) ** 2 + float(post_center[1]) ** 2 + float(post_center[2]) ** 2))
        metrics["post_max_extent"] = post_max_extent
        metrics["post_center_norm"] = post_center_norm

        if not (0.95 * target_max_extent <= post_max_extent <= 1.05 * target_max_extent):
            return (
                False,
                (
                    "normalized size out of expected range: "
                    f"post_max_extent={post_max_extent:.6f}, target={target_max_extent:.6f}"
                ),
                metrics,
            )
        if post_center_norm > 5e-3:
            return (
                False,
                (
                    "normalized center drift too large: "
                    f"center_norm={post_center_norm:.6e}"
                ),
                metrics,
            )

        return True, "normalized", metrics

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
