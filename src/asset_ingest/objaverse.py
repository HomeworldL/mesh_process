"""Objaverse source adapter with chunked download and normalized organization."""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import random
import shutil

import trimesh

from .base import (
    DEFAULT_MASS_KG,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
    relative_to_repo,
    sanitize_object_id,
    tqdm,
)
from .manifest import IngestManifest


class ObjaverseAdapter(BaseIngestAdapter):
    source_name = "Objaverse"
    version = "v1"
    fixed_subset_label = "Daily-Used"
    normalized_default_mass_kg = 50.0
    category_anno_url = (
        "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/"
        "category_annotation.json"
    )

    def _normalize_loaded_mesh(
        self,
        mesh: trimesh.Trimesh,
        *,
        target_max_extent: float = 1.0,
        min_extent_eps: float = 1e-9,
        max_aspect_ratio: float = 1e5,
    ) -> tuple[bool, str, trimesh.Trimesh | None, dict[str, float]]:
        metrics: dict[str, float] = {}
        mesh = mesh.copy()

        bounds = mesh.bounds
        if bounds is None or len(bounds) != 2:
            return False, "mesh has invalid bounds after cleanup", None, metrics

        extent = bounds[1] - bounds[0]
        if not all(math.isfinite(float(x)) for x in extent):
            return False, "non-finite bounds extent", None, metrics

        pre_max_extent = float(max(extent))
        pre_min_extent = float(min(extent))
        pre_aspect_ratio = float(pre_max_extent / max(pre_min_extent, min_extent_eps))
        metrics["pre_max_extent"] = pre_max_extent
        metrics["pre_min_extent"] = pre_min_extent
        metrics["pre_aspect_ratio"] = pre_aspect_ratio
        metrics["kept_vertex_count"] = float(len(mesh.vertices))

        if pre_max_extent <= min_extent_eps:
            return False, f"degenerate mesh extent: max_extent={pre_max_extent:.3e}", None, metrics
        if pre_aspect_ratio > max_aspect_ratio:
            return (
                False,
                (
                    "extreme aspect ratio before normalization: "
                    f"{pre_aspect_ratio:.3e} > {max_aspect_ratio:.3e}"
                ),
                None,
                metrics,
            )

        center = mesh.bounding_box.centroid
        scale = float(target_max_extent / pre_max_extent)
        try:
            mesh.apply_translation(-center)
            mesh.apply_scale(scale)
        except Exception as exc:
            return False, f"failed to normalize mesh transform: {exc}", None, metrics
        return True, "normalized", mesh, metrics

    def _ensure_category_annotation(self, cache_root: Path) -> Path | None:
        anno_dir = cache_root / "hf-objaverse-v1"
        anno_dir.mkdir(parents=True, exist_ok=True)
        anno_path = anno_dir / "category_annotation.json"
        if anno_path.exists():
            return anno_path

        try:
            from .base import download_url_file

            download_url_file(
                self.category_anno_url,
                anno_path,
                desc="Objaverse category annotation",
            )
            return anno_path
        except Exception:
            return None

    def _select_uids(
        self,
        all_uids: set[str],
        cache_root: Path,
    ) -> tuple[list[str], list[str]]:
        notes: list[str] = []
        anno_path = self._ensure_category_annotation(cache_root)
        if anno_path is None:
            raise RuntimeError("Objaverse download requires category_annotation.json for fixed Daily-Used selection")

        with open(anno_path, "r", encoding="utf-8") as f:
            anno = json.load(f)
        picked_uids = sorted(
            {
                item["object_index"].split(".glb")[0]
                for item in anno
                if item.get("label") == self.fixed_subset_label and item.get("object_index")
            }.intersection(all_uids)
        )
        notes.append(
            f"Objaverse fixed subset={self.fixed_subset_label}, matched_uids={len(picked_uids)}"
        )
        return picked_uids, notes

    def _clean_cache_tmp(self, cache_root: Path) -> int:
        cleaned = 0
        glb_dir = cache_root / "hf-objaverse-v1" / "glbs"
        if not glb_dir.exists():
            return cleaned
        for tmp_path in glb_dir.rglob("*.tmp"):
            tmp_path.unlink(missing_ok=True)
            cleaned += 1
        return cleaned

    def _link_or_copy(self, src_path: Path, dst_path: Path) -> str:
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        try:
            dst_path.symlink_to(src_path)
            return "symlink"
        except OSError:
            shutil.copy2(src_path, dst_path)
            return "copy"

    def download(self, cfg: IngestConfig) -> DownloadReport:
        try:
            import objaverse
        except Exception as exc:
            raise RuntimeError("Objaverse download requires `objaverse` package: pip install objaverse") from exc

        out_dir = cfg.source_download_dir
        objects_dir = out_dir / "objects"
        cache_root = out_dir / "objaverse_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        objects_dir.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)

        report = DownloadReport(source=self.source_name)

        # Try to keep cache inside repo download dir instead of default ~/.objaverse.
        os.environ.setdefault("OBJAVERSE_HOME", str(cache_root))

        all_uids = set(objaverse.load_uids())
        uids, subset_notes = self._select_uids(all_uids=all_uids, cache_root=cache_root)
        report.notes.extend(subset_notes)

        cleaned = self._clean_cache_tmp(cache_root)
        if cleaned:
            report.notes.append(f"removed {cleaned} stale tmp files")

        # Chunked retry to avoid losing all progress when one batch fails.
        chunk_size = max(100, cfg.workers * 20)
        max_retries = 3
        written = 0
        mode_stats = {"symlink": 0, "copy": 0}

        chunk_starts = list(range(0, len(uids), chunk_size))
        chunk_iter = tqdm(chunk_starts, desc="Objaverse chunks", unit="chunk") if tqdm is not None else chunk_starts

        for i in chunk_iter:
            chunk = uids[i : i + chunk_size]
            last_error: Exception | None = None
            downloaded: dict[str, str] | None = None
            for _ in range(max_retries):
                try:
                    downloaded = objaverse.load_objects(
                        uids=chunk,
                        download_processes=max(1, cfg.workers),
                    )
                    break
                except Exception as exc:
                    last_error = exc
            if downloaded is None:
                report.notes.append(f"failed chunk [{i}:{i + chunk_size}] error={last_error}")
                continue

            item_iter = downloaded.items()
            if tqdm is not None:
                item_iter = tqdm(
                    downloaded.items(),
                    total=len(downloaded),
                    desc=f"mirror [{i}:{i + len(chunk)}]",
                    unit="obj",
                    leave=False,
                )

            for uid, src in item_iter:
                src_path = Path(src)
                dst_path = objects_dir / f"{uid}{src_path.suffix.lower()}"
                mode = self._link_or_copy(src_path=src_path, dst_path=dst_path)
                mode_stats[mode] += 1
                written += 1

        report.downloaded_files.append(relative_to_repo(cfg.repo_root, objects_dir))
        report.notes.append(
            f"selected_uids={len(uids)}, mirrored={written}, symlink={mode_stats['symlink']}, copy={mode_stats['copy']}"
        )
        report.notes.append(
            "If objaverse still writes to ~/.objaverse in your environment, you can symlink "
            "<repo>/assets/objects/raw/Objaverse/objaverse_cache to ~/.objaverse."
        )
        return report

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        src_dir = cfg.source_download_dir / "objects"
        processed_dir = cfg.source_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not src_dir.exists():
            report.notes.append(f"missing Objaverse objects dir: {src_dir}")
            return report

        mesh_paths = sorted(
            p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".glb", ".gltf", ".obj"}
        )
        total_candidates = len(mesh_paths)
        if cfg.sample_n is not None:
            if cfg.sample_n <= 0:
                raise ValueError("--sample-n must be a positive integer")
            sample_n = min(cfg.sample_n, total_candidates)
            rng = random.Random(cfg.sample_seed)
            mesh_paths = rng.sample(mesh_paths, sample_n)
            report.notes.append(
                f"objaverse organize sampled {sample_n}/{total_candidates} files "
                f"(seed={cfg.sample_seed})"
            )
        iterable = tqdm(mesh_paths, desc="Objaverse organize", unit="obj") if tqdm is not None else mesh_paths

        seen_ids: set[str] = set()
        for mesh_path in iterable:
            base_name = sanitize_object_id(mesh_path.stem)
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

            out_obj = out_dir / "raw.obj"
            if out_obj.exists() and not cfg.force:
                continue

            try:
                ok_load, load_reason, mesh = self.load_obj_mesh(
                    mesh_path,
                    remove_unreferenced_vertices=False,
                )
                if not ok_load or mesh is None:
                    report.failed_items.append(f"{mesh_path.name}: load failed ({load_reason})")
                    shutil.rmtree(out_dir, ignore_errors=True)
                    continue
                export_texture = self.mesh_has_texture(mesh)
                ok_norm, norm_reason, norm_mesh, norm_metrics = self._normalize_loaded_mesh(mesh)
                if not ok_norm or norm_mesh is None:
                    report.failed_items.append(
                        f"{mesh_path.name}: normalize failed ({norm_reason}); metrics={norm_metrics}"
                    )
                    shutil.rmtree(out_dir, ignore_errors=True)
                    continue
                self.export_trimesh_obj_assets(
                    norm_mesh,
                    out_dir,
                    export_texture=export_texture,
                )
                ok_post, post_reason = self.validate_normalized_obj_export(
                    out_obj,
                    target_max_extent=1.0,
                    metrics=norm_metrics,
                )
                if not ok_post:
                    report.failed_items.append(
                        f"{mesh_path.name}: normalized export invalid ({post_reason}); metrics={norm_metrics}"
                    )
                    shutil.rmtree(out_dir, ignore_errors=True)
                    continue
                report.organized_objects += 1
            except Exception as exc:
                report.failed_items.append(f"{mesh_path.name}: {exc}")

        report.notes.append(
            "Objaverse normalize policy: center=AABB center to origin, scale=max_extent to 1.0; "
            "drop if pre_max_extent<=1e-9 or pre_aspect_ratio>1e5 or post_max_extent not in [0.95,1.05] or post_center_norm>5e-3"
        )
        self.write_manifest_for_organize(cfg, report)
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage="https://objaverse.allenai.org/",
            download_method="objaverse_api",
            notes="Downloaded via objaverse.load_objects and mirrored under assets/objects/raw/Objaverse/objects",
            default_mass_kg=self.normalized_default_mass_kg,
        )
