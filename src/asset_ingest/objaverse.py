"""Objaverse source adapter with chunked download and normalized organization."""

from __future__ import annotations

from pathlib import Path
import json
import os
import random
import shutil
import re

from .base import (
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
from .manifest import IngestManifest


class ObjaverseAdapter(BaseIngestAdapter):
    source_name = "Objaverse"
    version = "v1"
    category_anno_url = (
        "https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/"
        "category_annotation.json"
    )
    uid_pattern = re.compile(r"^[0-9a-fA-F-]{16,}$")

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

    @staticmethod
    def _split_subset_tokens(subset: str | None) -> list[str]:
        if not subset:
            return []
        return [x.strip() for x in subset.split(",") if x.strip()]

    @classmethod
    def _looks_like_uid(cls, value: str) -> bool:
        return bool(cls.uid_pattern.match(value))

    def _subset_uids_from_file(self, subset: str) -> set[str] | None:
        candidate = Path(subset).expanduser()
        if not candidate.exists() or not candidate.is_file():
            return None
        with open(candidate, "r", encoding="utf-8") as f:
            values = {line.strip() for line in f if line.strip()}
        return {v for v in values if self._looks_like_uid(v)}

    def _select_uids(
        self,
        all_uids: set[str],
        cfg: IngestConfig,
        cache_root: Path,
    ) -> tuple[list[str], list[str]]:
        """Resolve subset into UID selection.

        Supported subset modes:
        - None: all uids
        - comma-separated category labels (requires annotation file)
        - comma-separated explicit uids
        - path to a txt file with one uid per line
        """
        notes: list[str] = []
        if not cfg.subset:
            return sorted(all_uids), notes

        from_file = self._subset_uids_from_file(cfg.subset)
        if from_file is not None:
            picked = sorted(from_file.intersection(all_uids))
            notes.append(
                f"subset mode=file, requested={len(from_file)}, matched={len(picked)}"
            )
            return picked, notes

        subset_tokens = self._split_subset_tokens(cfg.subset)
        uid_tokens = {x for x in subset_tokens if self._looks_like_uid(x)}
        category_tokens = {x for x in subset_tokens if x not in uid_tokens}

        picked_uids: set[str] = set()
        if uid_tokens:
            picked_uids.update(uid_tokens.intersection(all_uids))
            notes.append(
                f"subset mode=uid list, requested={len(uid_tokens)}, matched={len(picked_uids)}"
            )

        if category_tokens:
            anno_path = self._ensure_category_annotation(cache_root)
            if anno_path is None:
                notes.append(
                    "subset categories provided but annotation download unavailable; "
                    "only uid tokens were applied"
                )
            else:
                with open(anno_path, "r", encoding="utf-8") as f:
                    anno = json.load(f)
                cat_uids = {
                    item["object_index"].split(".glb")[0]
                    for item in anno
                    if item.get("label") in category_tokens and item.get("object_index")
                }
                matched = cat_uids.intersection(all_uids)
                picked_uids.update(matched)
                notes.append(
                    f"subset mode=category, categories={len(category_tokens)}, "
                    f"matched_uids={len(matched)}"
                )

        if not picked_uids:
            notes.append("subset produced 0 objects")
            return [], notes
        return sorted(picked_uids), notes

    @staticmethod
    def _clean_cache_tmp(cache_root: Path) -> int:
        cleaned = 0
        glb_dir = cache_root / "hf-objaverse-v1" / "glbs"
        if not glb_dir.exists():
            return cleaned
        for tmp_path in glb_dir.rglob("*.tmp"):
            tmp_path.unlink(missing_ok=True)
            cleaned += 1
        return cleaned

    @staticmethod
    def _link_or_copy(src_path: Path, dst_path: Path) -> str:
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
        uids, subset_notes = self._select_uids(all_uids=all_uids, cfg=cfg, cache_root=cache_root)
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

        try:
            import trimesh
        except Exception as exc:
            raise RuntimeError("Objaverse organize requires `trimesh`: pip install trimesh") from exc

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
                if mesh_path.suffix.lower() == ".obj":
                    shutil.copy2(mesh_path, out_obj)
                    mtl_src = next((p for p in sorted(mesh_path.parent.glob("*.mtl")) if p.is_file()), None)
                    tex_src = next((p for p in sorted(mesh_path.parent.glob("*.png")) if p.is_file()), None)
                    canonicalize_texture_assets(
                        dst_dir=out_dir,
                        raw_obj_path=out_obj,
                        texture_src=tex_src,
                        mtl_src=mtl_src,
                        create_mtl_if_texture=False,
                    )
                else:
                    loaded = trimesh.load(mesh_path, process=False, force="scene")
                    if isinstance(loaded, trimesh.Trimesh):
                        loaded.export(out_obj)
                    else:
                        if not loaded.geometry:
                            report.failed_items.append(f"empty scene: {mesh_path.name}")
                            continue
                        # Prefer scene export first (better chance to keep materials/textures).
                        try:
                            loaded.export(out_obj)
                        except Exception:
                            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
                            mesh.export(out_obj)

                    # Normalize optional texture assets generated during GLB/GLTF export.
                    mtl_src = next((p for p in sorted(out_dir.glob("*.mtl")) if p.is_file()), None)
                    tex_src = next((p for p in sorted(out_dir.glob("*.png")) if p.is_file()), None)
                    canonicalize_texture_assets(
                        dst_dir=out_dir,
                        raw_obj_path=out_obj,
                        texture_src=tex_src,
                        mtl_src=mtl_src,
                        create_mtl_if_texture=False,
                    )
                report.organized_objects += 1
            except Exception as exc:
                report.failed_items.append(f"{mesh_path.name}: {exc}")

        self.write_manifest_for_organize(cfg, report)
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage="https://objaverse.allenai.org/",
            download_method="objaverse_api",
            notes="Downloaded via objaverse.load_objects and mirrored under assets/objects/raw/Objaverse/objects",
            default_mass_kg=DEFAULT_MASS_KG,
        )
