"""DGN adapter: download/extract archive, organize into DGN/DDG processed datasets."""

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
        # EN: One source archive is organized into two processed datasets:
        # the full DGN set and the DDG prefix subset. Reuse the shared builder
        # for both outputs instead of maintaining two manifest code paths.
        # ZH: 一个源压缩包会整理出两个 processed 数据集：完整 DGN 和
        # DDG 前缀子集。这里复用同一套 manifest 构建逻辑，避免维护两份实现。
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage="https://huggingface.co/datasets/JiayiChenPKU/BODex",
            download_method="url_zip",
            notes="Derived from DGN_obj_raw.zip; object split by prefix into DGN / DDG",
            default_mass_kg=self.normalized_default_mass_kg,
            dataset_name=dataset_name,
            processed_dir=cfg.processed_root / dataset_name,
            missing_has_texture_policy="none",
            mtl_resolver=lambda _obj_dir: None,
        )

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
            dgn_ready = dgn_dst_obj.exists() and not cfg.force
            if not dgn_dst_obj.exists() or cfg.force:
                # EN: Normalize first, then export through the shared trimesh
                # stage-1 writer. This keeps DGN's geometry policy separate from
                # the actual asset export policy.
                # ZH: 先做归一化，再走共享的一阶段 trimesh 导出逻辑。
                # 这样 DGN 的几何归一化策略和最终资产导出策略是分离的。
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
                dgn_ready = True

            # EN: DDG is a derived subset view of DGN: any source object whose
            # name starts with `ddg` is copied into the secondary DDG dataset.
            # ZH: DDG 是 DGN 的派生子集视图：凡是源文件名以 `ddg` 开头的对象，
            # 都会再复制到第二个 DDG 数据集中。
            if raw_name.lower().startswith("ddg") and dgn_ready:
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
                    shutil.copy2(dgn_dst_obj, ddg_dst_obj)
                    ddg_count += 1

        # EN: Keep ingest CLI compatibility by writing the primary manifest at
        # processed/DGN/manifest.json, then emit the derived DDG manifest too.
        # ZH: 为保持 ingest CLI 兼容性，主 manifest 仍写到
        # processed/DGN/manifest.json，同时额外写出派生的 DDG manifest。
        print("[DGN] Writing DGN manifest and scanning DGN bbox stats...")
        self.write_manifest_for_organize(cfg, report)
        # EN: Also emit the DDG manifest so stage-2/3 can treat DDG like a
        # normal dataset root.
        # ZH: 同时写出 DDG manifest，让二三阶段也能把 DDG 当成普通数据集处理。
        print("[DGN] Writing DDG manifest...")
        ddg_manifest = self._build_manifest_for_dataset(cfg, "DDG")
        ddg_manifest_path = cfg.processed_root / "DDG" / "manifest.json"
        ddg_manifest.save(ddg_manifest_path)

        report.notes.append(f"organized from: {relative_to_repo(cfg.repo_root, src_root)}")
        report.notes.append(
            "DGN/DDG normalize policy: center=AABB center to origin, scale=max_extent to 1.0; "
            "drop if pre_max_extent<=1e-9 or pre_aspect_ratio>1e5 or post_max_extent not in [0.95,1.05] or post_center_norm>5e-3"
        )
        report.notes.append(
            f"DGN/DDG normalize stats: normalized={normalized_count}, dropped_bad_scale={dropped_bad_scale}"
        )
        report.notes.append(f"DGN objects (full set): {dgn_count}")
        report.notes.append(f"DDG objects (subset, prefix ddg*): {ddg_count}")
        report.notes.append(f"DDG manifest: {relative_to_repo(cfg.repo_root, ddg_manifest_path)}")
        print("[DGN] Scanning DDG bbox stats...")
        self.append_bbox_stats_for_dataset(
            cfg,
            report,
            dataset_dir=ddg_processed_dir,
            dataset_name="DDG",
        )
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        # By design this adapter's source manifest points to processed/DGN.
        return self._build_manifest_for_dataset(cfg, "DGN")
