"""Base abstraction for object-source ingest (download, organize, manifest)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from urllib.request import urlopen

from .manifest import (
    IngestManifest,
    ManifestSource,
    ManifestSummary,
    ObjectRecord,
    validate_manifest_dict,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None


DEFAULT_MASS_KG = 0.1
CANONICAL_RAW_OBJ_NAME = "raw.obj"
CANONICAL_MTL_NAME = "textured.mtl"
CANONICAL_TEXTURE_NAME = "texture_map.png"


@dataclass
class IngestConfig:
    repo_root: Path
    source_name: str
    download_root: Path
    raw_root: Path
    processed_root: Path
    force: bool = False
    workers: int = 8
    subset: str | None = None
    sample_n: int | None = None
    sample_seed: int = 0
    credentials: dict[str, Any] = field(default_factory=dict)

    @property
    def source_download_dir(self) -> Path:
        # Download artifacts are stored directly in raw/<source>.
        return self.source_raw_dir

    @property
    def source_raw_dir(self) -> Path:
        return self.raw_root / self.source_name

    @property
    def source_processed_dir(self) -> Path:
        return self.processed_root / self.source_name


@dataclass
class DownloadReport:
    source: str
    downloaded_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class OrganizeReport:
    source: str
    organized_objects: int = 0
    failed_items: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    manifest_errors: list[str] = field(default_factory=list)


class BaseIngestAdapter(ABC):
    """Common interface for object source adapters."""

    source_name: str = ""
    version: str | None = None

    @abstractmethod
    def download(self, cfg: IngestConfig) -> DownloadReport:
        """
        EN: Download raw source artifacts into `assets/objects/raw/<source>`.
        ZH: 下载原始数据到 `assets/objects/raw/<source>`。

        Used by / 用于: all adapters.
        """
        raise NotImplementedError

    @abstractmethod
    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        """
        EN: Convert downloaded source files into canonical stage-1 outputs.
        ZH: 将下载结果整理为一阶段统一导出格式。

        Used by / 用于: all adapters.
        """
        raise NotImplementedError

    @abstractmethod
    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        """
        EN: Build the manifest for the adapter's primary processed dataset.
        ZH: 为当前 adapter 的主数据集构建 manifest。

        Used by / 用于: all adapters.
        """
        raise NotImplementedError

    def validate_manifest(
        self,
        manifest: IngestManifest,
        cfg: IngestConfig,
        check_paths: bool = True,
    ) -> list[str]:
        """
        EN: Validate manifest schema and optionally verify referenced paths.
        ZH: 校验 manifest 结构，并按需检查路径是否存在。

        Used by / 用于: all adapters via `verify` and `write_manifest_for_organize`.
        """
        return validate_manifest_dict(
            manifest.to_dict(),
            repo_root=cfg.repo_root,
            check_paths=check_paths,
        )

    @staticmethod
    def _detect_texture_policy(has_texture_values: list[str]) -> str:
        """
        EN: Collapse per-object `has_texture` flags into summary policy.
        ZH: 将逐物体 `has_texture` 汇总成 manifest summary 里的策略字段。

        Used by / 用于: shared manifest builder in base.py.
        """
        texture_states = set(has_texture_values)
        if texture_states == {"true"}:
            return "all"
        if texture_states == {"false"}:
            return "none"
        if not texture_states:
            return "unknown"
        return "mixed"

    @staticmethod
    def _default_mtl_path(obj_dir: Path) -> Path | None:
        """
        EN: Prefer canonical `textured.mtl`, otherwise fall back to the first local `.mtl`.
        ZH: 优先使用规范名 `textured.mtl`，否则回退到目录中的第一个 `.mtl`。

        Used by / 用于: shared manifest builder in base.py.
        """
        canonical = obj_dir / CANONICAL_MTL_NAME
        if canonical.exists():
            return canonical
        return next((p for p in sorted(obj_dir.glob("*.mtl")) if p.is_file()), None)

    def build_manifest_from_processed_dir(
        self,
        cfg: IngestConfig,
        *,
        homepage: str,
        download_method: str,
        notes: str,
        default_mass_kg: float,
        dataset_name: str | None = None,
        processed_dir: Path | None = None,
        name_resolver: Callable[[str, Path], str] | None = None,
        category_resolver: Callable[[str, Path], str | None] | None = None,
        mass_resolver: Callable[[str, Path], float] | None = None,
        mtl_resolver: Callable[[Path], Path | None] | None = None,
        missing_has_texture_policy: str = "unknown",
    ) -> IngestManifest:
        """
        EN: Shared manifest builder for adapters whose organized layout is
        `processed/<dataset>/<object_id>/raw.obj` plus optional texture sidecars.
        ZH: 适用于这类 adapter 的通用 manifest 构建器：
        `processed/<dataset>/<object_id>/raw.obj`，并可带可选纹理文件。

        Used by / 用于: YCB, RealDex, KIT, DexNet, GraspNet, HOPE, MSO, Objaverse,
        ShapeNetCore, ShapeNetSem.
        """
        manifest_dataset = dataset_name or self.source_name
        target_dir = processed_dir if processed_dir is not None else cfg.source_processed_dir
        mtl_picker = mtl_resolver or self._default_mtl_path

        manifest = IngestManifest.create(dataset=manifest_dataset, version=self.version)
        manifest.source = ManifestSource(
            homepage=homepage,
            download_method=download_method,
            notes=notes,
        )

        if not target_dir.exists():
            manifest.summary = ManifestSummary(
                num_objects=0,
                num_categories=0,
                has_texture_policy=missing_has_texture_policy,
                default_mass_kg=default_mass_kg,
            )
            return manifest

        objects: list[ObjectRecord] = []
        categories: set[str] = set()
        has_texture_values: list[str] = []

        for obj_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
            object_id = obj_dir.name
            mesh_path = obj_dir / CANONICAL_RAW_OBJ_NAME
            if not mesh_path.exists():
                continue

            category = category_resolver(object_id, obj_dir) if category_resolver is not None else None
            if category is not None:
                categories.add(category)

            texture_files = [p.name for p in sorted(obj_dir.glob("*.png"))]
            has_texture = "true" if texture_files else "false"
            has_texture_values.append(has_texture)

            mtl_path = mtl_picker(obj_dir)
            mass_kg = (
                mass_resolver(object_id, obj_dir)
                if mass_resolver is not None
                else default_mass_kg
            )
            object_name = (
                name_resolver(object_id, obj_dir)
                if name_resolver is not None
                else object_id
            )
            objects.append(
                ObjectRecord(
                    object_id=object_id,
                    name=object_name,
                    category=category,
                    mesh_path=relative_to_repo(cfg.repo_root, mesh_path),
                    mesh_format="obj",
                    mass_kg=mass_kg,
                    has_texture=has_texture,
                    mtl_path=(relative_to_repo(cfg.repo_root, mtl_path) if mtl_path is not None else None),
                    texture_files=texture_files,
                )
            )

        manifest.objects = objects
        manifest.summary = ManifestSummary(
            num_objects=len(objects),
            num_categories=len(categories),
            has_texture_policy=self._detect_texture_policy(has_texture_values),
            default_mass_kg=default_mass_kg,
        )
        return manifest

    @staticmethod
    def normalize_obj_center_and_scale(
        src_obj_path: Path,
        dst_obj_path: Path,
        *,
        target_max_extent: float = 1.0,
        min_extent_eps: float = 1e-9,
        max_aspect_ratio: float = 1e5,
    ) -> tuple[bool, str, dict[str, float]]:
        """
        EN: Normalize OBJ geometry with the ShapeNet policy:
        1) translate AABB center to origin
        2) uniformly scale so max AABB extent == target_max_extent
        ZH: 按 ShapeNet 规则归一化 OBJ：
        1) 将 AABB 中心平移到原点
        2) 等比例缩放到最大包围盒边长为 `target_max_extent`

        Used by / 用于: ShapeNetCore, ShapeNetSem, DGN, DDG.
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
            stripped = line.lstrip()
            toks = stripped.split()
            if not toks or toks[0] != "v" or len(toks) < 4:
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

        dst_obj_path.parent.mkdir(parents=True, exist_ok=True)
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
        post_center_norm = float(
            math.sqrt(
                float(post_center[0]) ** 2
                + float(post_center[1]) ** 2
                + float(post_center[2]) ** 2
            )
        )
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
    def _read_obj_bbox_metrics(obj_path: Path) -> dict[str, float] | None:
        """
        EN: Read per-axis bbox size, max extent, and bbox volume from OBJ vertices.
        ZH: 从 OBJ 顶点读取包围盒三轴尺寸、最大边长和包围盒体积。

        Used by / 用于: bbox summary helper in base.py.
        """
        min_xyz = [math.inf, math.inf, math.inf]
        max_xyz = [-math.inf, -math.inf, -math.inf]
        vertex_count = 0

        try:
            with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.lstrip()
                    toks = stripped.split()
                    if not toks or toks[0] != "v" or len(toks) < 4:
                        continue
                    try:
                        vals = [float(toks[1]), float(toks[2]), float(toks[3])]
                    except Exception:
                        continue
                    if not all(math.isfinite(v) for v in vals):
                        continue
                    for axis, value in enumerate(vals):
                        min_xyz[axis] = min(min_xyz[axis], value)
                        max_xyz[axis] = max(max_xyz[axis], value)
                    vertex_count += 1
        except Exception:
            return None

        if vertex_count == 0:
            return None

        extents = [max_xyz[i] - min_xyz[i] for i in range(3)]
        if not all(math.isfinite(v) and v >= 0.0 for v in extents):
            return None

        return {
            "extent_x": float(extents[0]),
            "extent_y": float(extents[1]),
            "extent_z": float(extents[2]),
            "max_extent": float(max(extents)),
            "volume": float(extents[0] * extents[1] * extents[2]),
        }

    @staticmethod
    def _format_bbox_record(note_prefix: str, record: dict[str, Any]) -> str:
        """
        EN: Format one bbox summary line for organize report notes.
        ZH: 将单条 bbox 极值统计格式化为 organize report 的说明文本。

        Used by / 用于: bbox summary helper in base.py.
        """
        return (
            f"{note_prefix}: object={record['object_id']}, "
            f"size=[{record['extent_x']:.6g}, {record['extent_y']:.6g}, {record['extent_z']:.6g}], "
            f"max_extent={record['max_extent']:.6g}, volume={record['volume']:.6g}"
        )

    def append_bbox_stats_for_dataset(
        self,
        cfg: IngestConfig,
        report: OrganizeReport,
        *,
        dataset_dir: Path | None = None,
        dataset_name: str | None = None,
    ) -> None:
        """
        EN: Append dataset-level bbox min/max summaries after organize.
        ZH: 在 organize 结束后追加数据集级别的 bbox 极值统计。

        Used by / 用于: all adapters that call `write_manifest_for_organize`;
        also manually called by DGNAdapter for derived DDG.
        """
        target_dir = dataset_dir if dataset_dir is not None else cfg.source_processed_dir
        label = dataset_name or target_dir.name or cfg.source_name
        if not target_dir.exists():
            report.notes.append(f"{label} bbox stats: missing processed dir {target_dir}")
            return

        raw_meshes: list[tuple[str, Path]] = []
        for obj_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
            raw_obj = obj_dir / CANONICAL_RAW_OBJ_NAME
            if raw_obj.is_file():
                raw_meshes.append((obj_dir.name, raw_obj))

        if not raw_meshes:
            report.notes.append(f"{label} bbox stats: no organized {CANONICAL_RAW_OBJ_NAME} found")
            return

        min_extent_rec: dict[str, Any] | None = None
        max_extent_rec: dict[str, Any] | None = None
        min_volume_rec: dict[str, Any] | None = None
        max_volume_rec: dict[str, Any] | None = None
        invalid_count = 0

        for object_id, raw_obj in raw_meshes:
            metrics = self._read_obj_bbox_metrics(raw_obj)
            if metrics is None:
                invalid_count += 1
                continue

            record: dict[str, Any] = {"object_id": object_id, **metrics}
            if min_extent_rec is None or record["max_extent"] < min_extent_rec["max_extent"]:
                min_extent_rec = record
            if max_extent_rec is None or record["max_extent"] > max_extent_rec["max_extent"]:
                max_extent_rec = record
            if min_volume_rec is None or record["volume"] < min_volume_rec["volume"]:
                min_volume_rec = record
            if max_volume_rec is None or record["volume"] > max_volume_rec["volume"]:
                max_volume_rec = record

        valid_count = len(raw_meshes) - invalid_count
        report.notes.append(
            f"{label} bbox stats: scanned={len(raw_meshes)}, valid={valid_count}, invalid={invalid_count}"
        )
        if min_extent_rec is not None:
            report.notes.append(
                self._format_bbox_record(f"{label} bbox min max_extent", min_extent_rec)
            )
        if max_extent_rec is not None:
            report.notes.append(
                self._format_bbox_record(f"{label} bbox max max_extent", max_extent_rec)
            )
        if min_volume_rec is not None:
            report.notes.append(
                self._format_bbox_record(f"{label} bbox min volume", min_volume_rec)
            )
        if max_volume_rec is not None:
            report.notes.append(
                self._format_bbox_record(f"{label} bbox max volume", max_volume_rec)
            )

    @staticmethod
    def default_manifest_path(cfg: IngestConfig) -> Path:
        """
        EN: Return the default manifest output path for the adapter's primary dataset.
        ZH: 返回当前 adapter 主数据集的默认 manifest 路径。

        Used by / 用于: `write_manifest_for_organize`.
        """
        return cfg.source_processed_dir / "manifest.json"

    def write_manifest_for_organize(self, cfg: IngestConfig, report: OrganizeReport) -> None:
        """
        EN: Persist manifest, validate it, and append standard organize-time stats.
        ZH: 写出 manifest、执行校验，并追加标准化的一阶段统计信息。

        Used by / 用于: almost all adapters as the final step of `organize`.
        """
        manifest = self.build_manifest(cfg)
        out_path = self.default_manifest_path(cfg)
        manifest.save(out_path)
        report.manifest_path = relative_to_repo(cfg.repo_root, out_path)
        errs = self.validate_manifest(manifest, cfg, check_paths=True)
        report.manifest_errors = errs
        if errs:
            report.notes.append(f"manifest validation failed: {len(errs)} issues")
        else:
            report.notes.append("manifest validation passed")
        self.append_bbox_stats_for_dataset(cfg, report)

    @staticmethod
    def parse_common_args() -> argparse.ArgumentParser:
        """
        EN: Define common CLI arguments shared by all ingest adapters.
        ZH: 定义所有 ingest adapter 共用的命令行参数。

        Used by / 用于: `src/ingest_assets.py`.
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--workers", type=int, default=8)
        parser.add_argument("--subset", type=str, default=None)
        parser.add_argument(
            "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
        )
        parser.add_argument("--download-root", type=Path, default=None)
        parser.add_argument("--raw-root", type=Path, default=None)
        parser.add_argument("--processed-root", type=Path, default=None)
        return parser

    @classmethod
    def config_from_args(cls, args: argparse.Namespace, source_name: str) -> IngestConfig:
        """
        EN: Convert parsed CLI args into the normalized ingest config object.
        ZH: 将命令行参数整理成统一的 `IngestConfig` 配置对象。

        Used by / 用于: `src/ingest_assets.py` for all adapters.
        """
        repo_root = Path(args.repo_root).resolve()
        raw_root = (
            Path(args.raw_root).resolve()
            if args.raw_root is not None
            else repo_root / "assets" / "objects" / "raw"
        )
        # Keep argument for compatibility, but raw is the effective download root now.
        download_root = (
            Path(args.download_root).resolve()
            if args.download_root is not None
            else raw_root
        )
        processed_root = (
            Path(args.processed_root).resolve()
            if args.processed_root is not None
            else repo_root / "assets" / "objects" / "processed"
        )
        return IngestConfig(
            repo_root=repo_root,
            source_name=source_name,
            download_root=download_root,
            raw_root=raw_root,
            processed_root=processed_root,
            force=bool(args.force),
            workers=int(args.workers),
            subset=args.subset,
            sample_n=getattr(args, "sample_n", None),
            sample_seed=int(getattr(args, "sample_seed", 0)),
        )


def sanitize_object_id(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "object"


def relative_to_repo(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve())).replace(os.sep, "/")


def download_google_drive_file(file_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import gdown  # type: ignore

        gdown.download(id=file_id, output=str(output_path), quiet=False)
        return
    except Exception:
        pass

    gdown_bin = shutil.which("gdown")
    if gdown_bin:
        cmd = [gdown_bin, f"https://drive.google.com/uc?id={file_id}", "-O", str(output_path)]
        subprocess.run(cmd, check=True)
        return

    raise RuntimeError(
        "Google Drive download requires 'gdown' (python package or CLI). "
        "Install with: pip install gdown"
    )


def download_url_file(url: str, output_path: Path, desc: str | None = None) -> None:
    """Download a URL with streaming + tqdm progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urlopen(url)
    total = req.length if req.length is not None else None

    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc or output_path.name,
            leave=True,
        )
    else:
        print(
            f"[download] {output_path.name} (tqdm not installed, progress disabled)",
            file=sys.stderr,
        )

    with open(output_path, "wb") as f:
        while True:
            buf = req.read(1024 * 1024)
            if not buf:
                break
            f.write(buf)
            if pbar is not None:
                pbar.update(len(buf))
    if pbar is not None:
        pbar.close()


def copy_file_with_progress(src_path: Path, dst_path: Path, desc: str | None = None) -> None:
    """Copy large file with tqdm progress bar."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    total = src_path.stat().st_size
    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc or f"copy {src_path.name}",
            leave=True,
        )
    with open(src_path, "rb") as fsrc, open(dst_path, "wb") as fdst:
        while True:
            chunk = fsrc.read(1024 * 1024)
            if not chunk:
                break
            fdst.write(chunk)
            if pbar is not None:
                pbar.update(len(chunk))
    if pbar is not None:
        pbar.close()


def extract_zip(
    zip_path: Path,
    output_dir: Path,
    force: bool = False,
    show_progress: bool = True,
    skip_member: Callable[[str], bool] | None = None,
) -> Path:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        if skip_member is not None:
            members = [m for m in members if not skip_member(m.filename)]
        bar = None
        if show_progress and tqdm is not None:
            bar = tqdm(members, desc=f"extract {zip_path.name}", unit="file")
        iterable = bar if bar is not None else members
        for member in iterable:
            zf.extract(member, output_dir)
        if bar is not None:
            bar.close()
    return output_dir


def _read_text_lossy(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def detect_obj_material_name(obj_path: Path) -> str | None:
    """Return first 'usemtl' material name in OBJ, if present."""
    if not obj_path.exists():
        return None
    for line in _read_text_lossy(obj_path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("usemtl "):
            parts = stripped.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
    return None


def rewrite_obj_material_refs(
    obj_path: Path,
    mtl_name: str = CANONICAL_MTL_NAME,
    ensure_usemtl: str | None = None,
) -> None:
    """Rewrite OBJ to reference canonical MTL file, and optionally ensure one usemtl."""
    if not obj_path.exists():
        return
    lines = _read_text_lossy(obj_path).splitlines()
    out: list[str] = []
    saw_mtllib = False
    saw_usemtl = False

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("mtllib "):
            if not saw_mtllib:
                out.append(f"mtllib {mtl_name}")
                saw_mtllib = True
            continue
        if low.startswith("usemtl "):
            saw_usemtl = True
        out.append(line)

    if not saw_mtllib:
        out.insert(0, f"mtllib {mtl_name}")

    if ensure_usemtl and not saw_usemtl:
        insert_at = 1 if out and out[0].lower().startswith("mtllib ") else 0
        out.insert(insert_at, f"usemtl {ensure_usemtl}")

    _write_text(obj_path, "\n".join(out) + "\n")


def write_canonical_mtl(
    dst_mtl_path: Path,
    material_name: str,
    texture_name: str = CANONICAL_TEXTURE_NAME,
) -> None:
    content = (
        f"newmtl {material_name}\n"
        "# shader_type beckmann\n"
        f"map_Kd {texture_name}\n"
    )
    _write_text(dst_mtl_path, content)


def rewrite_mtl_to_canonical_texture(
    src_mtl_path: Path,
    dst_mtl_path: Path,
    material_name: str,
    texture_name: str = CANONICAL_TEXTURE_NAME,
) -> None:
    """Rewrite MTL with canonical material + texture map filename."""
    lines = _read_text_lossy(src_mtl_path).splitlines()
    out: list[str] = []
    saw_newmtl = False
    saw_map_kd = False

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("newmtl "):
            if not saw_newmtl:
                out.append(f"newmtl {material_name}")
                saw_newmtl = True
            continue
        if low.startswith("map_kd "):
            out.append(f"map_Kd {texture_name}")
            saw_map_kd = True
            continue
        out.append(line)

    if not saw_newmtl:
        out.insert(0, f"newmtl {material_name}")
    if not saw_map_kd:
        out.append(f"map_Kd {texture_name}")

    _write_text(dst_mtl_path, "\n".join(out) + "\n")


def canonicalize_texture_assets(
    dst_dir: Path,
    raw_obj_path: Path,
    texture_src: Path | None = None,
    mtl_src: Path | None = None,
    create_mtl_if_texture: bool = False,
    default_material_name: str = "material_0",
) -> tuple[Path | None, Path | None]:
    """Normalize optional texture assets to texture_map.png and textured.mtl."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_tex_path: Path | None = None
    dst_mtl_path: Path | None = None

    # Resolve source texture: explicit > existing texture_map.png > first png in dst dir.
    if texture_src is None:
        maybe_tex = dst_dir / CANONICAL_TEXTURE_NAME
        if maybe_tex.exists():
            texture_src = maybe_tex
        else:
            pngs = sorted(p for p in dst_dir.glob("*.png") if p.name != CANONICAL_TEXTURE_NAME)
            if pngs:
                texture_src = pngs[0]
    if texture_src is not None and texture_src.exists():
        dst_tex_path = dst_dir / CANONICAL_TEXTURE_NAME
        if texture_src.resolve() != dst_tex_path.resolve():
            shutil.copy2(texture_src, dst_tex_path)

    material_name = detect_obj_material_name(raw_obj_path) or default_material_name

    # Resolve source mtl: explicit > existing canonical mtl > first mtl in dst dir.
    if mtl_src is None:
        maybe_mtl = dst_dir / CANONICAL_MTL_NAME
        if maybe_mtl.exists():
            mtl_src = maybe_mtl
        else:
            mtls = sorted(p for p in dst_dir.glob("*.mtl") if p.name != CANONICAL_MTL_NAME)
            if mtls:
                mtl_src = mtls[0]

    if dst_tex_path is not None:
        dst_mtl_path = dst_dir / CANONICAL_MTL_NAME
        if mtl_src is not None and mtl_src.exists():
            rewrite_mtl_to_canonical_texture(
                src_mtl_path=mtl_src,
                dst_mtl_path=dst_mtl_path,
                material_name=material_name,
                texture_name=CANONICAL_TEXTURE_NAME,
            )
        elif create_mtl_if_texture:
            write_canonical_mtl(
                dst_mtl_path=dst_mtl_path,
                material_name=material_name,
                texture_name=CANONICAL_TEXTURE_NAME,
            )
        else:
            dst_mtl_path = None

    # Clean non-canonical texture/material files in organized output dir.
    for p in sorted(dst_dir.glob("*.png")):
        if p.name != CANONICAL_TEXTURE_NAME:
            p.unlink(missing_ok=True)
    for p in sorted(dst_dir.glob("*.mtl")):
        if p.name != CANONICAL_MTL_NAME:
            p.unlink(missing_ok=True)

    if dst_mtl_path is not None and dst_mtl_path.exists():
        rewrite_obj_material_refs(
            obj_path=raw_obj_path,
            mtl_name=CANONICAL_MTL_NAME,
            ensure_usemtl=material_name,
        )
    return dst_tex_path, dst_mtl_path
