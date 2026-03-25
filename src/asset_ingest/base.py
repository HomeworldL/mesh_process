"""Base abstraction for object-source ingest (download, organize, manifest)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from urllib.request import urlopen

import trimesh
from trimesh.exchange.obj import export_obj

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
CANONICAL_TEXTURE_NAME = "textured.png"


@dataclass
class IngestConfig:
    repo_root: Path
    source_name: str
    raw_root: Path
    processed_root: Path
    force: bool = False
    workers: int = 8
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

    def _detect_texture_policy(self, has_texture_values: list[str]) -> str:
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

    def append_texture_stats_for_manifest(
        self,
        manifest: IngestManifest,
        report: OrganizeReport,
    ) -> None:
        """
        EN: Append textured/untextured object counts from the final manifest.
        ZH: 基于最终 manifest 追加有纹理/无纹理对象统计。

        Used by / 用于: `write_manifest_for_organize`.
        """
        textured_count = sum(1 for obj in manifest.objects if obj.has_texture == "true")
        untextured_count = sum(1 for obj in manifest.objects if obj.has_texture == "false")
        report.notes.append(
            f"texture stats: textured={textured_count}, untextured={untextured_count}"
        )

    def _default_mtl_path(self, obj_dir: Path) -> Path | None:
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

    def _parse_obj_vertex_xyz(self, line: str) -> tuple[float, float, float] | None:
        """
        EN: Parse one OBJ `v` record into xyz floats.
        ZH: 将一行 OBJ `v` 记录解析为 xyz 浮点坐标。

        Used by / 用于: normalized OBJ export and bbox helpers.
        """
        toks = line.lstrip().split()
        if not toks or toks[0] != "v" or len(toks) < 4:
            return None
        try:
            xyz = (float(toks[1]), float(toks[2]), float(toks[3]))
        except Exception:
            return None
        if not all(math.isfinite(v) for v in xyz):
            return None
        return xyz

    def load_obj_mesh(
        self,
        src_obj_path: Path,
        *,
        remove_unreferenced_vertices: bool = True,
    ) -> tuple[bool, str, Any | None]:
        """
        EN: Load one OBJ into a trimesh mesh object for canonical stage-1 export.
        ZH: 用 trimesh 加载一个 OBJ，为一阶段统一导出做准备。

        Used by / 用于: canonical organize export for DGN, YCB, and future adapters.
        """
        try:
            mesh = trimesh.load(str(src_obj_path))
        except Exception as e:
            return False, f"failed to load obj: {e}", None

        if isinstance(mesh, trimesh.Scene):
            try:
                mesh = mesh.to_geometry()
            except Exception as e:
                return False, f"failed to bake scene geometry: {e}", None

            if isinstance(mesh, trimesh.Scene):
                geoms = list(getattr(mesh, "geometry", {}).values())
                if not geoms:
                    return False, "loaded object is an empty trimesh scene", None
                if len(geoms) == 1:
                    mesh = geoms[0]
                else:
                    try:
                        mesh = trimesh.util.concatenate(geoms)
                    except Exception as e:
                        return False, f"failed to merge baked scene geometry: {e}", None

        if not isinstance(mesh, trimesh.Trimesh):
            return False, "loaded object is not a trimesh mesh", None
        if mesh.vertices is None or len(mesh.vertices) == 0:
            return False, "loaded object has no valid mesh vertices", None
        if mesh.faces is None or len(mesh.faces) == 0:
            return False, "loaded object has no valid mesh faces", None

        try:
            mesh = mesh.copy()
            if remove_unreferenced_vertices:
                mesh.remove_unreferenced_vertices()
        except Exception as e:
            return False, f"failed to clean mesh: {e}", None

        if mesh.vertices is None or len(mesh.vertices) == 0:
            return False, "mesh has no vertices after cleanup", None
        if mesh.faces is None or len(mesh.faces) == 0:
            return False, "mesh has no faces after cleanup", None
        return True, "loaded", mesh

    def mesh_has_texture(self, mesh: Any) -> bool:
        """
        EN: Best-effort check whether a trimesh mesh carries texture-based visuals.
        ZH: 尽力判断 trimesh mesh 是否带有基于纹理贴图的 visual 信息。

        Used by / 用于: canonical organize export for textured datasets such as YCB.
        """
        visual = getattr(mesh, "visual", None)
        if visual is None:
            return False
        uv = getattr(visual, "uv", None)
        material = getattr(visual, "material", None)
        image = None
        if material is not None:
            image = (
                getattr(material, "image", None)
                or getattr(material, "baseColorTexture", None)
            )
        return uv is not None and image is not None

    def export_trimesh_obj_assets(
        self,
        mesh: Any,
        dst_dir: Path,
        *,
        export_texture: bool,
        obj_name: str = CANONICAL_RAW_OBJ_NAME,
    ) -> tuple[Path, Path | None, Path | None]:
        """
        EN: Export one trimesh mesh into unified stage-1 assets:
        `raw.obj` and optional `textured.mtl` + `textured.png`.
        ZH: 将 trimesh mesh 统一导出为一阶段资产：
        `raw.obj`，以及可选的 `textured.mtl` + `textured.png`。

        Used by / 用于: DGN, YCB, and future adapters that standardize organize output
        via trimesh export.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_obj_path = dst_dir / obj_name

        for p in sorted(dst_dir.glob("*.mtl")):
            p.unlink(missing_ok=True)
        for p in sorted(dst_dir.glob("*.png")):
            p.unlink(missing_ok=True)
        for p in sorted(dst_dir.glob("*.jpg")):
            p.unlink(missing_ok=True)
        for p in sorted(dst_dir.glob("*.jpeg")):
            p.unlink(missing_ok=True)

        try:
            obj_text, sidecars = export_obj(
                mesh,
                include_normals=False,
                include_texture=bool(export_texture),
                return_texture=True,
                write_texture=False,
                mtl_name=CANONICAL_MTL_NAME,
                digits=9,
            )
            dst_obj_path.write_text(obj_text, encoding="utf-8")

            dst_tex_path: Path | None = None
            dst_mtl_path: Path | None = None
            mtl_bytes = sidecars.get(CANONICAL_MTL_NAME)
            texture_keys = [
                name
                for name in sorted(sidecars.keys())
                if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]

            if export_texture and texture_keys:
                src_tex_name = texture_keys[0]
                texture_payload = sidecars[src_tex_name]
                dst_tex_path = dst_dir / CANONICAL_TEXTURE_NAME
                if isinstance(texture_payload, bytes):
                    dst_tex_path.write_bytes(texture_payload)
                else:
                    dst_tex_path.write_text(str(texture_payload), encoding="utf-8")

                if mtl_bytes is not None:
                    mtl_text = (
                        mtl_bytes.decode("utf-8", errors="ignore")
                        if isinstance(mtl_bytes, bytes)
                        else str(mtl_bytes)
                    )
                    mtl_text = mtl_text.replace(src_tex_name, CANONICAL_TEXTURE_NAME)
                    dst_mtl_path = dst_dir / CANONICAL_MTL_NAME
                    dst_mtl_path.write_text(mtl_text, encoding="utf-8")
            elif export_texture and mtl_bytes is not None:
                dst_mtl_path = dst_dir / CANONICAL_MTL_NAME
                if isinstance(mtl_bytes, bytes):
                    dst_mtl_path.write_bytes(mtl_bytes)
                else:
                    dst_mtl_path.write_text(str(mtl_bytes), encoding="utf-8")
            return dst_obj_path, dst_tex_path, dst_mtl_path
        except Exception as e:
            raise RuntimeError(f"failed to export trimesh obj assets to {dst_dir}: {e}") from e

    def load_normalized_obj_mesh(
        self,
        src_obj_path: Path,
        *,
        target_max_extent: float = 1.0,
        min_extent_eps: float = 1e-9,
        max_aspect_ratio: float = 1e5,
    ) -> tuple[bool, str, Any | None, dict[str, float]]:
        """
        EN: Load and normalize OBJ geometry with the ShapeNet policy, returning a mesh.
        ZH: 按 ShapeNet 规则加载并归一化 OBJ，返回归一化后的 mesh 对象。

        Used by / 用于: DGN organize and normalize_obj_center_and_scale wrapper.
        """
        metrics: dict[str, float] = {}
        ok, msg, mesh = self.load_obj_mesh(
            src_obj_path,
            remove_unreferenced_vertices=True,
        )
        if not ok or mesh is None:
            return False, msg, None, metrics

        pre_vertex_count = int(len(mesh.vertices))
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
        metrics["dropped_unreferenced_vertices"] = float(pre_vertex_count - len(mesh.vertices))

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
        except Exception as e:
            return False, f"failed to normalize mesh transform: {e}", None, metrics
        return True, "normalized", mesh, metrics

    def normalize_obj_center_and_scale(
        self,
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
        ok, msg, mesh, metrics = self.load_normalized_obj_mesh(
            src_obj_path,
            target_max_extent=target_max_extent,
            min_extent_eps=min_extent_eps,
            max_aspect_ratio=max_aspect_ratio,
        )
        if not ok or mesh is None:
            return False, msg, metrics

        dst_obj_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.export_trimesh_obj_assets(
                mesh,
                dst_obj_path.parent,
                export_texture=False,
                obj_name=dst_obj_path.name,
            )
        except Exception as e:
            return False, f"failed to export normalized obj: {e}", metrics

        ok_post, post_reason = self.validate_normalized_obj_export(
            dst_obj_path,
            target_max_extent=target_max_extent,
            metrics=metrics,
        )
        if not ok_post:
            return False, post_reason, metrics

        return True, "normalized", metrics

    def validate_normalized_obj_export(
        self,
        obj_path: Path,
        *,
        target_max_extent: float = 1.0,
        metrics: dict[str, float] | None = None,
    ) -> tuple[bool, str]:
        """
        EN: Validate the exported normalized OBJ against size/center expectations.
        ZH: 校验归一化导出的 OBJ 是否满足尺寸与中心约束。

        Used by / 用于: normalized organize export for DGN and wrapper normalization helper.
        """
        if metrics is None:
            metrics = {}

        post_metrics = self._read_obj_bbox_metrics(obj_path)
        if post_metrics is None:
            return False, "normalized obj has invalid bbox metrics"

        post_max_extent = float(post_metrics["max_extent"])
        post_center = [
            (post_metrics["min_x"] + post_metrics["max_x"]) * 0.5,
            (post_metrics["min_y"] + post_metrics["max_y"]) * 0.5,
            (post_metrics["min_z"] + post_metrics["max_z"]) * 0.5,
        ]
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
            return False, (
                "normalized size out of expected range: "
                f"post_max_extent={post_max_extent:.6f}, target={target_max_extent:.6f}"
            )
        if post_center_norm > 5e-3:
            return False, (
                "normalized center drift too large: "
                f"center_norm={post_center_norm:.6e}"
            )
        return True, "normalized"

    def _read_obj_bbox_metrics(self, obj_path: Path) -> dict[str, float] | None:
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
                    xyz = self._parse_obj_vertex_xyz(line)
                    if xyz is None:
                        continue
                    for axis, value in enumerate(xyz):
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
            "min_x": float(min_xyz[0]),
            "min_y": float(min_xyz[1]),
            "min_z": float(min_xyz[2]),
            "max_x": float(max_xyz[0]),
            "max_y": float(max_xyz[1]),
            "max_z": float(max_xyz[2]),
            "extent_x": float(extents[0]),
            "extent_y": float(extents[1]),
            "extent_z": float(extents[2]),
            "max_extent": float(max(extents)),
            "volume": float(extents[0] * extents[1] * extents[2]),
        }

    def _format_bbox_record(self, note_prefix: str, record: dict[str, Any]) -> str:
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

        Used by / 用于: all adapters that call `write_manifest_for_organize`.
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

    def default_manifest_path(self, cfg: IngestConfig) -> Path:
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
        self.append_texture_stats_for_manifest(manifest, report)

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
