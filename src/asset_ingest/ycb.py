"""YCB source adapter: download from official endpoints and normalize to raw format."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import tarfile
from urllib.request import Request, urlopen

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


class YCBAdapter(BaseIngestAdapter):
    source_name = "YCB"
    version = None

    objects_url = "https://ycb-benchmarks.s3.amazonaws.com/data/objects.json"
    base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
    files_to_download = ["berkeley_processed", "google_16k"]

    def _choose_source_mesh_dir(self, obj_dir: Path) -> tuple[Path, bool] | None:
        # EN: Prefer google_16k when present; only that branch is treated as
        # textured export in stage-1. Other branches still contribute geometry,
        # but we intentionally do not export texture sidecars from them.
        # ZH: 优先使用 google_16k；只有这一路在一阶段会导出纹理文件。
        # 其他分支仍可提供几何，但我们有意不从中导出纹理 sidecar。
        google_dir = obj_dir / "google_16k"
        if google_dir.is_dir() and (google_dir / "textured.obj").is_file():
            return google_dir, True

        # EN: Fallback for objects without google_16k: keep mesh only.
        # ZH: 对没有 google_16k 的对象回退到 poisson，但只保留几何网格。
        poisson_dir = obj_dir / "poisson"
        if poisson_dir.is_dir() and (poisson_dir / "textured.obj").is_file():
            return poisson_dir, False

        for child in sorted(p for p in obj_dir.iterdir() if p.is_dir()):
            if (child / "textured.obj").is_file():
                return child, False
        return None

    def _fetch_objects(self) -> list[str]:
        response = urlopen(self.objects_url)
        return json.loads(response.read())["objects"]

    def _tgz_url(self, obj_name: str, file_type: str) -> str:
        if file_type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
            return f"{self.base_url}berkeley/{obj_name}/{obj_name}_{file_type}.tgz"
        if file_type == "berkeley_processed":
            return f"{self.base_url}berkeley/{obj_name}/{obj_name}_berkeley_meshes.tgz"
        return f"{self.base_url}google/{obj_name}_{file_type}.tgz"

    def _check_url(self, url: str) -> bool:
        try:
            request = Request(url)
            request.get_method = lambda: "HEAD"
            urlopen(request)
            return True
        except Exception:
            return False

    def _download_file(self, url: str, filename: Path) -> int:
        req = urlopen(url)
        tmp = filename.with_suffix(filename.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        total = 0
        with open(tmp, "wb") as f:
            while True:
                buf = req.read(1024 * 1024)
                if not buf:
                    break
                f.write(buf)
                total += len(buf)
        tmp.replace(filename)
        return total

    def _extract_tgz(self, filename: Path, out_dir: Path) -> None:
        with tarfile.open(filename, "r:gz") as tf:
            tf.extractall(path=out_dir)
        filename.unlink(missing_ok=True)

    def _load_mass_map(self, cfg: IngestConfig) -> dict[str, float]:
        mass_path = cfg.repo_root / "src" / "ycb_mass.json"
        if not mass_path.exists():
            return {}
        with open(mass_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k): float(v) for k, v in raw.items()}

    def _resolve_mass(self, mass_map: dict[str, float], object_id: str) -> float:
        if object_id in mass_map:
            return float(mass_map[object_id])
        if "_" in object_id:
            tail = object_id.split("_", 1)[1]
            if tail in mass_map:
                return float(mass_map[tail])
        return DEFAULT_MASS_KG

    def download(self, cfg: IngestConfig) -> DownloadReport:
        out_dir = cfg.source_download_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report = DownloadReport(source=self.source_name)

        objects = self._fetch_objects()
        tasks: list[tuple[str, str]] = []
        for obj_name in objects:
            for file_type in self.files_to_download:
                tasks.append((obj_name, file_type))

        def done_marker(obj_name: str, file_type: str) -> Path:
            if file_type == "berkeley_processed":
                return out_dir / obj_name / "poisson" / "textured.obj"
            if file_type == "google_16k":
                return out_dir / obj_name / "google_16k" / "textured.obj"
            return out_dir / obj_name

        bar = tqdm(total=len(tasks), desc="YCB download", unit="pkg") if tqdm is not None else None
        for obj_name, file_type in tasks:
            try:
                if not cfg.force and done_marker(obj_name, file_type).exists():
                    continue

                url = self._tgz_url(obj_name, file_type)
                if not self._check_url(url):
                    report.notes.append(f"skip invalid url: {url}")
                    continue

                filename = out_dir / f"{obj_name}_{file_type}.tgz"
                if filename.exists():
                    filename.unlink(missing_ok=True)

                download_ok = False
                last_err: str | None = None
                for _ in range(2):
                    try:
                        size = self._download_file(url, filename)
                        if size <= 0:
                            last_err = "empty download file"
                            filename.unlink(missing_ok=True)
                            continue
                        self._extract_tgz(filename, out_dir)
                        download_ok = True
                        break
                    except tarfile.ReadError as exc:
                        last_err = f"bad tgz: {exc}"
                        filename.unlink(missing_ok=True)
                    except Exception as exc:
                        last_err = str(exc)
                        filename.unlink(missing_ok=True)

                if not download_ok:
                    report.notes.append(f"failed {obj_name}:{file_type} ({last_err})")
                    continue

                report.downloaded_files.append(relative_to_repo(cfg.repo_root, out_dir))
            finally:
                if bar is not None:
                    bar.update(1)
        if bar is not None:
            bar.close()

        return report

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        src_dir = cfg.source_download_dir
        processed_dir = cfg.source_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not src_dir.exists():
            report.notes.append(f"missing download dir: {src_dir}")
            return report

        obj_dirs = sorted(p for p in src_dir.iterdir() if p.is_dir())
        bar = tqdm(total=len(obj_dirs), desc="YCB organize", unit="obj") if tqdm is not None else None
        for obj_dir in obj_dirs:
            try:
                pick = self._choose_source_mesh_dir(obj_dir)
                if pick is None:
                    continue
                mesh_dir, has_google_texture = pick

                obj_name = sanitize_object_id(f"{self.source_name}_{obj_dir.name}")
                dst_dir = processed_dir / obj_name
                if dst_dir.exists() and cfg.force:
                    shutil.rmtree(dst_dir)
                dst_dir.mkdir(parents=True, exist_ok=True)

                dst_obj = dst_dir / "raw.obj"
                if dst_obj.exists() and not cfg.force:
                    continue

                src_obj = mesh_dir / "textured.obj"
                if not src_obj.exists():
                    report.failed_items.append(f"missing textured.obj for {obj_name}")
                    continue

                # EN: Always re-load the source mesh with trimesh so YCB organize
                # follows the same stage-1 export style as normalized datasets.
                # ZH: 始终先用 trimesh 重新加载源 mesh，这样 YCB 的 organize
                # 与归一化数据集共享同一套一阶段导出风格。
                # EN: Keep google_16k textured meshes as intact as possible and
                # avoid reindexing through unreferenced-vertex cleanup there.
                # For non-textured fallback branches, cleanup is still fine.
                # ZH: 对 google_16k 纹理网格尽量保持原始索引结构，不做未引用顶点清理；
                # 对无纹理回退分支则仍允许清理。
                ok_load, load_reason, mesh = self.load_obj_mesh(
                    src_obj,
                    remove_unreferenced_vertices=(not has_google_texture),
                )
                if not ok_load or mesh is None:
                    report.failed_items.append(f"load failed for {obj_name}: {load_reason}")
                    continue

                try:
                    # EN: Texture export is gated only by whether this object came
                    # from google_16k. This avoids exporting poisson's tiny fallback
                    # texture placeholders as real stage-1 textures.
                    # ZH: 是否导出纹理仅取决于对象是否来自 google_16k，
                    # 这样可以避免把 poisson 分支里的极小占位贴图当成真实纹理导出。
                    self.export_trimesh_obj_assets(
                        mesh,
                        dst_dir,
                        export_texture=has_google_texture,
                    )
                except Exception as exc:
                    report.failed_items.append(f"export failed for {obj_name}: {exc}")
                    continue

                report.organized_objects += 1
            finally:
                if bar is not None:
                    bar.update(1)
        if bar is not None:
            bar.close()

        self.write_manifest_for_organize(cfg, report)
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        mass_map = self._load_mass_map(cfg)
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage="https://www.ycbbenchmarks.com/object-models/",
            download_method="http",
            notes="Downloaded from YCB benchmarks S3 endpoints",
            default_mass_kg=DEFAULT_MASS_KG,
            mass_resolver=lambda object_id, _obj_dir: self._resolve_mass(mass_map, object_id),
        )
