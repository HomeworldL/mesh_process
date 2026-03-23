"""YCB source adapter: download from official endpoints and normalize to raw format."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import tarfile
from urllib.request import Request, urlopen

from .base import (
    CANONICAL_MTL_NAME,
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


class YCBAdapter(BaseIngestAdapter):
    source_name = "YCB"
    version = None

    objects_url = "https://ycb-benchmarks.s3.amazonaws.com/data/objects.json"
    base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
    files_to_download = ["berkeley_processed", "google_16k"]

    @staticmethod
    def _choose_source_mesh_dir(obj_dir: Path) -> tuple[Path, bool] | None:
        # If google_16k exists, treat object as textured-capable.
        google_dir = obj_dir / "google_16k"
        if google_dir.is_dir() and (google_dir / "textured.obj").is_file():
            return google_dir, True

        # Fallback for objects without google_16k: only keep mesh.
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

    @staticmethod
    def _check_url(url: str) -> bool:
        try:
            request = Request(url)
            request.get_method = lambda: "HEAD"
            urlopen(request)
            return True
        except Exception:
            return False

    @staticmethod
    def _download_file(url: str, filename: Path) -> int:
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

    @staticmethod
    def _resolve_mass(mass_map: dict[str, float], object_id: str) -> float:
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

        for obj_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
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

            shutil.copy2(src_obj, dst_obj)
            if has_google_texture:
                mtl_src = mesh_dir / "textured.mtl"
                tex_src = mesh_dir / "texture_map.png"
                canonicalize_texture_assets(
                    dst_dir=dst_dir,
                    raw_obj_path=dst_obj,
                    texture_src=(tex_src if tex_src.exists() else None),
                    mtl_src=(mtl_src if mtl_src.exists() else None),
                    create_mtl_if_texture=False,
                )

            report.organized_objects += 1

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
