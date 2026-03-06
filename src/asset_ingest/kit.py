"""KIT ObjectModels adapter: crawl official list, download meshes.zip, normalize meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import urllib.parse
import zipfile
from urllib.request import urlopen

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


@dataclass(frozen=True)
class _KitArchive:
    object_name: str
    object_id: str
    url: str


class KITAdapter(BaseIngestAdapter):
    source_name = "KIT"
    version = None

    base_url = "https://archive.iar.kit.edu/Projects/ObjectModelsWebUI/"
    list_url = f"{base_url}index.php?section=listAll"

    @staticmethod
    def _subset_filter(cfg: IngestConfig) -> set[str] | None:
        if not cfg.subset:
            return None
        values = {x.strip() for x in cfg.subset.split(",") if x.strip()}
        return values or None

    def _fetch_archives(self) -> list[_KitArchive]:
        html = urlopen(self.list_url).read().decode("utf-8", "ignore")
        rel_links = re.findall(r'href="(tmp\.php\?[^"]+)"', html)

        out: list[_KitArchive] = []
        seen_names: set[str] = set()
        for rel in rel_links:
            full_url = urllib.parse.urljoin(self.base_url, rel)
            parsed = urllib.parse.urlparse(full_url)
            q = urllib.parse.parse_qs(parsed.query)
            dp = (q.get("dp") or [""])[0]
            if not dp.endswith("/meshes.zip"):
                continue

            parts = dp.split("/")
            # Expected: Objects/<object_name>/meshes.zip
            if len(parts) < 3 or parts[0] != "Objects":
                continue
            object_name = parts[1].strip()
            if not object_name or object_name in seen_names:
                continue

            obj_id = (q.get("id") or [""])[0].strip() or sanitize_object_id(object_name)
            out.append(_KitArchive(object_name=object_name, object_id=obj_id, url=full_url))
            seen_names.add(object_name)

        out.sort(key=lambda x: x.object_name.lower())
        return out

    @staticmethod
    def _download_file(url: str, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        with urlopen(url) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dst)

    @staticmethod
    def _extract_mesh_zip(zip_path: Path, out_dir: Path, force: bool) -> None:
        if out_dir.exists() and force:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                # Keep only the actual mesh files and flatten one level.
                name = Path(info.filename).name
                if not name:
                    continue
                target = out_dir / name
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    def download(self, cfg: IngestConfig) -> DownloadReport:
        out_dir = cfg.source_download_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report = DownloadReport(source=self.source_name)

        archives = self._fetch_archives()
        subset = self._subset_filter(cfg)
        if subset is not None:
            archives = [a for a in archives if a.object_name in subset]

        zip_root = out_dir / "archives"
        extract_root = out_dir / "objects"
        zip_root.mkdir(parents=True, exist_ok=True)
        extract_root.mkdir(parents=True, exist_ok=True)

        bar = tqdm(total=len(archives), desc="KIT download", unit="obj") if tqdm is not None else None
        for arc in archives:
            safe_name = sanitize_object_id(arc.object_name)
            zip_path = zip_root / f"{arc.object_id}_{safe_name}.zip"
            object_dir = extract_root / arc.object_name
            try:
                if not zip_path.exists() or cfg.force:
                    self._download_file(arc.url, zip_path)
                    report.downloaded_files.append(relative_to_repo(cfg.repo_root, zip_path))

                marker = list(object_dir.glob("*.obj"))
                if cfg.force or not marker:
                    self._extract_mesh_zip(zip_path, object_dir, force=cfg.force)
            except Exception as exc:
                report.notes.append(f"failed {arc.object_name}: {exc}")
            finally:
                if bar is not None:
                    bar.update(1)
        if bar is not None:
            bar.close()

        report.notes.append(f"archives discovered: {len(archives)}")
        return report

    @staticmethod
    def _choose_obj(obj_paths: list[Path]) -> Path:
        def score(path: Path) -> tuple[int, str]:
            n = path.name.lower()
            if n.endswith("_orig_tex.obj"):
                return (0, n)
            if n.endswith("_25k_tex.obj"):
                return (1, n)
            if n.endswith("_5k_tex.obj"):
                return (2, n)
            if n.endswith("_800_tex.obj"):
                return (3, n)
            if n.endswith("_tex.obj"):
                return (4, n)
            if n.endswith("_orig.obj"):
                return (5, n)
            return (10, n)

        return sorted(obj_paths, key=score)[0]

    @staticmethod
    def _pick_sidecar(folder: Path, chosen_obj: Path) -> tuple[Path | None, Path | None]:
        # Prefer sidecars matching selected OBJ stem.
        stem = chosen_obj.stem
        mtl = folder / f"{stem}.mtl"
        tex = folder / f"{stem}.png"
        mtl_src = mtl if mtl.is_file() else None
        tex_src = tex if tex.is_file() else None

        if mtl_src is None:
            mtl_src = next((p for p in sorted(folder.glob("*.mtl")) if p.is_file()), None)
        if tex_src is None:
            tex_src = next((p for p in sorted(folder.glob("*.png")) if p.is_file()), None)
        return mtl_src, tex_src

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        src_root = cfg.source_download_dir / "objects"
        processed_dir = cfg.source_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not src_root.exists():
            report.notes.append(f"missing source dir: {src_root}")
            return report

        subset = self._subset_filter(cfg)
        seen_ids: set[str] = set()

        for folder in sorted(p for p in src_root.iterdir() if p.is_dir()):
            if subset is not None and folder.name not in subset:
                continue

            objs = [x for x in folder.iterdir() if x.is_file() and x.suffix.lower() == ".obj"]
            if not objs:
                continue

            chosen = self._choose_obj(objs)
            base_name = sanitize_object_id(folder.name)
            base_id = sanitize_object_id(f"{self.source_name}_{base_name}")
            object_id = base_id
            suffix = 1
            while object_id in seen_ids:
                suffix += 1
                object_id = f"{base_id}_{suffix}"
            seen_ids.add(object_id)

            dst_dir = processed_dir / object_id
            if dst_dir.exists() and cfg.force:
                shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)

            dst_obj = dst_dir / CANONICAL_RAW_OBJ_NAME
            if dst_obj.exists() and not cfg.force:
                continue

            shutil.copy2(chosen, dst_obj)
            mtl_src, tex_src = self._pick_sidecar(folder, chosen)
            canonicalize_texture_assets(
                dst_dir=dst_dir,
                raw_obj_path=dst_obj,
                texture_src=tex_src,
                mtl_src=mtl_src,
                create_mtl_if_texture=False,
            )
            report.organized_objects += 1

        report.notes.append(f"organized from {src_root}")
        self.write_manifest_for_organize(cfg, report)
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        raw_dir = cfg.source_processed_dir
        manifest = IngestManifest.create(dataset=self.source_name, version=self.version)
        manifest.source = ManifestSource(
            homepage=self.list_url,
            download_method="http_crawl",
            notes="Crawled listAll page and downloaded per-object meshes.zip via tmp.php links",
        )

        if not raw_dir.exists():
            manifest.summary = ManifestSummary(
                num_objects=0,
                num_categories=0,
                has_texture_policy="unknown",
                default_mass_kg=DEFAULT_MASS_KG,
            )
            return manifest

        objects: list[ObjectRecord] = []
        texture_true_count = 0
        texture_false_count = 0

        for obj_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
            object_id = obj_dir.name
            mesh_path = obj_dir / CANONICAL_RAW_OBJ_NAME
            if not mesh_path.exists():
                continue

            mtl_path = obj_dir / CANONICAL_MTL_NAME
            texture_files = [p.name for p in obj_dir.glob("*.png")]
            has_texture = "true" if texture_files else "false"
            if has_texture == "true":
                texture_true_count += 1
            else:
                texture_false_count += 1

            objects.append(
                ObjectRecord(
                    object_id=object_id,
                    name=object_id,
                    category=None,
                    mesh_path=relative_to_repo(cfg.repo_root, mesh_path),
                    mesh_format="obj",
                    mass_kg=DEFAULT_MASS_KG,
                    has_texture=has_texture,
                    mtl_path=(relative_to_repo(cfg.repo_root, mtl_path) if mtl_path.exists() else None),
                    texture_files=texture_files,
                )
            )

        if texture_true_count and texture_false_count:
            texture_policy = "mixed"
        elif texture_true_count:
            texture_policy = "all"
        elif texture_false_count:
            texture_policy = "none"
        else:
            texture_policy = "unknown"

        manifest.objects = objects
        manifest.summary = ManifestSummary(
            num_objects=len(objects),
            num_categories=0,
            has_texture_policy=texture_policy,
            default_mass_kg=DEFAULT_MASS_KG,
        )
        return manifest
