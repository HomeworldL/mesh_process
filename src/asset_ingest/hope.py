"""HOPE source adapter: download folder, normalize meshes, build manifest."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen

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


class HOPEAdapter(BaseIngestAdapter):
    source_name = "HOPE"
    version = None
    mesh_scale = 0.01  # HOPE raw meshes are in centimeters; organize to meters.

    folder_url = "https://drive.google.com/drive/folders/1jiJS9KgcYAkfb8KJPp5MRlB0P11BStft"
    download_dirname = "hope_objects"
    required_files = {"textured.obj", "textured.mtl", "texture_map.png"}

    @staticmethod
    def _subset_filter(cfg: IngestConfig) -> set[str] | None:
        if not cfg.subset:
            return None
        values = {x.strip() for x in cfg.subset.split(",") if x.strip()}
        return values or None

    def _list_drive_files(self, out_dir: Path):
        try:
            import gdown  # type: ignore
        except Exception:
            return None

        dst_root = out_dir / self.download_dirname
        try:
            files = gdown.download_folder(
                url=self.folder_url,
                output=str(dst_root),
                quiet=True,
                remaining_ok=True,
                skip_download=True,
            )
        except TypeError:
            try:
                files = gdown.download_folder(
                    url=self.folder_url,
                    output=str(dst_root),
                    quiet=True,
                    remaining_ok=True,
                    skip_download=True,
                )
            except Exception:
                return None
        except Exception:
            return None
        return files

    def _download_folder(self, out_dir: Path, subset: set[str] | None = None) -> tuple[Path | None, list[str]]:
        """Download required HOPE files only and return local root + notes."""
        notes: list[str] = []
        dst_root = out_dir / self.download_dirname
        dst_root.mkdir(parents=True, exist_ok=True)

        files = self._list_drive_files(out_dir)
        if not files:
            return None, notes

        tasks: list[tuple[str, str, str, Path]] = []
        expected_by_object: dict[str, set[str]] = {}

        for item in files:
            file_id = getattr(item, "id", None)
            path_value = getattr(item, "path", None)
            local_path = getattr(item, "local_path", None)
            if not file_id or not path_value or not local_path:
                continue

            parts = Path(path_value).parts
            if "google_16k" not in parts:
                continue
            idx = parts.index("google_16k")
            if idx < 1 or idx + 1 >= len(parts):
                continue

            object_name = parts[idx - 1]
            filename = parts[-1]
            if filename not in self.required_files:
                continue
            if subset is not None and object_name not in subset:
                continue

            expected_by_object.setdefault(object_name, set()).add(filename)
            tasks.append((object_name, filename, str(file_id), Path(local_path)))

        if not tasks:
            notes.append("no required HOPE files matched from folder listing")
            return dst_root, notes

        try:
            import gdown  # type: ignore
        except Exception:
            return None, notes
        gdown_bin = shutil.which("gdown")

        def _direct_download_by_id(file_id: str, output_path: Path) -> bool:
            # Fallback for gdown failures: direct public download endpoint.
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = output_path.with_suffix(output_path.suffix + ".part")
                if tmp.exists():
                    tmp.unlink()
                url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
                with urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                if tmp.exists() and tmp.stat().st_size > 0:
                    tmp.replace(output_path)
                    return True
            except Exception:
                pass
            return False

        completed: set[str] = set()
        failures: list[str] = []

        def _is_object_complete(obj_name: str) -> bool:
            exp = expected_by_object.get(obj_name, set())
            if not exp:
                return False
            obj_dir = dst_root / obj_name / "google_16k"
            return all((obj_dir / fn).exists() for fn in exp)

        for obj_name in sorted(expected_by_object):
            if _is_object_complete(obj_name):
                completed.add(obj_name)

        pbar = tqdm(total=len(expected_by_object), desc="HOPE download", unit="obj") if tqdm is not None else None
        if pbar is not None and completed:
            pbar.update(len(completed))

        for obj_name, filename, file_id, local_path in tasks:
            if _is_object_complete(obj_name):
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                ok = False
                last_err: str | None = None
                for _ in range(3):
                    try:
                        try:
                            gdown.download(
                                id=file_id,
                                output=str(local_path),
                                quiet=True,
                                resume=True,
                            )
                        except TypeError:
                            gdown.download(
                                id=file_id,
                                output=str(local_path),
                                quiet=True,
                            )
                        if local_path.exists() and local_path.stat().st_size > 0:
                            ok = True
                            break
                        last_err = "empty output"
                    except Exception as exc:
                        last_err = str(exc)
                        if gdown_bin is not None:
                            try:
                                subprocess.run(
                                    [gdown_bin, f"https://drive.google.com/uc?id={file_id}", "-O", str(local_path)],
                                    check=True,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                )
                                if local_path.exists() and local_path.stat().st_size > 0:
                                    ok = True
                                    break
                            except Exception:
                                pass
                        if _direct_download_by_id(file_id, local_path):
                            ok = True
                            break
                if not ok:
                    failures.append(f"{obj_name}/{filename}: {last_err or 'download failed'}")
                    continue

            if _is_object_complete(obj_name) and obj_name not in completed:
                completed.add(obj_name)
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        if failures:
            notes.append(f"failed files: {len(failures)}")
            notes.extend(failures[:10])
        notes.append(f"downloaded/ready objects: {len(completed)}/{len(expected_by_object)}")
        return dst_root, notes

    @staticmethod
    def _resolve_objects_root(source_dir: Path) -> Path:
        if not source_dir.exists():
            return source_dir
        # Expected layout: <root>/<object_name>/google_16k/textured.obj
        direct_candidates = [p for p in source_dir.iterdir() if p.is_dir() and (p / "google_16k").is_dir()]
        if direct_candidates:
            return source_dir
        nested = sorted(
            p for p in source_dir.rglob("*")
            if p.is_dir() and any((c / "google_16k").is_dir() for c in p.iterdir() if c.is_dir())
        )
        if nested:
            return nested[0]
        return source_dir

    @staticmethod
    def _scale_obj_vertices(obj_path: Path, scale: float) -> int:
        """Scale OBJ `v` lines in-place and return count of scaled vertices."""
        lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        out_lines: list[str] = []
        scaled = 0
        for line in lines:
            if not line.startswith("v "):
                out_lines.append(line)
                continue
            parts = line.split()
            if len(parts) < 4:
                out_lines.append(line)
                continue
            try:
                x = float(parts[1]) * scale
                y = float(parts[2]) * scale
                z = float(parts[3]) * scale
            except ValueError:
                out_lines.append(line)
                continue

            # Keep optional per-vertex color/weight terms intact.
            out_lines.append(" ".join(["v", f"{x:.9g}", f"{y:.9g}", f"{z:.9g}", *parts[4:]]))
            scaled += 1

        obj_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        return scaled

    def download(self, cfg: IngestConfig) -> DownloadReport:
        out_dir = cfg.source_download_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        report = DownloadReport(source=self.source_name)
        subset = self._subset_filter(cfg)

        source_dir = out_dir / self.download_dirname
        if source_dir.exists() and cfg.force:
            shutil.rmtree(source_dir)

        if source_dir.exists() and not cfg.force:
            report.notes.append(f"skip existing: {relative_to_repo(cfg.repo_root, source_dir)}")
            return report

        downloaded_root, dl_notes = self._download_folder(out_dir, subset=subset)
        if downloaded_root is None:
            raise RuntimeError(
                "Failed to download HOPE folder via gdown. "
                "Install gdown and retry."
            )
        if not source_dir.exists():
            report.notes.append("downloaded via official HOPE folder link")
            report.downloaded_files.append(relative_to_repo(cfg.repo_root, downloaded_root))
        else:
            if downloaded_root is None:
                raise RuntimeError("HOPE download finished without output dir")
            report.notes.append("download refreshed")
        report.notes.extend(dl_notes)

        return report

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        source_dir = cfg.source_download_dir / self.download_dirname
        src_root = self._resolve_objects_root(source_dir)
        processed_dir = cfg.source_processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)
        report = OrganizeReport(source=self.source_name)

        if not source_dir.exists():
            report.notes.append(f"missing source dir: {source_dir}")
            return report

        subset = self._subset_filter(cfg)
        seen_ids: set[str] = set()

        object_dirs = sorted(
            p for p in src_root.iterdir() if p.is_dir() and (p / "google_16k").is_dir()
        )
        iterable = tqdm(object_dirs, desc="HOPE organize", unit="obj") if tqdm is not None else object_dirs

        for folder in iterable:
            if subset is not None and folder.name not in subset:
                continue

            mesh_dir = folder / "google_16k"
            src_obj = mesh_dir / "textured.obj"
            if not src_obj.exists():
                continue

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

            dst_obj = dst_dir / "raw.obj"
            if dst_obj.exists() and not cfg.force:
                continue

            shutil.copy2(src_obj, dst_obj)
            scaled_vertices = self._scale_obj_vertices(dst_obj, self.mesh_scale)
            if scaled_vertices == 0:
                raise RuntimeError(f"HOPE organize failed to scale mesh (no vertices found): {dst_obj}")
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

        report.notes.append(f"organized from {src_root}")
        report.notes.append("scaled raw.obj by 0.01 (cm->m)")
        self.write_manifest_for_organize(cfg, report)
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage="https://github.com/swtyree/hope-dataset",
            download_method="google_drive",
            notes=(
                "Official folder https://drive.google.com/drive/folders/"
                "1jiJS9KgcYAkfb8KJPp5MRlB0P11BStft ; "
                "per-object google_16k/textured.obj bundle"
            ),
            default_mass_kg=DEFAULT_MASS_KG,
        )
