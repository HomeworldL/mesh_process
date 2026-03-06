"""Base abstraction for object-source ingest (download, organize, manifest)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from urllib.request import urlopen

from .manifest import IngestManifest, validate_manifest_dict

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
        raise NotImplementedError

    @abstractmethod
    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        raise NotImplementedError

    @abstractmethod
    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        raise NotImplementedError

    def validate_manifest(
        self,
        manifest: IngestManifest,
        cfg: IngestConfig,
        check_paths: bool = True,
    ) -> list[str]:
        return validate_manifest_dict(
            manifest.to_dict(),
            repo_root=cfg.repo_root,
            check_paths=check_paths,
        )

    @staticmethod
    def default_manifest_path(cfg: IngestConfig) -> Path:
        return cfg.source_processed_dir / "manifest.json"

    def write_manifest_for_organize(self, cfg: IngestConfig, report: OrganizeReport) -> None:
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

    @staticmethod
    def parse_common_args() -> argparse.ArgumentParser:
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


def extract_zip(zip_path: Path, output_dir: Path, force: bool = False, show_progress: bool = True) -> Path:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
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
