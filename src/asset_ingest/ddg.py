"""DDG adapter: derive DDG objects by copying processed ddg* objects from DGN."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import shutil

from .base import (
    CANONICAL_RAW_OBJ_NAME,
    BaseIngestAdapter,
    DownloadReport,
    IngestConfig,
    OrganizeReport,
    relative_to_repo,
)
from .manifest import IngestManifest


class DDGAdapter(BaseIngestAdapter):
    source_name = "DDG"
    version = None
    normalized_default_mass_kg = 50.0

    def download(self, cfg: IngestConfig) -> DownloadReport:
        report = DownloadReport(source=self.source_name)
        report.notes.append("DDG download skipped: derived from processed/DGN")
        return report

    def _candidate_dirs(self, dgn_processed_dir: Path) -> list[Path]:
        if not dgn_processed_dir.exists():
            return []
        return sorted(
            p for p in dgn_processed_dir.iterdir() if p.is_dir() and p.name.lower().startswith("ddg")
        )

    def _missing_required_outputs(self, obj_dir: Path) -> list[str]:
        required_paths = [
            obj_dir / CANONICAL_RAW_OBJ_NAME,
            obj_dir / "manifold.obj",
            obj_dir / "coacd.obj",
            obj_dir / "simplified.obj",
            obj_dir / "visual.obj",
        ]
        missing: list[str] = []
        for path in required_paths:
            if (not path.is_file()) or path.stat().st_size <= 0:
                missing.append(path.name)

        meshes_dir = obj_dir / "meshes"
        if not any(meshes_dir.glob("coacd_convex_piece_*.obj")):
            missing.append("meshes/coacd_convex_piece_*.obj")
        if not any(obj_dir.glob("*.xml")):
            missing.append("*.xml")
        return missing

    def _load_json(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_process_manifest(
        self,
        ddg_manifest: dict[str, Any],
        dgn_process_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        out = deepcopy(ddg_manifest)
        dgn_objects = {
            obj.get("object_id"): obj
            for obj in dgn_process_manifest.get("objects", [])
            if isinstance(obj, dict) and isinstance(obj.get("object_id"), str)
        }

        merged_objects: list[dict[str, Any]] = []
        num_success = 0
        num_failed = 0

        for ddg_obj in out.get("objects", []):
            merged = deepcopy(ddg_obj)
            object_id = merged.get("object_id")
            dgn_obj = dgn_objects.get(object_id)
            if dgn_obj is None:
                merged["process_status"] = "missing"
                merged["process_error"] = f"object_id={object_id} missing in DGN manifest.process_meshes.json"
                merged["center_of_mass"] = None
                merged["principal_moments"] = None
                merged["principal_axes"] = None
                merged["visual_obj_path"] = None
                merged["visual_mtl_path"] = None
                merged["visual_texture_path"] = None
                num_failed += 1
            else:
                for key, value in dgn_obj.items():
                    if key == "object_id":
                        continue
                    merged[key] = deepcopy(value)
                if merged.get("process_status") == "success":
                    num_success += 1
                else:
                    num_failed += 1
            merged_objects.append(merged)

        out["objects"] = merged_objects
        out["process_meshes"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": "src/asset_ingest/ddg.py",
            "num_success": num_success,
            "num_failed": num_failed,
            "num_skipped": 0,
            "num_total": len(merged_objects),
            "elapsed_seconds": None,
            "avg_seconds_per_object": None,
            "num_processed_objects": len(merged_objects),
        }
        return out

    def _write_process_manifest_for_organize(
        self,
        cfg: IngestConfig,
        report: OrganizeReport,
        *,
        ddg_manifest_path: Path,
        dgn_process_manifest_path: Path,
    ) -> None:
        ddg_manifest = self._load_json(ddg_manifest_path)
        dgn_process_manifest = self._load_json(dgn_process_manifest_path)
        ddg_process_manifest = self._build_process_manifest(ddg_manifest, dgn_process_manifest)
        out_path = cfg.source_processed_dir / "manifest.process_meshes.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ddg_process_manifest, f, indent=2, ensure_ascii=False)
        report.notes.append(f"DDG process manifest: {relative_to_repo(cfg.repo_root, out_path)}")

    def organize(self, cfg: IngestConfig) -> OrganizeReport:
        report = OrganizeReport(source=self.source_name)
        dgn_processed_dir = cfg.processed_root / "DGN"
        ddg_processed_dir = cfg.source_processed_dir

        if not dgn_processed_dir.exists():
            report.notes.append(f"missing source dataset: {relative_to_repo(cfg.repo_root, dgn_processed_dir)}")
            return report

        dgn_manifest_path = dgn_processed_dir / "manifest.json"
        dgn_process_manifest_path = dgn_processed_dir / "manifest.process_meshes.json"
        if not dgn_manifest_path.is_file():
            report.notes.append(f"missing DGN manifest: {relative_to_repo(cfg.repo_root, dgn_manifest_path)}")
            return report
        if not dgn_process_manifest_path.is_file():
            report.notes.append(
                f"missing DGN process manifest: {relative_to_repo(cfg.repo_root, dgn_process_manifest_path)}"
            )
            return report

        source_dirs = self._candidate_dirs(dgn_processed_dir)
        if not source_dirs:
            report.notes.append("no ddg* object folders found under processed/DGN")
            return report

        if cfg.force and ddg_processed_dir.exists():
            shutil.rmtree(ddg_processed_dir)
        ddg_processed_dir.mkdir(parents=True, exist_ok=True)

        for src_dir in source_dirs:
            missing = self._missing_required_outputs(src_dir)
            if missing:
                report.failed_items.append(
                    f"{src_dir.name}: missing required DGN processed outputs: {', '.join(missing)}"
                )
                continue

            dst_dir = ddg_processed_dir / src_dir.name
            if dst_dir.exists():
                if cfg.force:
                    shutil.rmtree(dst_dir)
                else:
                    report.notes.append(f"skip existing: {relative_to_repo(cfg.repo_root, dst_dir)}")
                    continue

            shutil.copytree(src_dir, dst_dir)
            report.organized_objects += 1

        if not any(p.is_dir() for p in ddg_processed_dir.iterdir()):
            report.notes.append("no DDG objects copied from processed/DGN")
            return report

        self.write_manifest_for_organize(cfg, report)
        ddg_manifest_path = cfg.source_processed_dir / "manifest.json"
        self._write_process_manifest_for_organize(
            cfg,
            report,
            ddg_manifest_path=ddg_manifest_path,
            dgn_process_manifest_path=dgn_process_manifest_path,
        )
        report.notes.append(f"copied from: {relative_to_repo(cfg.repo_root, dgn_processed_dir)}")
        report.notes.append(
            "DDG organize reuses processed DGN ddg* object folders and preserves copied stage-2/stage-3 outputs."
        )
        return report

    def build_manifest(self, cfg: IngestConfig) -> IngestManifest:
        return self.build_manifest_from_processed_dir(
            cfg,
            homepage="https://huggingface.co/datasets/JiayiChenPKU/BODex",
            download_method="derived_from_processed_dgn",
            notes="Copied from processed/DGN object folders whose ids start with ddg",
            default_mass_kg=self.normalized_default_mass_kg,
            missing_has_texture_policy="none",
            mtl_resolver=lambda _obj_dir: None,
        )
