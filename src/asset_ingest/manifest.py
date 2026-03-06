"""Manifest dataclasses and validation for pre-processing ingest stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


TEXTURE_POLICY_VALUES = {"all", "none", "mixed", "unknown"}
TEXTURE_STATE_VALUES = {"true", "false", "unknown"}


@dataclass
class ManifestSource:
    homepage: str | None = None
    download_method: str | None = None
    notes: str | None = None


@dataclass
class ManifestSummary:
    num_objects: int = 0
    num_categories: int = 0
    has_texture_policy: str = "unknown"
    default_mass_kg: float = 0.1


@dataclass
class ObjectRecord:
    object_id: str
    name: str
    category: str | None
    mesh_path: str
    mesh_format: str
    mass_kg: float = 0.1
    has_texture: str = "unknown"
    mtl_path: str | None = None
    texture_files: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    license: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestManifest:
    dataset: str
    version: str | None
    generated_at: str
    source: ManifestSource
    summary: ManifestSummary
    objects: list[ObjectRecord]

    @classmethod
    def create(cls, dataset: str, version: str | None = None) -> "IngestManifest":
        return cls(
            dataset=dataset,
            version=version,
            generated_at=datetime.now(timezone.utc).isoformat(),
            source=ManifestSource(),
            summary=ManifestSummary(),
            objects=[],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "IngestManifest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source = ManifestSource(**data.get("source", {}))
        summary = ManifestSummary(**data.get("summary", {}))
        objects = [ObjectRecord(**obj) for obj in data.get("objects", [])]
        return cls(
            dataset=data["dataset"],
            version=data.get("version"),
            generated_at=data.get("generated_at", ""),
            source=source,
            summary=summary,
            objects=objects,
        )


def validate_manifest_dict(
    data: dict[str, Any],
    repo_root: Path | None = None,
    check_paths: bool = True,
) -> list[str]:
    errs: list[str] = []

    required_top = ["dataset", "generated_at", "source", "summary", "objects"]
    for key in required_top:
        if key not in data:
            errs.append(f"missing top-level field: {key}")
    if errs:
        return errs

    summary = data.get("summary", {})
    objects = data.get("objects", [])

    if summary.get("has_texture_policy") not in TEXTURE_POLICY_VALUES:
        errs.append(
            "summary.has_texture_policy must be one of "
            f"{sorted(TEXTURE_POLICY_VALUES)}"
        )

    default_mass = summary.get("default_mass_kg", 0.1)
    if not isinstance(default_mass, (int, float)) or float(default_mass) <= 0:
        errs.append("summary.default_mass_kg must be a positive number")

    if summary.get("num_objects") != len(objects):
        errs.append(
            f"summary.num_objects={summary.get('num_objects')} does not match "
            f"len(objects)={len(objects)}"
        )

    ids = set()
    categories = set()
    for idx, obj in enumerate(objects):
        prefix = f"objects[{idx}]"
        for key in [
            "object_id",
            "name",
            "mesh_path",
            "mesh_format",
            "has_texture",
            "mass_kg",
        ]:
            if key not in obj or obj[key] in (None, ""):
                errs.append(f"{prefix}: missing required field '{key}'")

        object_id = obj.get("object_id")
        if object_id:
            if object_id in ids:
                errs.append(f"duplicate object_id: {object_id}")
            ids.add(object_id)

        has_texture = obj.get("has_texture")
        if has_texture not in TEXTURE_STATE_VALUES:
            errs.append(
                f"{prefix}.has_texture must be one of {sorted(TEXTURE_STATE_VALUES)}"
            )

        mass = obj.get("mass_kg")
        if not isinstance(mass, (int, float)) or float(mass) <= 0:
            errs.append(f"{prefix}.mass_kg must be a positive number")

        if obj.get("category"):
            categories.add(obj["category"])

        if has_texture == "true":
            if not obj.get("mtl_path") and not obj.get("texture_files"):
                errs.append(
                    f"{prefix}: has_texture=true but neither mtl_path nor texture_files found"
                )

        if check_paths and repo_root is not None:
            mesh_rel = obj.get("mesh_path")
            if mesh_rel:
                mesh_abs = repo_root / mesh_rel
                if not mesh_abs.exists():
                    errs.append(f"{prefix}.mesh_path does not exist: {mesh_rel}")

    if summary.get("num_categories") != len(categories):
        errs.append(
            f"summary.num_categories={summary.get('num_categories')} does not match "
            f"unique categories={len(categories)}"
        )

    return errs
