#!/usr/bin/env python3
"""Build URDF/MJCF descriptions for processed objects.

Input:
  assets/objects/processed/<dataset>/manifest.process_meshes.json

Per object (process_status=success):
  - visual mesh: visual.obj
  - collision meshes: meshes/coacd_convex_piece_*.obj
  - mass: mass_kg from manifest.process_meshes.json
  - inertia diagonal: principal_moments from manifest.process_meshes.json

Outputs are written into each object folder:
  <object_id>.urdf
  <object_id>.xml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


@dataclass
class ObjectSpec:
    object_id: str
    obj_dir: Path
    visual_rel: str
    collision_rels: list[str]
    mass_kg: float
    principal_moments: list[float]
    visual_texture_rel: str | None


class ObjectDescriptionBuilder:
    def __init__(self, dataset_root: Path, urdf_prototype: Path, xml_prototype: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        self.urdf_base = ET.parse(str(urdf_prototype)).getroot()
        self.xml_base = ET.parse(str(xml_prototype)).getroot()

    @staticmethod
    def _replace_xml_attributes(root: ET.Element, path: str, attrs: dict[str, str]) -> None:
        node = root.find(path)
        if node is None:
            return
        node.attrib.update(attrs)

    @staticmethod
    def _find_mesh_path_from_record(dataset_root: Path, object_id: str, rec: dict) -> Path:
        mesh_path = rec.get("mesh_path")
        if isinstance(mesh_path, str) and mesh_path:
            p = Path(mesh_path)
            if not p.is_absolute():
                p = (Path(__file__).resolve().parents[1] / p).resolve()
            if p.name == "raw.obj":
                return p.parent
        return dataset_root / object_id

    def _collect_specs(self, manifest: dict, selected_ids: set[str] | None) -> tuple[list[ObjectSpec], int]:
        specs: list[ObjectSpec] = []
        skipped_non_success = 0

        objects = manifest.get("objects", [])
        if not isinstance(objects, list):
            raise RuntimeError("manifest objects is not a list")

        for rec in objects:
            if not isinstance(rec, dict):
                raise RuntimeError("manifest object entry is not a dict")
            object_id = rec.get("object_id")
            if not isinstance(object_id, str) or not object_id:
                raise RuntimeError(f"invalid object_id in manifest entry: {rec}")
            if selected_ids is not None and object_id not in selected_ids:
                continue

            status = rec.get("process_status")
            if status != "success":
                skipped_non_success += 1
                continue

            obj_dir = self._find_mesh_path_from_record(self.dataset_root, object_id, rec)
            visual_abs = obj_dir / "visual.obj"
            if not visual_abs.exists():
                raise RuntimeError(f"{object_id}: missing visual.obj")

            collision_abs = sorted((obj_dir / "meshes").glob("coacd_convex_piece_*.obj"))
            if not collision_abs:
                raise RuntimeError(f"{object_id}: missing collision pieces in meshes/")

            mass_kg = rec.get("mass_kg")
            if not isinstance(mass_kg, (int, float)) or float(mass_kg) <= 0:
                raise RuntimeError(f"{object_id}: invalid mass_kg={mass_kg}")

            pm = rec.get("principal_moments")
            if not isinstance(pm, list) or len(pm) != 3:
                raise RuntimeError(f"{object_id}: invalid principal_moments={pm}")
            try:
                principal_moments = [float(pm[0]), float(pm[1]), float(pm[2])]
            except Exception:
                raise RuntimeError(f"{object_id}: non-numeric principal_moments={pm}")

            visual_png = obj_dir / "textured_visual.png"
            visual_jpegs = sorted(
                [p for p in obj_dir.glob("*_visual.jpg") if p.is_file()]
                + [p for p in obj_dir.glob("*_visual.jpeg") if p.is_file()]
            )
            if visual_jpegs and not visual_png.exists():
                raise RuntimeError(
                    f"{object_id}: non-png visual texture found; expected textured_visual.png only"
                )
            extra_pngs = sorted(
                p for p in obj_dir.glob("*_visual.png")
                if p.is_file() and p.name != "textured_visual.png"
            )
            if extra_pngs:
                raise RuntimeError(
                    f"{object_id}: unexpected extra visual textures found; expected only textured_visual.png"
                )
            visual_texture_rel = visual_png.name if visual_png.exists() else None

            specs.append(
                ObjectSpec(
                    object_id=object_id,
                    obj_dir=obj_dir,
                    visual_rel="visual.obj",
                    collision_rels=[str(p.relative_to(obj_dir)).replace(os.sep, "/") for p in collision_abs],
                    mass_kg=float(mass_kg),
                    principal_moments=principal_moments,
                    visual_texture_rel=visual_texture_rel,
                )
            )

        return specs, skipped_non_success

    def _update_urdf(self, spec: ObjectSpec) -> ET.Element:
        new_urdf = copy.deepcopy(self.urdf_base)
        new_urdf.attrib["name"] = spec.object_id

        self._replace_xml_attributes(new_urdf, ".//visual/geometry/mesh", {"filename": spec.visual_rel})
        self._replace_xml_attributes(new_urdf, ".//inertial/mass", {"value": str(spec.mass_kg)})
        self._replace_xml_attributes(new_urdf, ".//visual/origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        self._replace_xml_attributes(new_urdf, ".//collision/origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

        ixx, iyy, izz = spec.principal_moments
        self._replace_xml_attributes(
            new_urdf,
            ".//inertial/inertia",
            {
                "ixx": f"{ixx:.12g}",
                "ixy": "0.0",
                "ixz": "0.0",
                "iyy": f"{iyy:.12g}",
                "iyz": "0.0",
                "izz": f"{izz:.12g}",
            },
        )
        self._replace_xml_attributes(new_urdf, ".//inertial/origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

        self._replace_xml_attributes(new_urdf, ".//collision/geometry/mesh", {"filename": spec.collision_rels[0]})

        col = new_urdf.find(".//collision")
        link = new_urdf.find(".//link")
        if col is not None and link is not None:
            insert_pos = list(link).index(col)
            for rel in spec.collision_rels[1:]:
                clone = copy.deepcopy(col)
                mesh_elem = clone.find("geometry/mesh")
                if mesh_elem is not None:
                    mesh_elem.attrib["filename"] = rel
                insert_pos += 1
                link.insert(insert_pos, clone)

        return new_urdf

    def _update_xml(self, spec: ObjectSpec) -> ET.Element:
        new_xml = copy.deepcopy(self.xml_base)
        new_xml.attrib["model"] = spec.object_id

        visual_mesh = new_xml.find(".//asset/mesh[@name='_name_visual_mesh']")
        if visual_mesh is not None:
            visual_mesh.attrib["name"] = f"{spec.object_id}_visual_mesh"
            visual_mesh.attrib["file"] = spec.visual_rel

        collision_mesh = new_xml.find(".//asset/mesh[@name='_name_collision_mesh']")
        if collision_mesh is not None:
            collision_mesh.attrib["name"] = f"{spec.object_id}_collision_mesh_0"
            collision_mesh.attrib["file"] = spec.collision_rels[0]
            asset_parent = new_xml.find("asset")
            if asset_parent is not None:
                insert_pos = list(asset_parent).index(collision_mesh)
                for i, rel in enumerate(spec.collision_rels[1:], start=1):
                    clone = copy.deepcopy(collision_mesh)
                    clone.attrib["name"] = f"{spec.object_id}_collision_mesh_{i}"
                    clone.attrib["file"] = rel
                    insert_pos += 1
                    asset_parent.insert(insert_pos, clone)

        texture_elem = new_xml.find(".//asset/texture[@name='_name_texture']")
        material_elem = new_xml.find(".//asset/material[@name='_name_material']")
        asset_parent = new_xml.find("asset")
        visual_geom = new_xml.find(".//geom[@name='_name_visual_geom']")

        if spec.visual_texture_rel is not None and texture_elem is not None and material_elem is not None:
            texture_elem.attrib["name"] = f"{spec.object_id}_texture"
            texture_elem.attrib["file"] = spec.visual_texture_rel
            material_elem.attrib["name"] = f"{spec.object_id}_material"
            material_elem.attrib["texture"] = f"{spec.object_id}_texture"
            if visual_geom is not None:
                visual_geom.attrib["material"] = f"{spec.object_id}_material"
        else:
            if asset_parent is not None:
                if texture_elem is not None:
                    asset_parent.remove(texture_elem)
                if material_elem is not None:
                    asset_parent.remove(material_elem)
            if visual_geom is not None and "material" in visual_geom.attrib:
                del visual_geom.attrib["material"]

        body = new_xml.find(".//worldbody/body[@name='_name']")
        if body is not None:
            body.attrib["name"] = spec.object_id

        joint = new_xml.find(".//freejoint[@name='_name_joint']")
        if joint is not None:
            joint.attrib["name"] = f"{spec.object_id}_joint"

        if visual_geom is not None:
            visual_geom.attrib["name"] = f"{spec.object_id}_visual_geom"
            visual_geom.attrib["mesh"] = f"{spec.object_id}_visual_mesh"

        collision_geom = new_xml.find(".//geom[@name='_name_collision_geom']")
        if collision_geom is not None:
            collision_geom.attrib["name"] = f"{spec.object_id}_collision_geom_0"
            collision_geom.attrib["mesh"] = f"{spec.object_id}_collision_mesh_0"

        inertial = new_xml.find(".//inertial")
        if inertial is not None:
            inertial.attrib["mass"] = str(spec.mass_kg)
            inertial.attrib["diaginertia"] = (
                f"{float(spec.principal_moments[0])} {float(spec.principal_moments[1])} {float(spec.principal_moments[2])}"
            )
            inertial.attrib["pos"] = "0 0 0"
            inertial.attrib["euler"] = "0 0 0"

        if collision_geom is not None:
            body_parent = new_xml.find(".//worldbody/body")
            if body_parent is not None:
                insert_pos = list(body_parent).index(collision_geom)
                for i in range(1, len(spec.collision_rels)):
                    clone = copy.deepcopy(collision_geom)
                    clone.attrib["name"] = f"{spec.object_id}_collision_geom_{i}"
                    clone.attrib["mesh"] = f"{spec.object_id}_collision_mesh_{i}"
                    insert_pos += 1
                    body_parent.insert(insert_pos, clone)

        return new_xml

    @staticmethod
    def _save_xml(root: ET.Element, out_path: Path, force: bool) -> bool:
        if out_path.exists() and not force:
            return False
        out_path.write_bytes(ET.tostring(root))
        return True

    @staticmethod
    def _validate_mujoco_xml(xml_path: Path) -> None:
        try:
            mujoco.MjModel.from_xml_path(str(xml_path))
        except Exception as e:
            raise RuntimeError(f"{xml_path.name}: MuJoCo compile failed: {e}") from e

    def build(self, object_ids: set[str] | None, force: bool) -> dict:
        manifest_path = self.dataset_root / "manifest.process_meshes.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run process_meshes.py first."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        specs, skipped_non_success = self._collect_specs(
            manifest=manifest,
            selected_ids=object_ids,
        )

        built = 0
        skipped = 0

        for spec in specs:
            urdf_root = self._update_urdf(spec)
            xml_root = self._update_xml(spec)
            urdf_path = spec.obj_dir / f"{spec.object_id}.urdf"
            xml_path = spec.obj_dir / f"{spec.object_id}.xml"
            w1 = self._save_xml(urdf_root, urdf_path, force=force)
            w2 = self._save_xml(xml_root, xml_path, force=force)
            if w1 or w2:
                built += 1
            else:
                skipped += 1

        validated = 0
        for spec in specs:
            xml_path = spec.obj_dir / f"{spec.object_id}.xml"
            if not xml_path.exists():
                raise RuntimeError(f"{spec.object_id}: missing xml after build: {xml_path}")
            print(f"[MuJoCo validate] {spec.object_id}")
            self._validate_mujoco_xml(xml_path)
            validated += 1

        return {
            "dataset": self.dataset_root.name,
            "manifest_path": str(manifest_path),
            "num_candidates": len(specs),
            "num_built": built,
            "num_skipped": skipped,
            "num_failed": skipped_non_success,
            "num_validated_mujoco": validated,
        }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    processed_root = repo_root / "assets" / "objects" / "processed"

    dataset_choices = []
    if processed_root.exists():
        dataset_choices = sorted([p.name for p in processed_root.iterdir() if p.is_dir()])

    parser = argparse.ArgumentParser(description="Build URDF/MJCF for processed objects.")
    parser.add_argument("--dataset", type=str, required=True, choices=dataset_choices)
    parser.add_argument("--processed-root", type=Path, default=processed_root)
    parser.add_argument("--prototype-urdf", type=Path, default=repo_root / "src" / "_prototype.urdf")
    parser.add_argument("--prototype-xml", type=Path, default=repo_root / "src" / "_prototype.xml")
    parser.add_argument("--object-id", type=str, default=None, help="Comma-separated object ids to build.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing urdf/xml.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.processed_root.resolve() / args.dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    selected_ids = None
    if args.object_id:
        selected_ids = {x.strip() for x in args.object_id.split(",") if x.strip()}

    builder = ObjectDescriptionBuilder(
        dataset_root=dataset_root,
        urdf_prototype=args.prototype_urdf.resolve(),
        xml_prototype=args.prototype_xml.resolve(),
    )
    report = builder.build(
        object_ids=selected_ids,
        force=bool(args.force),
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
