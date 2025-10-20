"""Generate urdf for objects in a folder with a provided prototype urdf file.
Script from https://github.com/harvard-microrobotics/object2urdf
and https://github.com/elpis-lab/YCB_Dataset

Modified to use processed obj files.
Added mass, inertia to the urdf and mujoco xml files.
"""

from scipy.spatial.transform import Rotation
import numpy as np
import os
import copy
import trimesh
from pathlib import Path

import json
import xml.etree.ElementTree as ET
import re
from collections import OrderedDict


class ObjectBuilder:
    def __init__(
        self,
        object_folder="",
        urdf_prototype="_prototype.urdf",
        xml_prototype="_prototype.xml",
    ):
        self.object_folder = os.path.abspath(object_folder)

        self.urdf_base = self._read_xml(urdf_prototype)
        self.xml_base = self._read_xml(xml_prototype)

    def collect_dataset_index(self):
        """
        Walk self.object_folder and build an OrderedDict index for each object-folder.
        Assumes layout:
           <self.object_folder>/
               002_master_chef_can/
                   raw/ textured.obj, textured.mtl, texture_map.png
                   inertia.obj
                   manifold.obj
                   coacd.obj
                   meshes/ coacd_convex_piece_*.obj
                   simplified.obj
        Returns OrderedDict keyed by folder name (sorted by leading digits if present).
        """
        root = os.path.abspath(self.object_folder)
        entries = []
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isdir(p):
                entries.append(name)

        # sort by leading number if name begins with digits, else lexicographic
        def leading_num(s):
            m = re.match(r"^(\d+)", s)
            return int(m.group(1)) if m else float("inf")

        entries.sort(key=lambda s: (leading_num(s), s.lower()))

        index = OrderedDict()
        for nm in entries:
            folder = os.path.join(root, nm)

            # helper to build abs/rel
            def abs_if_exists(*parts):
                pth = os.path.join(folder, *parts)
                return os.path.abspath(pth) if os.path.exists(pth) else None

            def rel_if_exists(abs_path):
                if abs_path is None:
                    return None
                return os.path.relpath(abs_path, folder).replace("\\", "/")

            # prefer files placed directly under folder or inside 'raw' subfolder
            inertia_abs = abs_if_exists("inertia.obj")
            manifold_abs = abs_if_exists("manifold.obj")
            coacd_abs = abs_if_exists("coacd.obj")
            simplified_abs = abs_if_exists("simplified.obj")
            # raw: may be in 'raw/textured.obj' or top-level 'textured.obj' or 'raw.obj'
            raw_abs = (
                abs_if_exists("raw", "textured.obj")
                or abs_if_exists("textured.obj")
                or abs_if_exists("raw.obj")
            )
            mtl_abs = abs_if_exists("textured.mtl") or abs_if_exists(
                "raw", "textured.mtl"
            )
            # texture file heuristics: look for common image names under folder or raw/
            texture_abs = None
            for candidate in [
                "texture_map.png",
                "texture.png",
                "diffuse.png",
                "albedo.png",
            ]:
                candidate_abs = abs_if_exists("raw", candidate) or abs_if_exists(
                    candidate
                )
                if candidate_abs:
                    texture_abs = candidate_abs
                    break

            # collect convex mesh pieces from common subfolders
            meshes_dir_candidates = [
                os.path.join(folder, "meshes"),
                os.path.join(folder, "convex_parts"),
                os.path.join(folder, "parts"),
            ]
            mesh_piece_abs = []
            for md in meshes_dir_candidates:
                if os.path.isdir(md):
                    # collect .obj files sorted
                    files = sorted(
                        [f for f in os.listdir(md) if f.lower().endswith(".obj")]
                    )
                    for f in files:
                        full = os.path.join(md, f)
                        mesh_piece_abs.append(os.path.abspath(full))
                    if mesh_piece_abs:
                        break

            # metadata
            metadata_abs = abs_if_exists("metadata.json")

            # build dict for this object
            obj_entry = {
                "name": nm,
                "folder_abs": os.path.abspath(folder),
                "inertia_abs": inertia_abs,
                "inertia_rel": rel_if_exists(inertia_abs),
                "manifold_abs": manifold_abs,
                "manifold_rel": rel_if_exists(manifold_abs),
                "coacd_abs": coacd_abs,
                "coacd_rel": rel_if_exists(coacd_abs),
                "simplified_abs": simplified_abs,
                "simplified_rel": rel_if_exists(simplified_abs),
                "raw_abs": raw_abs,
                "raw_rel": rel_if_exists(raw_abs),
                "mtl_abs": mtl_abs,
                "mtl_rel": rel_if_exists(mtl_abs),
                "texture_abs": texture_abs,
                "texture_rel": rel_if_exists(texture_abs),
                "meshes": {
                    "abs": mesh_piece_abs,
                    "rel": [
                        os.path.relpath(p, folder).replace("\\", "/")
                        for p in mesh_piece_abs
                    ],
                },
                "metadata_abs": metadata_abs,
            }
            index[nm] = obj_entry

        return index

    # Read and parse a URDF from a file
    def _read_xml(self, filename):
        root = ET.parse(filename).getroot()
        return root

    # Convert a list to a space-separated string
    def _list2str(self, in_list):
        out = ""
        for el in in_list:
            out += str(el) + " "
        return out[:-1]

    # Convert a space-separated string to a list
    def _str2list(self, in_str):
        out = in_str.split(" ")
        out = [float(el) for el in out]
        return out

    def save_to_obj(self, filename):
        name, ext = os.path.splitext(filename)
        obj_filename = name + ".obj"
        mesh = trimesh.load(filename)
        mesh.export(obj_filename)
        return obj_filename

    # Replace an attribute in a feild of a URDF
    def replace_urdf_attribute(self, urdf, feild, attribute, value):
        urdf = self.replace_urdf_attributes(urdf, feild, {attribute: value})
        return urdf

    # Replace several attributes in a feild of a URDF
    def replace_urdf_attributes(self, urdf, feild, attribute_dict, sub_feild=None):

        if sub_feild is None:
            sub_feild = []

        field_obj = urdf.find(feild)

        if field_obj is not None:
            if len(sub_feild) > 0:
                for child in reversed(sub_feild):
                    field_obj = ET.SubElement(field_obj, child)
            field_obj.attrib.update(attribute_dict)
            # field_obj.attrib = attribute_dict
        else:
            feilds = feild.split("/")
            new_feild = "/".join(feilds[0:-1])
            sub_feild.append(feilds[-1])
            self.replace_urdf_attributes(urdf, new_feild, attribute_dict, sub_feild)

    # Make an updated copy of the URDF for the current object
    def update_urdf(
        self,
        object_file,
        object_name,
        collision_files,
        mass_value=None,
        principal_inertia=None,
    ):
        """
        Create a URDF element for an object whose geometry has already been
        transformed to COM + principal-inertia frame (so visual/collision origins = 0).
        - object_file: path (relative) to visual mesh (ycb_inertia version)
        - collision_files: list of mesh paths for collision (from manifold/coacd)
        - mass_value: mass to write into <inertial><mass value="..."/>
        - principal_inertia: iterable of three principal moments (ixx, iyy, izz)
                             expressed about the COM in the mesh frame.
        """
        new_urdf = copy.deepcopy(self.urdf_base)
        new_urdf.attrib["name"] = object_name

        # Visual mesh (visual uses ycb_inertia visual obj)
        self.replace_urdf_attribute(
            new_urdf, ".//visual/geometry/mesh", "filename", object_file
        )

        # Update mass value if provided
        if mass_value is not None:
            self.replace_urdf_attribute(
                new_urdf, ".//inertial/mass", "value", str(mass_value)
            )

        # Because meshes are already in COM/principal frame, visual and collision
        # origins should be identity (0 0 0 position and 0 0 0 rpy).
        # Overwrite origins to zeros if those elements exist (be tolerant if not).
        self.replace_urdf_attributes(
            new_urdf,
            ".//visual/origin",
            {"xyz": "0 0 0", "rpy": "0 0 0"},
        )
        self.replace_urdf_attributes(
            new_urdf,
            ".//collision/origin",
            {"xyz": "0 0 0", "rpy": "0 0 0"},
        )

        # If principal inertia is supplied, write diagonal inertia and zero off-diagonals.
        # URDF inertia order: ixx ixy ixz iyy iyz izz
        if principal_inertia is not None:
            # ensure length-3 array
            pi = np.asarray(principal_inertia, dtype=float).flatten()
            if pi.size != 3:
                raise ValueError("principal_inertia must be length-3 (ixx, iyy, izz)")
            ixx, iyy, izz = [float(np.round(x, 12)) for x in pi]
            inertia_attrs = {
                "ixx": f"{ixx:.12g}",
                "ixy": "0.0",
                "ixz": "0.0",
                "iyy": f"{iyy:.12g}",
                "iyz": "0.0",
                "izz": f"{izz:.12g}",
            }

            # Replace or create inertial/inertia element
            self.replace_urdf_attributes(new_urdf, ".//inertial/inertia", inertia_attrs)

            # Ensure inertial/origin is identity (no offset/rotation)
            self.replace_urdf_attributes(
                new_urdf, ".//inertial/origin", {"xyz": "0 0 0", "rpy": "0 0 0"}
            )

        # Update the first collision mesh filename
        self.replace_urdf_attribute(
            new_urdf,
            ".//collision/geometry/mesh",
            "filename",
            collision_files[0],
        )

        # Duplicate additional collision entries if there are multiple collision parts
        col = new_urdf.find(".//collision")
        parent = new_urdf.find(".//link")
        for collision_file in collision_files[1:]:
            clone = copy.deepcopy(col)
            clone.find("geometry/mesh").attrib["filename"] = collision_file
            parent.insert(list(parent).index(col) + 1, clone)
            col = clone

        return new_urdf

    # Make an updated copy of the Mujoco XML for the current object
    def update_xml(
        self,
        object_file,
        object_name,
        collision_files,
        mass_value=None,
        principal_inertia=None,
        vis_collision=False,
    ):
        # Update the filenames and object name
        new_xml = copy.deepcopy(self.xml_base)
        new_xml.attrib["model"] = object_name

        # Update mesh file names in assets
        visual_mesh = new_xml.find(".//asset/mesh[@name='_name_visual_mesh']")
        if visual_mesh is not None:
            visual_mesh.attrib["name"] = f"{object_name}_visual_mesh"
            visual_mesh.attrib["file"] = object_file

        collision_mesh = new_xml.find(".//asset/mesh[@name='_name_collision_mesh']")
        if collision_mesh is not None:
            collision_mesh.attrib["name"] = f"{object_name}_collision_mesh_0"
            collision_mesh.attrib["file"] = collision_files[0]
            # Copy collision sections and replace mesh files one by one
            parent = new_xml.find("asset")
            for i, collision_file in enumerate(collision_files[1:]):
                clone = copy.deepcopy(collision_mesh)
                clone.attrib["name"] = f"{object_name}_collision_mesh_{i + 1}"
                clone.attrib["file"] = collision_file
                parent.insert(list(parent).index(collision_mesh) + i + 1, clone)

        # Update texture and material names
        texture = new_xml.find(".//asset/texture[@name='_name_texture']")
        if texture is not None:
            texture.attrib["name"] = f"{object_name}_texture"
            # Assume texture file is the same name in _prototype.xml
            texture_file = object_file.replace(
                os.path.basename(object_file), texture.attrib["file"]
            )
            texture.attrib["file"] = texture_file

        material = new_xml.find(".//asset/material[@name='_name_material']")
        if material is not None:
            material.attrib["name"] = f"{object_name}_material"
            material.attrib["texture"] = f"{object_name}_texture"

        # Update body name
        body = new_xml.find(".//worldbody/body[@name='_name']")
        if body is not None:
            body.attrib["name"] = object_name

        # Update joint name
        joint = new_xml.find(".//freejoint[@name='_name_joint']")
        if joint is not None:
            joint.attrib["name"] = f"{object_name}_joint"

        # Update geom names and mesh references
        
        visual_geom = new_xml.find(".//geom[@name='_name_visual_geom']")
        if visual_geom is not None:
            if not vis_collision:
                visual_geom.attrib["name"] = f"{object_name}_visual_geom"
                visual_geom.attrib["mesh"] = f"{object_name}_visual_mesh"
                visual_geom.attrib["material"] = f"{object_name}_material"
            else:
                # delete visual geom
                parent = new_xml.find(".//worldbody/body")
                parent.remove(visual_geom)

        collision_geom = new_xml.find(".//geom[@name='_name_collision_geom']")
        if collision_geom is not None:
            collision_geom.attrib["name"] = f"{object_name}_collision_geom_0"
            collision_geom.attrib["mesh"] = f"{object_name}_collision_mesh_0"
            if vis_collision:
                collision_geom.attrib["contype"] = "0"
                collision_geom.attrib["conaffinity"] = "0"
                collision_geom.attrib["group"] = "0"

        # Update mass if provided
        inertial = new_xml.find(".//inertial")
        if inertial is not None and mass_value is not None:
            inertial.attrib["mass"] = str(mass_value)
        if inertial is not None and principal_inertia is not None:
            inertial.attrib["diaginertia"] = (
                f"{float(principal_inertia[0])} {float(principal_inertia[1])} {float(principal_inertia[2])}"
            )
            inertial.attrib["pos"] = "0 0 0"
            inertial.attrib["euler"] = "0 0 0"

        # Update the collision mesh files
        # Copy collision sections and replace mesh files one by one
        parent = new_xml.find(".//worldbody/body")
        for i, collision_file in enumerate(collision_files[1:]):
            clone = copy.deepcopy(collision_geom)
            clone.attrib["name"] = f"{object_name}_collision_geom_{i + 1}"
            clone.attrib["mesh"] = f"{object_name}_collision_mesh_{i + 1}"
            parent.insert(list(parent).index(collision_geom) + i + 1, clone)

        return new_xml

    # Save a file to a file
    def save_file(self, new_file, out_file, overwrite=False):

        # Do not overwrite the file unless the option is True
        if os.path.exists(out_file) and not overwrite:
            return

        # Save the file
        mydata = ET.tostring(new_file)
        with open(out_file, "wb") as f:
            f.write(mydata)

    # Build a URDF from an object file
    def build_description(
        self,
        info,
        output_folder=None,
        force_overwrite=False,
    ):
        nm = info["name"]
        metadata_abs = info["metadata_abs"]
        with open(metadata_abs, 'r') as f:
            metadata_content = f.read()  # 这才是JSON字符串
            metadata = json.loads(metadata_content)
        mass_value = metadata[nm]["mass_used"]
        principal_inertia = metadata[nm]["principal_moments"]

        visual_file = info["inertia_rel"]
        collision_files = info["meshes"]["rel"]

        # # If no output folder is specified, use the base object folder
        if output_folder is None:
            output_folder = os.path.dirname(info["inertia_abs"])

        urdf_out = self.update_urdf(
            visual_file,
            nm,
            collision_files=collision_files,
            mass_value=mass_value,
            principal_inertia=principal_inertia,
        )
        out_file = os.path.join(output_folder, nm + ".urdf")
        self.save_file(urdf_out, out_file, force_overwrite)
        xml_out = self.update_xml(
            visual_file,
            nm,
            collision_files=collision_files,
            mass_value=mass_value,
            principal_inertia=principal_inertia,
            vis_collision=False,
        )
        out_file = os.path.join(output_folder, nm + ".xml")
        self.save_file(xml_out, out_file, force_overwrite)

    # Build the URDFs for all objects in your library.
    def build_library(self, **kwargs):
        """
        Build URDF / MJCF for each object folder under self.object_folder.
        kwargs forwarded to build_description (e.g., force_overwrite=True).
        """
        print("\nFOLDER: %s" % (self.object_folder))

        # we get one representative file per object folder
        obj_index = self.collect_dataset_index()

        print(f"Found {len(obj_index)} objects (folders) under {self.object_folder}")

        for name, info in obj_index.items():
            print(f"Building: {name}")
            try:
                self.build_description(info, **kwargs)
            except Exception as e:
                print(f"  [Error building {name}]: {e}")

        print("Done building library.")

