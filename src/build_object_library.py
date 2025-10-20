"""Build URDF for objects.
Modified from https://github.com/harvard-microrobotics/object2urdf
and https://github.com/elpis-lab/YCB_Dataset
"""

import os
from object_builder import ObjectBuilder

# Build entire libraries of URDFs
# This is only suitable for objects built with single obj/stl file
# Models such as robots or articulated objects will not work properly
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
object_folder = root_dir + "/assets/YCB/ycb_datasets"

builder = ObjectBuilder(
    object_folder,
    "src/_prototype.urdf",
    "src/_prototype.xml",
)
builder.build_library(
    # Build URDF and XML
    force_overwrite=True,
)
