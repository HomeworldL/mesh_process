# mesh_process

## Introduction
A small toolkit to preprocess YCB-style textured OBJ models and generate URDF / MuJoCo XML (MJCF) from the processed outputs, where the object coordinate system is the primary inertial frame.

Modified from:
* [MeshProcess](https://github.com/JYChen18/MeshProcess)
* [YCB_Dataset](https://github.com/elpis-lab/YCB_Dataset)

Some related works:
* [obj2mjcf](https://github.com/kevinzakka/obj2mjcf)
* [object2urdf](https://github.com/harvard-microrobotics/object2urdf)
* [YCB_sim](https://github.com/vikashplus/YCB_sim)
* [MJCF-Component](https://github.com/AvalonGuo/MJCF-Component)

### Features
* The object coordinate system coincides with the principal inertial system.

* Batch pipeline: copy raw .obj/.mtl/texture → transform to COM & principal axes → watertight (CoACD) → convex parts (CoACD) → simplify (ACVD).

* Export URDF and MuJoCo XML using inertia.obj (visual) and convex parts (collision).

* Metadata (mass, COM, principal moments/axes) saved per object.

### Quick layout

``` bash
assets/YCB/ycb_datasets/
└── 002_master_chef_can/
    ├── raw/               # source textured.obj, textured.mtl, textures...
    ├── inertia.obj        # mesh in COM + principal-axes frame
    ├── manifold.obj       # watertight output from CoACD
    ├── coacd.obj          # coacd main output
    ├── meshes/            # convex pieces (coacd_convex_piece_*.obj)
    ├── simplified.obj     # ACVD simplified mesh
    └── metadata.json
```

## Getting Started

### Dependencies

``` bash
pip install trimesh
```

Build the third-party package following their [installation guide](https://github.com/JYChen18/MeshProcess)


### Run

``` bash
# from repository root
python src/proc.py \
  --src assets/YCB/ycb \
  --dst assets/YCB/ycb_datasets \
  --acvd-vertnum 2000 \
  --acvd-gradation 1.5
```

* --skip-existing — skip outputs that already exist

* --force — overwrite outputs

* --preview — show trimesh preview for the inertia-transformed mesh (requires GUI)

* --coacd-quiet / --acvd-quiet — suppress external tool stdout

* --mass <json> — optional JSON map of object masses for inertia computation