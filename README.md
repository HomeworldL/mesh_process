# mesh_process

## Introduction

Toolkit for preparing object meshes for grasping/simulation pipelines. It supports:

- mesh alignment to COM + principal inertia axes,
- watertight mesh generation (CoACD remesh output),
- convex decomposition for collision meshes,
- mesh simplification (ACVD),
- URDF and MuJoCo XML (MJCF) export with inertial parameters.

## Unified Data Layout

All datasets use the same layout:

```bash
assets/objects/
├── raw/
│   ├── YCB/...
│   ├── RealDex/...
│   ├── GraspNet/...
│   ├── Objaverse/...
│   └── ShapeNet*/...
└── processed/
    ├── <dataset>/<object_id>/raw.obj (+ optional textures)
    └── <dataset>/manifest.json
```

## Dependencies

```bash
pip install trimesh objaverse gdown tqdm pymeshlab pillow
# optional: MuJoCo viewer script
pip install mujoco
```

Build third-party binaries first:

- `third_party/CoACD/build/main`
- `third_party/ACVD/bin/ACVD`

## Quick Visualization

Randomly preview one OBJ from a processed dataset:

```bash
python src/visualize_obj.py --dataset YCB
```

Optional:

```bash
python src/visualize_obj.py --dataset RealDex --seed 42
python src/visualize_obj.py --dataset YCB --mesh-type visual
python src/visualize_obj.py --dataset YCB --mesh-type inertia
```

Notes:
- `--mesh-type raw|inertia|visual` chooses `raw.obj` / `inertia.obj` / `visual.obj`.
- If `--object-id` is provided, it visualizes exactly that object folder.

MuJoCo MJCF visualization:

```bash
python src/visualize_obj_mujcoco.py --dataset YCB
python src/visualize_obj_mujcoco.py --dataset YCB --object-id YCB_001_chips_can
```

## Stage 1: Ingest (Download/Organize/Verify)

Use a single source-ingest entrypoint (`src/ingest_assets.py`), parallel to `process_meshes.py` and `build_object_descriptions.py`.

Currently completed and used adapters:
- `GraspNet`, `HOPE`, `KIT`, `MSO`, `Objaverse`, `RealDex`, `YCB`, `DexNet`

```bash
# YCB example
python src/ingest_assets.py download --source YCB
python src/ingest_assets.py organize --source YCB
python src/ingest_assets.py verify --source YCB --check-paths

# RealDex / GraspNet object archives (Google Drive)
python src/ingest_assets.py download --source RealDex
python src/ingest_assets.py organize --source RealDex

python src/ingest_assets.py download --source GraspNet
python src/ingest_assets.py organize --source GraspNet

# HOPE object folder (Google Drive)
python src/ingest_assets.py download --source HOPE
python src/ingest_assets.py organize --source HOPE

# KIT / DexNet
python src/ingest_assets.py download --source KIT
python src/ingest_assets.py organize --source KIT
python src/ingest_assets.py download --source DexNet
python src/ingest_assets.py organize --source DexNet

# Objaverse (use --subset to filter categories from category_annotation.json)
python src/ingest_assets.py download --source Objaverse --subset Daily-Used
python src/ingest_assets.py organize --source Objaverse
# For evaluation subsets: deterministic random sample of N objects
python src/ingest_assets.py organize --source Objaverse --sample-n 500
python src/ingest_assets.py organize --source Objaverse --sample-n 500 --sample-seed 0

# ShapeNet (license/login protected):
# 1) manual download, then place archive under assets/objects/raw/<dataset>/
#    or set env var SHAPENET_CORE_ARCHIVE / SHAPENET_SEM_ARCHIVE
python src/ingest_assets.py download --source ShapeNetCore
python src/ingest_assets.py organize --source ShapeNetCore
python src/ingest_assets.py download --source ShapeNetSem
python src/ingest_assets.py organize --source ShapeNetSem
```

Notes:
- Google Drive downloads use `gdown` (`pip install gdown`).
- Download stage uses `tqdm` progress bars (YCB stream download, archive extract, Objaverse batch mirror).
- `organize` will build `assets/objects/processed/<dataset>/manifest.json` automatically.
- `mass_kg` is included per object; when unknown, default value is `0.1`.
- Canonical organized file names are: `raw.obj`, optional `texture_map.png`, optional `textured.mtl`.
  - ShapeNetSem/ShapeNetCore default behavior is OBJ-only (`raw.obj` after normalization). Texture export is optional via env (`SHAPENET_EXPORT_TEXTURES=1` or source-specific switch).
- ShapeNetCore / ShapeNetSem usually require terms acceptance and authenticated/manual archive retrieval.

## Dataset-Specific Structures

### YCB

- Download cache/extract: `assets/objects/raw/YCB/*`
- Organized objects: `assets/objects/processed/YCB/YCB_<object_name>/raw.obj`
- Keep `textured.mtl` and `texture_map.png` when present.

### RealDex

- Downloaded archive: `assets/objects/raw/RealDex/RealDex-objmodels.zip`
- Extracted source: `assets/objects/raw/RealDex/RealDex-objmodels/models/*.obj`
- Organized output: `assets/objects/processed/RealDex/RealDex_<object_name>/raw.obj`
- RealDex object models are treated as OBJ-only (no texture assets).

### GraspNet

- Extracted source (current verified layout): `assets/objects/raw/GraspNet/models/models/<id>/`
- Per object folder contains files such as `textured.obj`, `textured.mtl`, `texture_map.png`.
- Organized output:
  - `assets/objects/processed/GraspNet/GraspNet_<id>/raw.obj`
  - keep only `textured.mtl` and `texture_map.png` next to `raw.obj`.

### Objaverse

- Mirrored downloads: `assets/objects/raw/Objaverse/objects/*.glb`
- Organized objects: `assets/objects/processed/Objaverse/Objaverse_<name>/raw.obj`
- Optional organize sampling: pass `--sample-n N` and optional `--sample-seed` (default `0`).
- Raw mirror may use symlinks to cache files; this is expected.

### HOPE

- Downloaded source (official folder mirror): `assets/objects/raw/HOPE/hope_objects/`
- Official layout per object: `<object_name>/google_16k/{textured.obj,textured.mtl,texture_map.png}`
- Organized objects: `assets/objects/processed/HOPE/HOPE_<object_name>/raw.obj`
- If textures exist, normalized outputs are `textured.mtl` + `texture_map.png`.

### KIT

- Download mode: crawl official list page and fetch each object's `meshes.zip` from `tmp.php` links.
- Raw download/extract layout:
  - `assets/objects/raw/KIT/archives/<id>_<object>.zip`
  - `assets/objects/raw/KIT/objects/<object_name>/...`
- Organized objects: `assets/objects/processed/KIT/KIT_<object_name>/raw.obj`
- If textures exist, normalized outputs are `textured.mtl` + `texture_map.png`.

### DexNet

- Default `download` uses Google Drive file id `1dwzHMGI_bqekBoqpFgDRQSXsDIZj--v8`
  (`https://drive.google.com/file/d/1dwzHMGI_bqekBoqpFgDRQSXsDIZj--v8/view`).
- Optional override priority remains:
  `DEXNET_SOURCE_DIR` > `DEXNET_ARCHIVE` > `DEXNET_URL` > default Google Drive source.
- Organized objects: `assets/objects/processed/DexNet/DexNet_<object_name>/raw.obj`
- If textures exist, normalized outputs are `textured.mtl` + `texture_map.png`.
- Adapter is integrated in the same Stage-1 flow as other completed datasets.

### ShapeNetCore / ShapeNetSem

- Requires manual/authenticated archive acquisition.
- Recommended flow: put archive under `assets/objects/raw/<dataset>/`, then run `download` + `organize`.
- Organized output is normalized under `assets/objects/processed/<dataset>/`.
- ShapeNetCore details:
  - `download`: selective extraction from `ShapeNetCore.v2.zip` using `filter_lists.py` (allowed synsets + denylist ids), only keeps `model_normalized.{obj,mtl,json}` and referenced `images/*`.
  - `organize`: load each OBJ then normalize geometry (AABB center -> origin, max extent -> `1.0`), drop abnormal meshes.
  - texture export is optional (default off): set `SHAPENET_CORE_EXPORT_TEXTURES=1` or `SHAPENET_EXPORT_TEXTURES=1`; exported textures are canonicalized to single `texture_map.png` + `textured.mtl`.
- ShapeNetSem details:
  - `download`: metadata + filtered `models-OBJ` (obj/mtl) + full `models-textures`.
  - `organize`: load each OBJ then normalize geometry (AABB center -> origin, max extent -> `1.0`), drop abnormal meshes.
  - texture export is optional (default off): set `SHAPENET_SEM_EXPORT_TEXTURES=1` or `SHAPENET_EXPORT_TEXTURES=1`; exported textures are canonicalized to single `texture_map.png` + `textured.mtl`.
  - abnormal-mesh rejection thresholds:
    - pre-normalization `max_extent <= 1e-9`, or pre-normalization aspect ratio (`max_extent / min_extent`) `> 1e5`
    - post-normalization `max_extent` not in `[0.95, 1.05]`
    - post-normalization AABB center norm `> 5e-3`

## Stage 2: Process Meshes (`process_meshes.py`)

Run Stage-2 on organized objects:

```bash
python src/process_meshes.py --dataset YCB --workers 8
```

Flow per object:
- `mesh_transform` (always runs): regenerate `inertia.obj` and inertial principal data.
  - principal-axis assignment now selects signed/permuted axes that remain close to original XYZ orientation while still aligning to principal directions.
  - for negative-signed inertia from flipped winding / negative signed volume, Stage-2 flips sign and enforces a positive inertia floor.
  - if eigenvalues are still non-positive, Stage-2 falls back to AABB-based diagonal inertia (box approximation), avoiding zero `diaginertia` in exported MJCF/URDF.
- `mesh_manifold_and_convex_decomp`: generate `manifold.obj` + `coacd.obj`.
- convex export: write `meshes/coacd_convex_piece_*.obj`.
- `mesh_simplify`: generate `simplified.obj` (skip if exists unless `--force`).
- `mesh_visual`: generate compressed visual assets (`visual.obj`, optional `visual.mtl`, optional `visual_texture_map.png`) with geometry decimation + OBJ text slimming + texture compression.
  - visual geometry is decimated from `inertia.obj` (fallback `raw.obj` if missing).
  - Textured objects are treated as canonical single-texture input (`textured.mtl + texture_map.png`) and produce one `visual_texture_map.png`.
  - before marking an object as success, Stage-2 validates that `visual.obj` is loadable as a non-empty scene with valid bounds.

Parallel and overwrite behavior:
- Default is step-level skip where applicable.
- `--force` recomputes all outputs.
- `--workers` enables object-level parallel processing.
- `--preview` requires `--workers 1`.

Main output report:
- `assets/objects/processed/<dataset>/manifest.process_meshes.json`
- Per object fields include `process_status`, `process_error`, `center_of_mass`, `principal_moments`, `principal_axes`, and visual asset paths.

## Stage 3: Build URDF / MJCF (`build_object_descriptions.py`)

Build simulation descriptions from Stage-2 outputs:

```bash
python src/build_object_descriptions.py --dataset YCB --force
```

Data source:
- `assets/objects/processed/<dataset>/manifest.process_meshes.json`
- mass: `mass_kg`
- inertia: `principal_moments`

Mesh usage:
- visual mesh: `visual.obj`
- collision meshes: `meshes/coacd_convex_piece_*.obj`

Outputs per object:
- `<object_id>.urdf`
- `<object_id>.xml`

## Stage 4: Multi-view Point Cloud Sampling (`sample_pointcloud_views.py`)

Sample partial point clouds from multiple camera viewpoints around each object:

```bash
python src/sample_pointcloud_views.py --dataset YCB --views 25 --points 4096
```

Input:
- Stage-3 object MJCF: `<object_id>.xml`
- Stage-2 process report: `manifest.process_meshes.json` (only `process_status=success` objects are sampled by default)

Output per object:
- `assets/objects/processed/<dataset>/<object_id>/vision_data/pc_views.npz`

Default data layout in `pc_views.npz`:
- `camera_intrinsic` (`fx, fy, cx, cy, width, height`)
- `camera_extrinsic` (`[V, 4, 4]`)
- `camera_position` (`[V, 3]`)
- `camera_lookat` (`[V, 3]`)
- `point_cloud` (`[V, N, 3]`)
- `valid_point_num` (`[V]`)
- `rgb_path` (reserved interface, currently empty)
