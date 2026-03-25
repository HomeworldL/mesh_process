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
python src/visualize_obj.py --dataset YCB --mesh-type manifold
```

Notes:
- `--mesh-type raw|manifold|visual` chooses `raw.obj` / `manifold.obj` / `visual.obj`.
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

# Objaverse (download defaults to the fixed Daily-Used subset)
python src/ingest_assets.py download --source Objaverse
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
- Except `DDG`, stage-1 `organize` exports canonical assets through `trimesh`: `raw.obj`, plus optional `textured.png` and `textured.mtl`.
- `organize` reports now include `texture stats: textured=..., untextured=...`, computed from the final `manifest.json`.
- `mass_kg` is included per object; when unknown, default mass depends on dataset normalization policy:
  - normalized object datasets (`ShapeNetCore`, `ShapeNetSem`, `DGN`, `DDG`, `Objaverse`): `50.0 kg`
  - non-normalized datasets (`YCB`, `RealDex`, `GraspNet`, `HOPE`, `KIT`, `MSO`, `DexNet`): `0.1 kg`
- Canonical organized file names are: `raw.obj`, optional `textured.png`, optional `textured.mtl`.
  - ShapeNetSem/ShapeNetCore default behavior is OBJ-only (`raw.obj` after normalization). Texture export is optional via env (`SHAPENET_EXPORT_TEXTURES=1` or source-specific switch).
- ShapeNetCore / ShapeNetSem usually require terms acceptance and authenticated/manual archive retrieval.

## Dataset-Specific Structures

### YCB

- Download cache/extract: `assets/objects/raw/YCB/*`
- Organized objects: `assets/objects/processed/YCB/YCB_<object_name>/raw.obj`
- Source preference: `google_16k > poisson > any child containing textured.obj`.
- Only `google_16k` is allowed to export texture sidecars; fallback branches stay geometry-only.

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
- Only `textured.obj` is accepted; if missing, the whole object is skipped.
- Texture export requires both `trimesh` texture detection and source `texture_map.png`.

### Objaverse

- Mirrored downloads: `assets/objects/raw/Objaverse/objects/*.glb`
- Organized objects: `assets/objects/processed/Objaverse/Objaverse_<name>/raw.obj`
- Optional organize sampling: pass `--sample-n N` and optional `--sample-seed` (default `0`).
- Raw mirror may use symlinks to cache files; this is expected.
- Organize now normalizes geometry like DGN: AABB center -> origin, max extent -> `1.0`.
- Texture export is decided by `trimesh` texture detection on the source `glb/gltf`.
- Default fallback mass is `50.0 kg`.

### HOPE

- Downloaded source (official folder mirror): `assets/objects/raw/HOPE/hope_objects/`
- Official layout per object: `<object_name>/google_16k/{textured.obj,textured.mtl,texture_map.png}`
- Organized objects: `assets/objects/processed/HOPE/HOPE_<object_name>/raw.obj`
- Organize rescales source meshes from centimeters to meters (`0.01`).
- Texture export requires both `trimesh` texture detection and source `texture_map.png`.

### KIT

- Download mode: crawl official list page and fetch each object's `meshes.zip` from `tmp.php` links.
- Raw download/extract layout:
  - `assets/objects/raw/KIT/archives/<id>_<object>.zip`
  - `assets/objects/raw/KIT/objects/<object_name>/...`
- Organized objects: `assets/objects/processed/KIT/KIT_<object_name>/raw.obj`
- Source preference: `25k_tex > 5k_tex > Orig_tex > 800_tex > 25k > 5k > Orig > 800`.
- Organize rescales source meshes from millimeters to meters (`0.001`).
- Texture export requires both `trimesh` texture detection and a same-stem source PNG.

### MSO

- Raw source: sparse checkout of `mujoco_scanned_objects/models`.
- Organized objects: `assets/objects/processed/MSO/MSO_<object_name>/raw.obj`
- If OBJ references a missing MTL but source `texture.png` exists, organize synthesizes a minimal MTL before loading.
- Texture export requires both `trimesh` texture detection and source `texture.png`.

### DexNet

- Default `download` uses Google Drive file id `1dwzHMGI_bqekBoqpFgDRQSXsDIZj--v8`
  (`https://drive.google.com/file/d/1dwzHMGI_bqekBoqpFgDRQSXsDIZj--v8/view`).
- Optional override priority remains:
  `DEXNET_SOURCE_DIR` > `DEXNET_ARCHIVE` > `DEXNET_URL` > default Google Drive source.
- Organized objects: `assets/objects/processed/DexNet/DexNet_<object_name>/raw.obj`
- If textures are exported, canonical outputs are `textured.mtl` + `textured.png`.
- Adapter is integrated in the same Stage-1 flow as other completed datasets.

### DGN / DDG

- Source archive is organized by `DGNAdapter`; output includes:
  - full set: `assets/objects/processed/DGN/`
  - derived subset (`ddg*` prefix): `assets/objects/processed/DDG/`
- `DGN` normalizes OBJ geometry: AABB center -> origin, max extent -> `1.0`, then exports OBJ-only via `trimesh`.
- `DDG` is the only organize path that does not re-export with `trimesh`; it copies processed `ddg*` object folders from `processed/DGN` and rebuilds manifests.
- These datasets are treated as normalized object models for default mass policy (`50.0 kg` when source mass is unknown).

### ShapeNetCore / ShapeNetSem

- Requires manual/authenticated archive acquisition.
- Recommended flow: put archive under `assets/objects/raw/<dataset>/`, then run `download` + `organize`.
- Organized output is normalized under `assets/objects/processed/<dataset>/`.
- These datasets are treated as normalized object models for default mass policy (`50.0 kg` when source mass is unknown).
- ShapeNetCore details:
  - `download`: selective extraction from `ShapeNetCore.v2.zip` using `filter_lists.py` (allowed synsets + denylist ids), only keeps `model_normalized.{obj,mtl,json}` and referenced `images/*`.
  - `organize`: load each OBJ then normalize geometry (AABB center -> origin, max extent -> `1.0`), drop abnormal meshes.
  - texture export is optional (default off): set `SHAPENET_CORE_EXPORT_TEXTURES=1` or `SHAPENET_EXPORT_TEXTURES=1`; exported textures are canonicalized to single `textured.png` + `textured.mtl`.
- ShapeNetSem details:
  - `download`: metadata + filtered `models-OBJ` (obj/mtl) + full `models-textures`.
  - `organize`: load each OBJ then normalize geometry (AABB center -> origin, max extent -> `1.0`), drop abnormal meshes.
  - texture export is optional (default off): set `SHAPENET_SEM_EXPORT_TEXTURES=1` or `SHAPENET_EXPORT_TEXTURES=1`; exported textures are canonicalized to single `textured.png` + `textured.mtl`.
  - abnormal-mesh rejection thresholds:
    - pre-normalization `max_extent <= 1e-9`, or pre-normalization aspect ratio (`max_extent / min_extent`) `> 1e5`
    - post-normalization `max_extent` not in `[0.95, 1.05]`
    - post-normalization AABB center norm `> 5e-3`

## Stage 2: Process Meshes (`process_meshes.py`)

Refactor note: see `docs/stage2_manifold_first_refactor.md` for this manifold-first redesign and compatibility details.

Run Stage-2 on organized objects:

```bash
python src/process_meshes.py --dataset YCB --workers 8
```

Flow per object:
- `mesh_manifold_and_convex_decomp`: run CoACD once from `raw.obj` to generate `manifold.obj` + `coacd.obj`.
  - object fails immediately if CoACD fails.
  - Stage-2 requires `manifold.obj` to satisfy `is_watertight=True` and `is_volume=True`; otherwise object fails.
- principal-frame inertia step runs on `manifold.obj` (not on `raw.obj`):
  - inertia computation now directly uses trimesh built-ins: `moment_inertia`, `moment_inertia_frame`, `principal_inertia_components`, `principal_inertia_transform`.
  - principal-axis assignment selects signed/permuted axes that remain close to original XYZ orientation while still aligning to principal directions.
  - transformed manifold overwrites `manifold.obj` in the principal frame.
- raw mesh is transformed by the same principal-frame transform and directly compressed to visual assets (`visual.obj`, optional `textured_visual.mtl`, optional `textured_visual.png`); `inertia.obj` is no longer generated.
  - Textured objects are treated as canonical single-texture input (`textured.mtl + textured.png`) and produce one `textured_visual.png`.
  - before marking an object as success, Stage-2 validates that `visual.obj` is loadable as a non-empty scene with valid bounds.
- `coacd.obj` is transformed by the same principal-frame transform, then convex export writes `meshes/coacd_convex_piece_*.obj`.
  - invalid CoACD parts are skipped during export if empty, if bbox max extent is below threshold, or if absolute volume is below threshold.
- `mesh_simplify`: generate `simplified.obj` from transformed `manifold.obj` (skip if exists unless `--force`).

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
