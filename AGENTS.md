# Repository Guidelines

## Project Structure & Module Organization

- `src/`: main code.
  - `ingest_assets.py`: Stage-1 CLI (`download/organize/verify`) for object sources.
  - `visualize_obj.py`: utility to preview one OBJ from `processed/<source>` (random by default, or specify object id, supports raw/inertia/visual).
  - `visualize_obj_mujcoco.py`: utility to preview one built MJCF (`.xml`) in MuJoCo viewer.
  - `asset_ingest/`: source adapters and manifest schema.
    - `base.py`: abstract adapter interface.
    - `manifest.py`: manifest dataclass + validation.
    - source adapters: `ycb.py`, `realdex.py`, `graspnet.py`, `hope.py`, `kit.py`, `mso.py`, `objaverse.py`, etc.
  - `process_meshes.py`: Stage-2 mesh processing (`raw -> inertia -> manifold/coacd -> simplified -> visual`).
  - `build_object_descriptions.py`: Stage-3 URDF/MJCF generation.
  - `download_*.py`, `organize_*.py`: legacy or source-specific helper scripts.
  - `utils/`: reusable utilities (`util_file.py` etc.).
- `assets/objects/`:
  - `raw/<source>/`: source downloads/extracted original files (no separate `download/` root).
  - `processed/<source>/<name>/`: organized object assets (`raw.obj`, optional textures) + `manifest.json`.
- `third_party/`: external tools (`CoACD`, `ACVD`).

## Adapter Status (Current)

- Fully implemented and used now:
  - `YCB`, `RealDex`, `GraspNet`, `HOPE`, `KIT`, `MSO`, `Objaverse`, `DexNet`.
- Present but still evolving / not primary:
  - `ShapeNetCore`, `ShapeNetSem`.
  - current ShapeNetCore/ShapeNetSem flow: selective extraction + per-object OBJ normalization (center-to-origin + max-extent-to-1.0) before organize output.
  - abnormal-mesh drop thresholds in normalize step:
    - pre `max_extent <= 1e-9` or pre aspect ratio (`max_extent/min_extent`) `> 1e5`
    - post `max_extent` not in `[0.95, 1.05]` or post center norm `> 5e-3`

## Build, Test, and Development Commands

- Install dependencies: `pip install trimesh objaverse gdown tqdm pymeshlab pillow`
- Stage-1 ingest:
  - `python src/ingest_assets.py download --source YCB`
  - `python src/ingest_assets.py organize --source YCB`
  - `python src/ingest_assets.py verify --source YCB --check-paths`
  - `python src/ingest_assets.py download --source RealDex`
  - `python src/ingest_assets.py download --source GraspNet`
  - `python src/ingest_assets.py download --source HOPE`
  - `python src/ingest_assets.py download --source KIT`
  - `python src/ingest_assets.py download --source MSO`
  - `python src/ingest_assets.py download --source Objaverse --subset Daily-Used`
  - `python src/ingest_assets.py organize --source Objaverse --sample-n 500 --sample-seed 0`
  - `python src/ingest_assets.py download --source DexNet`
  - `python src/ingest_assets.py organize --source DexNet`
- Stage-2 process: `python src/process_meshes.py --dataset YCB --workers 8`
- Stage-3 export: `python src/build_object_descriptions.py --dataset YCB --force`
- Quick mesh preview: `python src/visualize_obj.py --dataset YCB`
  - mesh selection: `--mesh-type raw|inertia|visual`
Build `third_party/CoACD` and `third_party/ACVD` binaries before running `process_meshes.py`.

## Stage-2 Process Flow (`process_meshes.py`)

- Input: `assets/objects/processed/<dataset>/<object_id>/raw.obj`
- Per-object steps:
  - `mesh_transform` (always run) -> `inertia.obj` + principal inertia info
    - principal axes are assigned with signed/permuted matching so new XYZ remains as close as possible to original XYZ while still aligned to principal directions
    - when raw mesh orientation is negative (negative signed volume / negative raw inertia trace), Stage-2 flips face winding and normals while writing `inertia.obj`
  - `mesh_manifold_and_convex_decomp` -> `manifold.obj`, `coacd.obj`
  - convex export -> `meshes/coacd_convex_piece_*.obj`
  - `mesh_simplify` -> `simplified.obj` (default skip when exists)
  - `mesh_visual` -> `visual.obj` + optional visual material/texture compression outputs
    - textured input is canonical single texture (`textured.mtl + texture_map.png`) and outputs single `visual_texture_map.png`
    - visual geometry is decimated from `inertia.obj` (fallback to `raw.obj` only if `inertia.obj` is missing)
    - object is marked success only if generated `visual.obj` can be loaded as a non-empty scene with valid bounds
  - inertia robustness:
    - if trimesh returns negative-signed inertia (e.g., flipped winding / negative signed volume), Stage-2 flips sign and enforces a positive floor
    - if inertia eigenvalues are still non-positive, fallback to AABB-based diagonal inertia (box approximation) to avoid zero `diaginertia` in MJCF/URDF
- Default behavior:
  - step-level skip where outputs already exist
  - `--force` recomputes all steps
  - `--workers` enables object-level parallel processing
  - `--preview` only with `--workers 1`
- Dataset report: `assets/objects/processed/<dataset>/manifest.process_meshes.json`

## Stage-3 Description Build (`build_object_descriptions.py`)

- Input: `assets/objects/processed/<dataset>/manifest.process_meshes.json`
- Uses per-object:
  - `mass_kg` from manifest
  - `principal_moments` from manifest
  - visual mesh: `visual.obj`
  - collision meshes: `meshes/coacd_convex_piece_*.obj`
- Output per object:
  - `<object_id>.urdf`
  - `<object_id>.xml`

## Stage-4 Viewpoint Point Cloud Sampling (`sample_pointcloud_views.py`)

- Goal:
  - sample object-centric partial point clouds from multiple camera views for each processed object
  - RGB interface should be reserved, but RGB data is optional in current implementation
- Input:
  - Stage-3 MJCF: `assets/objects/processed/<dataset>/<object_id>/<object_id>.xml`
  - Stage-2 visual mesh and collision assets already prepared
- Default sampling:
  - `25` camera viewpoints per object
  - viewpoints sampled around object center (spherical/circular strategy with small noise)
  - each view generates one partial point cloud from rendered depth
- Output format (one file per object):
  - `assets/objects/processed/<dataset>/<object_id>/vision_data/pc_views.npz`
  - suggested keys:
    - `object_id` (string)
    - `camera_intrinsic` (`fx, fy, cx, cy, width, height`)
    - `camera_extrinsic` (`[25, 4, 4]`)
    - `camera_position` (`[25, 3]`)
    - `point_cloud` (`[25, N, 3]`, fixed `N` points per view after downsample)
    - `valid_point_num` (`[25]`)
    - `rgb_path` (reserved interface, optional/empty in current stage)
- Notes:
  - fixed `N` per view makes downstream dataloaders and batching easier than variable-length arrays
  - keep Stage-4 independent from Stage-2/3 generation logic (no overwrite on existing meshes/descriptions)
  - default command:
    - `python src/sample_pointcloud_views.py --dataset YCB --views 25 --points 4096`

## Coding Style & Naming Conventions

- Python 3, 4-space indentation, `snake_case` for functions/files, `PascalCase` for classes.
- Keep scripts CLI-driven with `argparse`; avoid hardcoded machine-specific paths.
- Preserve dataset-agnostic path conventions under `assets/objects/{raw,processed}`.
- Manifest mass policy: if source mass is unknown, write `mass_kg=0.1` in manifest.
- Manifest texture policy: `has_texture` is determined only by whether `.png` texture files exist in the organized object folder.
- Organized texture files must be `.png`; if source texture is non-png (e.g. `.jpg`), convert to `.png` during organize and do not keep the original non-png copy.
- Organized asset canonical naming:
  - mesh: `raw.obj`
  - texture: `texture_map.png`
  - material: `textured.mtl`
  - `textured.mtl` should reference `texture_map.png` (`map_Kd texture_map.png`).
  - ShapeNetSem/ShapeNetCore default organize mode is OBJ-only; optional texture export can be enabled by env and is then canonicalized to single `texture_map.png`.
- Organized object id naming:
  - default convention is `<SourceName>_<object_name>` (sanitize via `sanitize_object_id`).
  - handle collisions by suffix (`_2`, `_3`, ...).

## Common Adapter Patterns (Keep Consistent)

- Implement all 3 methods: `download`, `organize`, `build_manifest`.
- `organize` must call `self.write_manifest_for_organize(cfg, report)` at end.
- Use `canonicalize_texture_assets(...)` during organize if texture/mtl may exist.
- `manifest` path fields must use repo-relative paths via `relative_to_repo(...)`.
- `has_texture` must follow `.png`-only policy.
- If source mass is unavailable, always use `DEFAULT_MASS_KG` (`0.1`).
- Prefer `tqdm` for long loops (download and organize).

## Common Issues & References

- Objaverse raw files may appear as symlinks under `assets/objects/raw/Objaverse/objects/`; this is expected (`symlink` preferred, copy fallback).
- Objaverse organize supports deterministic sampling for eval:
  - `--sample-n N --sample-seed S` (default seed `0`).
- HOPE Google Drive can fail per-file when downloading whole folder recursively.
  - Current adapter lists files first, then downloads only required 3 files (`textured.obj`, `textured.mtl`, `texture_map.png`) with retries/fallback.
- KIT source is crawled from official list page and per-object `meshes.zip` links; do not hardcode object list.
- MSO uses git sparse checkout from GitHub mirror; keep progress visible in terminal output.

## Testing Guidelines

- No formal test suite yet; validate changes with a small object subset.
- Minimum validation:
  1. run `ingest_assets.py` (`download -> organize -> verify`),
  2. run `process_meshes.py --dataset <source>`,
  3. run `build_object_descriptions.py --dataset <source>`.
- Verify outputs per object: `inertia.obj`, `manifold.obj`, `meshes/*.obj`, `simplified.obj`, `visual.obj`, optional `visual.mtl`, optional `visual_texture_map.png`, `.urdf`, `.xml`.
  - For textured objects, current expectation is single visual texture file (`visual_texture_map.png`).

## Manifest Format

- Manifest path: `assets/objects/processed/<source>/manifest.json`
- Top-level fields:
  - `dataset`, `version`, `generated_at`
  - `source`: `homepage`, `download_method`, `notes`
  - `summary`: `num_objects`, `num_categories`, `has_texture_policy`, `default_mass_kg`
  - `objects`: list of per-object records
- Per-object fields:
  - `object_id`, `name`, `category`
  - `mesh_path`, `mesh_format`
  - `mass_kg`, `has_texture`
  - `mtl_path`, `texture_files`
- Current convention:
  - If dataset has no class taxonomy, write `category=null` for each object and `summary.num_categories=0`.
  - `has_texture` must follow the `.png`-only rule above.

## Commit & Pull Request Guidelines

- Use concise imperative commit messages (existing history: `Update proc.py`, `[add] preprocess YCB models`).
- Recommended format: `<scope>: <change>` (example: `objaverse: fix robust downloader path handling`).
- PRs should include changed dataset(s), commands executed, and sample output paths.
- Remote rule (SSH): `git remote set-url origin git@github.com:HomeworldL/mesh_process.git`

## Data & Security Notes

- Do not commit downloaded datasets or generated assets.
- Keep large files under `assets/objects/raw` locally; include only code and docs in commits.
