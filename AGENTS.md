# Repository Guidelines

## Project Structure & Module Organization

- `src/`: main code.
  - `ingest_assets.py`: Stage-1 CLI (`download/organize/verify`) for object sources.
  - `visualize_obj.py`: utility to preview one OBJ from `processed/<source>` (random by default, or specify object id, supports raw/manifold/visual).
  - `visualize_obj_mujcoco.py`: utility to preview one built MJCF (`.xml`) in MuJoCo viewer.
  - `asset_ingest/`: source adapters and manifest schema.
    - `base.py`: abstract adapter interface.
    - `manifest.py`: manifest dataclass + validation.
    - source adapters: `ycb.py`, `realdex.py`, `graspnet.py`, `hope.py`, `kit.py`, `mso.py`, `objaverse.py`, etc.
  - `process_meshes.py`: Stage-2 mesh processing (`raw -> manifold -> inertia-frame transform -> visual/coacd/simplified`).
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
- Normalization + default mass policy:
  - normalized datasets: `ShapeNetCore`, `ShapeNetSem`, `DGN`, `DDG` -> default `mass_kg=50.0`
  - non-normalized datasets: `YCB`, `RealDex`, `GraspNet`, `HOPE`, `KIT`, `MSO`, `Objaverse`, `DexNet` -> default `mass_kg=0.1`

## Build, Test, and Development Commands

- Install dependencies: `pip install trimesh objaverse gdown tqdm pillow`
- Stage-1 ingest:
  - `python src/ingest_assets.py download --source YCB`
  - `python src/ingest_assets.py organize --source YCB`
  - `python src/ingest_assets.py verify --source YCB --check-paths`
  - `python src/ingest_assets.py download --source RealDex`
  - `python src/ingest_assets.py download --source GraspNet`
  - `python src/ingest_assets.py download --source HOPE`
  - `python src/ingest_assets.py download --source KIT`
  - `python src/ingest_assets.py download --source MSO`
  - `python src/ingest_assets.py download --source Objaverse`
  - `python src/ingest_assets.py organize --source Objaverse --sample-n 500 --sample-seed 0`
  - `python src/ingest_assets.py download --source DexNet`
  - `python src/ingest_assets.py organize --source DexNet`
- Stage-2 process: `python src/process_meshes.py --dataset YCB --workers 8`
- Stage-3 export: `python src/build_object_descriptions.py --dataset YCB --force`
- Quick mesh preview: `python src/visualize_obj.py --dataset YCB`
  - mesh selection: `--mesh-type raw|manifold|visual`
Build `third_party/CoACD` and `third_party/ACVD` binaries before running `process_meshes.py`.

## Stage-2 Process Flow (`process_meshes.py`)

- Input: `assets/objects/processed/<dataset>/<object_id>/raw.obj`
- Per-object steps:
  - `mesh_manifold_and_convex_decomp` (single CoACD run) -> `manifold.obj`, `coacd.obj`
    - if generation fails, object is failed
    - `manifold.obj` must satisfy `trimesh.is_watertight=True` and `trimesh.is_volume=True`, otherwise object is failed
  - principal inertia computation is performed on `manifold.obj` (not on raw)
    - inertia computation directly uses trimesh built-ins: `moment_inertia`, `moment_inertia_frame`, `principal_inertia_components`, `principal_inertia_transform`
    - principal axes are assigned with signed/permuted matching so new XYZ remains as close as possible to original XYZ while still aligned to principal directions
    - transformed manifold overwrites `manifold.obj` in principal-inertia frame
  - raw mesh is transformed by the same principal-frame transform before visual export (no `inertia.obj` output is kept)
  - `coacd.obj` is transformed by the same principal-frame transform
  - convex export -> `meshes/coacd_convex_piece_*.obj`
    - invalid CoACD parts are skipped during export if empty, if bbox max extent is below threshold, or if absolute volume is below threshold
  - `mesh_simplify` -> `simplified.obj` (default skip when exists)
  - `mesh_visual` -> `visual.obj` + optional visual material/texture compression outputs
    - textured input is fixed canonical stage-1 `textured.mtl` + `textured.png`, and stage-2 outputs fixed `textured_visual.mtl` + `textured_visual.png`
    - untextured visual meshes are simplified from transformed `raw.obj` with trimesh quadric decimation and exported without UV/normal sidecars
    - textured visual meshes are not geometrically simplified; `visual.obj` is the transformed raw mesh in principal-inertia frame, while only the visual PNG is compressed and the visual MTL is rewritten
    - stage-2 visual mesh generation does not use pymeshlab and does not apply algorithm fallback
    - object is marked success only if generated `visual.obj` can be loaded as a non-empty scene with valid bounds
- Default behavior:
  - step-level skip where outputs already exist
  - `--force` recomputes all steps
  - `--workers` enables object-level parallel processing
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

## Coding Style & Naming Conventions

- Python 3, 4-space indentation, `snake_case` for functions/files, `PascalCase` for classes.
- Keep scripts CLI-driven with `argparse`; avoid hardcoded machine-specific paths.
- Preserve dataset-agnostic path conventions under `assets/objects/{raw,processed}`.
- Manifest mass policy:
  - normalized datasets (`ShapeNetCore`, `ShapeNetSem`, `DGN`, `DDG`): if source mass unknown, write `mass_kg=50.0`
  - other datasets: if source mass unknown, write `mass_kg=0.1`
- Manifest texture policy: `has_texture` is determined only by whether `.png` texture files exist in the organized object folder.
- Organized asset naming:
  - mesh is always exported as `raw.obj`
  - stage-1 OBJ export is unified through `trimesh.exchange.obj.export_obj(...)`
  - organized `raw.obj` is exported without `vn`
  - if stage-1 exports texture sidecars, use the unified names `textured.mtl` and `textured.png`
  - ShapeNetSem/ShapeNetCore default organize mode is OBJ-only
- Organized object id naming:
  - default convention is `<SourceName>_<object_name>` (sanitize via `sanitize_object_id`).
  - handle collisions by suffix (`_2`, `_3`, ...).

## Common Adapter Patterns (Keep Consistent)

- Implement all 3 methods: `download`, `organize`, `build_manifest`.
- `organize` must call `self.write_manifest_for_organize(cfg, report)` at end.
- Prefer trimesh-based export during organize for adapters that standardize OBJ assets.
- Do not use the old `canonicalize_texture_assets(...)` path for new stage-1 organize logic.
- `manifest` path fields must use repo-relative paths via `relative_to_repo(...)`.
- `has_texture` must follow `.png`-only policy.
- If source mass is unavailable, follow dataset policy above (`50.0` for normalized datasets; otherwise `0.1`).
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
- Verify outputs per object: `manifold.obj`, `coacd.obj`, `meshes/*.obj`, `simplified.obj`, `visual.obj`, optional `textured_visual.mtl`, optional `textured_visual.png`, `.urdf`, `.xml`.
  - For textured objects, current expectation is exactly one rewritten visual MTL (`textured_visual.mtl`) plus one compressed visual PNG (`textured_visual.png`).

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
