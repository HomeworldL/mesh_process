# Changelog

## Unreleased

### Stage-1 Ingest
- Unified organize export around `trimesh.exchange.obj.export_obj(...)`.
- Standardized stage-1 outputs to:
  - geometry-only objects: `raw.obj`
  - textured objects: `raw.obj`, `textured.mtl`, `textured.png`
- Removed stage-1 canonical texture rewrite helpers and simplified adapter export flow.
- Split DGN normalization into:
  - load/normalize mesh
  - export mesh assets
- Kept ShapeNetCore and ShapeNetSem in OBJ-only normalized organize mode by default.
- Updated textured adapters so `remove_unreferenced_vertices()` is only applied on geometry-only export paths.
- Added a progress bar to `python src/ingest_assets.py organize --source YCB`.
- Added independent `DDGAdapter` and removed DDG-specific organize logic from `DGNAdapter`.
- `DDG organize` now derives objects from processed `DGN`, copies `ddg*` object folders, and rebuilds both `manifest.json` and `manifest.process_meshes.json`.
- Added organize progress bars for `MSO`, `GraspNet`, `RealDex`, and `KIT`.
- Added shared organize-time `texture stats: textured=..., untextured=...` reporting for all datasets through `BaseIngestAdapter.write_manifest_for_organize()`.
- Tightened textured organize export gates:
  - `MSO`: require both trimesh texture detection and source `texture.png`; synthesize missing referenced `.mtl` from `texture.png` before loading when needed.
  - `GraspNet`: require `textured.obj`; export texture only if source `texture_map.png` exists and trimesh detects texture.
  - `HOPE`: export texture only if source `texture_map.png` exists and trimesh detects texture.
  - `KIT`: select meshes by `25k_tex > 5k_tex > Orig_tex > 800_tex > 25k > 5k > Orig > 800`, require same-stem PNG for textured export, and rescale meshes from millimeters to meters.
- Confirmed `RealDex` organize remains OBJ-only (no texture assets).
- Changed `Objaverse` organize to:
  - fixed `Daily-Used` subset download by default,
  - normalize geometry like `DGN` (AABB center to origin, max extent to `1.0`),
  - use normalized default mass `50.0 kg`.
- Fixed `load_obj_mesh()` scene loading to bake scene graph transforms with `scene.to_geometry()` before mesh export, which corrects transformed `glb/gltf` assets such as many `Objaverse` samples.

### Stage-2 Mesh Processing
- Reworked `visual` generation:
  - untextured objects: simplify with trimesh
  - textured objects: keep geometry in principal-inertia frame, only compress texture and rewrite visual sidecars
- Removed all `pymeshlab` usage from `process_meshes.py`.
- Standardized stage-2 textured visual outputs to:
  - `visual.obj`
  - `textured_visual.mtl`
  - `textured_visual.png`
- Simplified textured visual input assumptions to canonical stage-1 files:
  - `textured.mtl`
  - `textured.png`
- Kept visual orientation correction against `manifold.obj`.
- Fixed principal inertia computation order for negative-volume meshes:
  - face inversion now happens before reading inertia quantities.
- Improved error reporting so object ids are included in stage failures.
- Changed CoACD / ACVD default timeout from `100s` to `180s`, and fixed CLI help text.
- Removed preview logic and `--preview` from `process_meshes.py`.
- Added fixed CoACD seed plumbing to `process_meshes.py` (`--coacd-seed`, default `0`) for reproducible stage-2 decomposition.

### Stage-3 Description Build
- Updated visual texture lookup to use the fixed stage-2 output name:
  - `textured_visual.png`

### Debug / Maintenance
- Moved ad-hoc debug scripts into local `history/`:
  - `cleanup_visual_outputs.py`
  - `debug_repro_decimation_crash.py`
  - `validate_coacd_transform_equivalence.py`
  - `inspect_mesh.py`
- Added `history/` to `.gitignore`.
- Added `tests/mesh_debug_checks.py` to keep currently useful debug checks:
  - OBJ / visual / normals inspection
  - CoACD transform equivalence check
- Added `src/visualize_glb.py` for quick `glb/gltf/obj` inspection and optional visualization.

### Docs
- Updated `AGENTS.md` to match current stage-1 and stage-2 naming and behavior.
- Updated `README.md` to document current ingest defaults, canonical texture naming, dataset-specific organize differences, and the `Objaverse` fixed subset behavior.

### Known Notes
- Incremental reruns of `process_meshes.py` without `--force` can still leave stale outputs in different frames across `manifold.obj`, `visual.obj`, and `coacd.obj`. Current safe workflow is to rerun with `--force` when recomputing processed objects.
