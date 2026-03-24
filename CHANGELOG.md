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

### Docs
- Updated `AGENTS.md` to match current stage-1 and stage-2 naming and behavior.

### Known Notes
- Incremental reruns of `process_meshes.py` without `--force` can still leave stale outputs in different frames across `manifold.obj`, `visual.obj`, and `coacd.obj`. Current safe workflow is to rerun with `--force` when recomputing processed objects.
