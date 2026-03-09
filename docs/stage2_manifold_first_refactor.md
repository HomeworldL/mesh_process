# Stage-2 Manifold-First Refactor (2026-03-09)

## Background

The previous Stage-2 pipeline computed principal inertia directly from `raw.obj`. For non-watertight / non-volume meshes, inertia and principal axes could be physically invalid and unstable.

## What Changed

Stage-2 (`src/process_meshes.py`) is now manifold-first:

1. Run CoACD once to generate `manifold.obj` and `coacd.obj` from `raw.obj`.
2. Validate `manifold.obj` with trimesh:
   - `is_watertight == True`
   - `is_volume == True`
   - otherwise mark object as failed.
3. Compute COM / principal axes / principal moments from `manifold.obj`.
4. Apply the principal-frame transform to `manifold.obj` and overwrite it.
5. Apply the same transform to `raw.obj` (temporary file), then directly generate `visual.obj`.
   - `inertia.obj` is no longer generated or kept.
6. Apply the same transform to `coacd.obj`, then export `meshes/coacd_convex_piece_*.obj`.
7. Run simplification from transformed `manifold.obj` to generate `simplified.obj`.

## Function-Level Refactor

- Use one CoACD function:
  - `mesh_manifold_and_convex_decomp(...)` (manifold remesh output + convex decomposition output)
- `mesh_visual(...)` now accepts explicit source OBJ path.
- Old `mesh_transform(...)` output behavior was replaced with `compute_principal_transform(...)`:
  - compute transform/inertia metadata only
  - inertia-related values are read from trimesh built-ins (`moment_inertia`, `moment_inertia_frame`, `principal_inertia_components`, `principal_inertia_transform`)
  - no `inertia.obj` file output
- Added `validate_manifold_mesh(...)` for strict manifold checks.

## Output Contract Changes

Per object, canonical Stage-2 outputs are now:

- `manifold.obj` (already transformed into principal frame)
- `coacd.obj`
- `meshes/coacd_convex_piece_*.obj`
- `simplified.obj`
- `visual.obj` (+ optional `visual.mtl`, optional `visual_texture_map.png`)

Removed from Stage-2 canonical outputs:

- `inertia.obj`

## Compatibility Notes

- Stage-3 (`build_object_descriptions.py`) consumes inertia data from `manifest.process_meshes.json` fields (`principal_moments`), so it remains compatible with this refactor.
- `src/visualize_obj.py` mesh-type options are now `raw|manifold|visual`.
- Existing old folders that still contain `inertia.obj` are harmless but no longer required by the pipeline.

## Motivation and Expected Benefit

This change ensures inertial quantities are derived from validated water-tight volume meshes, significantly reducing invalid inertia failures and unstable principal-axis alignment caused by open/raw meshes.
