# Third-Party Libraries

This directory contains local checkouts of third-party libraries used by this repository.

## CoACD

- Upstream: `https://github.com/SarahWeiii/CoACD.git`
- Local checkout: `third_party/CoACD`
- Current HEAD: `585b976`
- Current modification:
  - Local uncommitted source change in `main.cpp`
  - Current diff changes remesh OBJ export from:
    - `m.SaveOBJ(params.remesh_output_name);`
  - To:
    - `Model tmp = m;`
    - `tmp.Recover(bbox);`
    - `tmp.SaveOBJ(params.remesh_output_name);`

## ACVD

- Upstream: `https://github.com/valette/ACVD.git`
- Local checkout: `third_party/ACVD`
- Current HEAD: `30acb26`
- Current modification:
  - No tracked source edits observed in the local worktree
  - Local untracked build artifacts are present, including:
    - `CMakeCache.txt`
    - `CMakeFiles/`
    - `bin/`
    - generated `*Config.cmake`, `*Targets.cmake`, `Makefile`, and nested build directories
