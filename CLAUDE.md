# Conventions for this repository

- English only in code, comments, generated text and reports (no Turkish characters).
- Terse comments: say why, not what. Module docstrings state the method and its sources.
- Science lives in `lattice/` (no web code there); `lattice_app/` only calls it.
- Units mm, N, MPa. Voigt order 11, 22, 33, 23, 13, 12 with engineering shear. FE frame of the prismatic
  cells: x = extrusion axis; the app shows the print frame (z = build direction).
- A change that alters the meshes must bump `lattice.pipeline.MESH_VERSION` (cache keys depend on it).
- Bundled datasets in `lattice_app/data/` are read-only results; regenerate them with `scripts/build_seed.py`.
- Never commit `workspace/` or solver output (`*.frd`, `*.dat`); commit only your own files, with a pathspec.
- Tests: `pytest -q` must pass before a commit.
