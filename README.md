# lattice-homogenization

Effective linear-elastic stiffness of 3D-printed lattice cells from periodic-RVE finite element analysis,
with a browser app to explore the results and to run new CalculiX jobs.

- Cells: grid, triangles, honeycomb, cubic (plate), sheet gyroid. Material: PLA. More cells and
  materials can be added (see "Extending").
- Data shipped with the repo: Abaqus/Standard results (production meshes for rho = 0.1 ... 0.7 plus
  mesh-refinement levels) and a CalculiX reference set from the same decks.
- The app runs CalculiX (`ccx`) on your computer and stores every run in a local registry; identical
  runs are not repeated.

```
lattice/       science package: geometry, periodic meshes, decks, PBC, post-processing, density laws,
               convergence (GCI), CalculiX runner, VTU export
lattice_app/   app: FastAPI server, job queue, results registry, browser UI (static/), bundled data
scripts/       command-line pipeline (CAD, Abaqus decks, post-processing, figures)
docs/          method.md (theory), pipeline.md (FE pipeline and verification), specs/
tests/         pytest
workspace/     created on first start, not in git: settings, results registry, job folders
```

## Quick start on Windows with PyCharm

Needed: Python 3.10 or newer (3.14 works), PyCharm, Git for Windows (https://git-scm.com/download/win),
and the CalculiX solver, here `C:\calculix\ccx_i4_static.exe` (and optionally `cgx_STATIC.exe`).

**1. Get the code.** On the PyCharm welcome screen choose *Clone Repository* (or *File > New > Project
from Version Control*), URL `https://github.com/furkannerdogan/lattice-homogenization.git`, pick a folder,
*Clone*. This is the same as typing in a terminal:

```bat
git clone https://github.com/furkannerdogan/lattice-homogenization.git
cd lattice-homogenization
```

`git clone` downloads the repository (all files and their history) into a new folder; `cd` moves the
terminal into that folder so the next commands act on it.

**2. Create the virtual environment.** A virtual environment (`.venv` folder) is a private set of Python
packages for this project only. PyCharm usually offers to create one when it opens the project; accept it
with Python 3.14 as the base. If it does not: *Settings > Project > Python Interpreter > Add Interpreter >
Add Local Interpreter > Virtualenv Environment > New*, base interpreter Python 3.14, location
`<project>\.venv`.

**3. Install the package.** Open PyCharm's terminal (*View > Tool Windows > Terminal*, Alt+F12). It
activates the virtual environment automatically: the prompt starts with `(.venv)`. Then:

```bat
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

`pip install` reads `pyproject.toml` in the current folder (`.`) and installs the dependencies (numpy,
scipy, gmsh, fastapi, uvicorn; `[dev]` adds pytest). `-e` (editable) links the installed package to this
folder, so changes you make to the code take effect without reinstalling.

**4. Point the app to CalculiX.** The app looks for `ccx_i4_static.exe` (also `ccx_static`, `ccx`,
`ccx_2.xx`) on PATH and in the usual folders (`C:\calculix`, its `bin` subfolder, the PrePoMax `Solver`
folder), so with `C:\calculix\ccx_i4_static.exe` nothing is needed. To set it explicitly:

```bat
python -m lattice_app config --ccx C:\calculix\ccx_i4_static.exe --cgx C:\calculix\cgx_STATIC.exe
python -m lattice_app doctor
```

`doctor` prints the Python packages, the solver it found with its version, and the workspace folder.
The path can also be set in the app (*Settings > CalculiX > Auto-detect / Save*) or with the environment
variable `LATTICE_CCX`.

**5. Start the app.**

```bat
python -m lattice_app serve
```

`python -m lattice_app` runs the app package; `serve` starts a local web server and opens
http://127.0.0.1:8765 in your browser. 127.0.0.1 means this computer only; nobody else on the network
can reach it. Stop it with Ctrl+C in the terminal. Options: `--port 8800`, `--no-browser`.

To start it with PyCharm's Run button instead: *Run > Edit Configurations > + > Python*, switch
"Script path" to **Module name** `lattice_app`, *Parameters* `serve`, *Working directory* the project
folder, OK.

**6. Check the installation (optional).** `pytest -q` runs the tests (about 20 s; the CalculiX test
is skipped if ccx is not found).

**Updating later:** *Git > Pull* in PyCharm (or `git pull`), then `pip install -e ".[dev]"` again only if
`pyproject.toml` changed.

Windows may ask once whether to allow `ccx_i4_static.exe` to run (SmartScreen); allow it.

## The app

| Tab | What it shows |
|---|---|
| Explorer | rho slider (continuous), 3D surface of the directional Young's modulus E(n), polar sections E(theta) or G(theta) in the print plane x-y and the vertical planes x-z, y-z (optionally all cells overlaid), C (print or FE frame, MPa or / E_s), engineering constants, A^U, Zener ratio, virtual tests from C, this cell's moduli vs rho |
| Trends | E, G, nu, A^U, E_max/E_min vs rho for all cells, FE points, Voigt and Hashin-Shtrikman bounds |
| Density laws | fitted laws with leave-one-out errors; Poisson-ratio interpolation; the C_ij entry laws for reference |
| Mesh & refinement | mesh preview for any n_t (nodes, element sizes, quality), refinement studies: C_ij vs h, error vs h, observed order p, Richardson extrapolation, GCI; queue a new study |
| Stress fields | von Mises per load case for CalculiX runs made in the app; deformation scale, periodic tiling, sequential or banded colours, ParaView export (.vtu), open in cgx |
| Jobs | queue CalculiX runs (one or several n_t), live solver log, cancel |
| Data | every stored result; CSV of C in the print frame |
| Settings | ccx/cgx paths, threads, materials |

Frames: the app shows results in the **print frame** (x, y in the print plane, z = build direction).
The FE meshes of the prismatic cells use x = extrusion axis; the FE-frame matrix is one click away.

How continuous rho works: the FE results of the selected source (production mesh, rho = 0.1 ... 0.7) are
reduced to the independent engineering constants of each cell's symmetry; moduli follow density laws
E/E_s = h(rho) (mixture or power-mixture form, chosen by leave-one-out error, h(1) = 1), Poisson ratios a
monotone interpolation, and C(rho) is assembled from them. At the FE densities the result matches the FE
matrix (shown in the status line). Outside 0.1 ... 0.7 the app marks the values as extrapolated. Why the
moduli and not the C_ij entries are fitted: `lattice_app/analysis.py` docstring.

Materials: C scales with E_s at fixed nu_s, so a new material with nu_s = 0.35 uses the existing results
at once. A material with another nu_s needs its own runs (Jobs tab); until then the explorer shows no data
for it. Add materials in *Settings > Materials*, each with the source of its values.

Large runs: check the node count in the Mesh tab before queueing. Gyroid at rho = 0.1 with n_t = 2 has
about 470k nodes; the direct solver in the static ccx build may then need many GB of memory and a long
time.

## Command line without the browser

```bat
python -m lattice_app run honeycomb 0.3 --nt 2 3 4 6     # CalculiX jobs into the app registry
python scripts/run_ccx.py cubic 0.3 --nt 2 --out out\ccx   # one run, files only
python scripts/to_paraview.py out\ccx\cubic_rho30_nt2.inp  # .vtu for ParaView
```

Abaqus decks, post-processing and static figures: `docs/pipeline.md`.

## Extending

- Material: *Settings > Materials* (stored in `workspace/materials.json`), or add it to
  `lattice/materials.py` with its source.
- Cell type: geometry and wall-thickness relation in `lattice/geometry.py`, a mesher in `lattice/mesh.py`
  (must give matching nodes on opposite faces), virtual tests in `lattice/loadcases.py`, symmetry class in
  `lattice/post.py` and `lattice_app/analysis.py`. Production mesh level: `PRODUCTION_NT`.
- Results computed elsewhere: `python scripts/build_seed.py <source> <post_dir>` turns a
  `scripts/postprocess.py` output folder into a bundled dataset.

## References

Celik, I. B. et al. (2008). Procedure for estimation and reporting of uncertainty due to discretization in
CFD applications. J. Fluids Eng. 130, 078001. Roache, P. J. (1998). Verification and Validation in
Computational Science and Engineering. Hermosa. Hashin, Z., Shtrikman, S. (1963). A variational approach to
the theory of the elastic behaviour of multiphase materials. J. Mech. Phys. Solids 11, 127-140.
Ranganathan, S. I., Ostoja-Starzewski, M. (2008). Universal elastic anisotropy index. Phys. Rev. Lett. 101,
055504. Schneider, J., Kumar, S. (2023). Comparative performance evaluation of microarchitected lattices
processed via SLS, MJ, and DLP 3D printing methods. J. Mater. Res. Technol. 26, 7182-7198. Pszczolkowski, B. et al. (2022). Materials 15, 149.
