# lattice_hom: periodic homogenization of 3D-printed lattice cells

Effective linear-elastic stiffness of five idealized unit cells (grid, triangles, honeycomb, cubic,
sheet gyroid) at relative densities 0.1 ... 0.7, from periodic-RVE finite element analysis in
Abaqus/Standard or CalculiX. Design and decisions: `docs/specs/2026-11-03-lattice-homogenization-design.md`.

```
CAD (STL/STEP)  ->  periodic mesh  ->  input deck (PBC + load cases)  ->  Abaqus / CalculiX
                                                                             |
      C (6x6), E/G/nu, polar plots, UT/ET/SS tests, von Mises plots, density law, mesh convergence
```

## Install

Python 3.10+: `pip install numpy scipy gmsh trimesh scikit-image matplotlib playwright pytest`
(`fast-simplification` for the overview pictures). CalculiX `ccx` is optional (local checks).

## Cells and conventions

| Pattern | Cell (CAD frame, mm) | Wall thickness t for density rho (L = 20 mm) |
|---|---|---|
| grid | 20 x 20 x 20 | rho = 1 - (1 - t/L)^2 |
| triangles | 20 x 34.64 x 20 | rho = 1 - (1 - sqrt3 t/L)^2 (triangle side L) |
| honeycomb | 20 x 34.64 x 20 | rho = 1 - (1 - t/L)^2 (hexagon flat-to-flat L) |
| cubic | 20 x 20 x 20 | rho = 1 - (1 - t/L)^3 (SC plate lattice, Schneider & Kumar 2023 Eq. 4) |
| gyroid | 20 x 20 x 20 | sheet abs(g) <= c, g = Schneider & Kumar Eq. 1; c from the density |

- CAD/STL frame: z = build direction (prismatic cells extruded along z), ready for a slicer.
- FE frame: for grid, triangles and honeycomb **x = extrusion axis** (the fiber axis of the HD
  repository), (x, y, z)_fe = (z, x, y)_cad. Cubic and gyroid: same as CAD.
- Units mm, N, MPa. PLA: E = 2100 MPa (static tensile test of FDM PLA, Pszczolkowski et al.,
  Materials 15 (2022) 149), nu = 0.35 (assumed). Linear: C/E_s depends only on nu.
- Voigt order 11, 22, 33, 23, 13, 12, engineering shear strain (as the HD repository).

## Workflow

```sh
python scripts/make_cad.py                       # out/cad: 35 STL (+ 28 STEP), cad_summary.json
python scripts/render_cad.py                     # out/figures: cells_overview.png, arrays.png
python scripts/make_decks.py convergence         # out/decks/convergence: 40 Abaqus decks
python scripts/make_decks.py production          # out/decks/production: 35 Abaqus decks
```

Run in Abaqus (each deck is complete: mesh, material, periodic equations, all steps):

```bat
cd out\decks\production
abaqus job=honeycomb_rho30_nt3 cpus=4 interactive
abaqus python abaqus_extract.py honeycomb_rho30_nt3.odb --fields
```
or `run_all.bat` in the deck folder (`abaqus_extract.py` is copied next to the decks). Keep the `.json` next to each deck: the extraction and the
post-processing read the step list and reference-point labels from it.

Post-process (any mix of Abaqus and CalculiX results):

```sh
python scripts/postprocess.py out/decks/production out/decks/convergence --out out/post
python scripts/contours.py out/decks/production honeycomb 0.3     # von Mises pictures
```

Local check without Abaqus: `python scripts/run_ccx.py honeycomb 0.3 --nt 2`.

## What each deck contains

| Step | Kind | Gives |
|---|---|---|
| E11 E22 E33 G23 G13 G12 | strain cases: one Voigt strain (0.002), all others zero; every reference-point DOF prescribed | the six columns of C |
| UT_x, UT_y, UT_z | uniaxial tension, other stresses zero | E, Poisson ratios, von Mises field |
| ET_yz, ET_xy | equibiaxial tension (two equal normal stresses), other stresses zero | equibiaxial modulus |
| SS_xy, SS_yz | simple shear (lower triangle of the displacement gradient held at zero) | G |
| UT_yz45 (grid) | uniaxial tension at 45 deg in the section plane | in-plane anisotropy |

Cubic and gyroid have UT_x, ET_xy, SS_xy (cubic symmetry makes the other directions equal).
Virtual tests are stress-controlled with 1 MPa: forces RF_k[i] = sigma_ik V / L_k on the reference points.

Periodic boundary conditions: `u(x + t_k) - u(x) = u(RP_k)`, written as `*Equation` with the
algorithm of the HD repository (`hd/fe/abaqus/periodic_equations.py`, copied in `lattice/pbc.py`).
Average stress: `sigma_ik = RF_k[i] L_k / V`, V = full cell volume (holes included).

## Meshes

- grid, cubic: structured C3D20R, `n_t` elements across the wall, graded away from it.
- triangles, honeycomb: periodic all-quad section mesh (gmsh), extruded in 4 layers, C3D20R.
- gyroid: C3D10 from marching tetrahedra on a periodic grid; the level c is corrected so the
  meshed density is exact (the gyroid shape changes by O(h^2)).
- Opposite faces carry matching nodes by construction; deck generation stops if they do not.

## Verification done here (CalculiX 2.21)

- Solid cell (C3D20R and C3D10): C equals the bulk PLA stiffness to print precision; UT gives
  E, SS gives G exactly; steps are independent.
- Every pattern: C symmetric to 1e-8, normal-shear coupling zero, symmetries of the cell met
  (cubic, tetragonal, transversely isotropic C22 = C23 + 2 C44 for triangles and honeycomb).
- Virtual tests agree with the strain predicted from C.
- Mesh convergence (rho = 0.1 and 0.4, 2-6 elements across the thinnest wall; error measured as
  |C_ij - C_ij(h->0)| / sqrt(C_ii C_jj)): triangles, honeycomb, cubic within 0.5 % at n_t = 3;
  grid in-plane shear converges slowest (sharp 90-degree corners), within 1 % from n_t = 6;
  gyroid (rho = 0.4) 1.4 % at n_t = 2, 0.8 % at n_t = 2.5.
  Production levels (`PRODUCTION_NT` in `scripts/make_decks.py`): grid 8, triangles 3, honeycomb 3,
  cubic 3, gyroid 2. Details: `out/post_conv/convergence.json` and `convergence_*.png`.
- Reference results of 33 of the 35 production decks computed here in CalculiX, on the same meshes as
  the Abaqus decks: `out/post/stiffness.csv`, `results.json`, figures. Not run here: gyroid rho = 0.1
  and 0.2 (470k and 122k nodes, too large for this 2-core machine).

Not done here: running the decks in Abaqus (no licence in this environment), and
`scripts/abaqus_extract.py` checked on grid_rho40_nt2 in Abaqus: C matches CalculiX to 1e-7.

## Reference results (CalculiX, production meshes)

`out/post/`: `stiffness.csv` (C, engineering constants / E_s, A^U, symmetry checks per cell and rho),
`virtual_tests.csv` (UT/ET/SS moduli from the FE tests and from C; agree to 2e-6), `laws.json`
(density laws per stiffness entry with leave-one-out errors), figures `constants_vs_rho.png`,
`anisotropy_vs_rho.png`, `polar_E.png`, `laws_*.png`; `out/figures/mises_*.png` (von Mises).

Main observations:
- Prismatic cells along the extrusion axis: E = rho E_s exactly (walls stretch in parallel).
- In the print plane at rho = 0.1: E/E_s = 0.055 grid (along a wall), 0.038 triangles, 0.0019
  honeycomb (wall bending); in-plane shear G/E_s = 8e-5 grid, 0.015 triangles, 5e-4 honeycomb.
- Honeycomb and triangles are transversely isotropic (6-fold axis): E(n) is a circle in the print
  plane. Grid is strongly anisotropic in the print plane (A^U = 251 at rho = 0.1).
- Cubic: Zener ratio 0.42 - 0.71 (soft in shear relative to the axes). Gyroid: nearly isotropic,
  A^U = 0.01 - 0.03, Zener ratio about 1.1 - 1.15.
- Density laws: mixture form within 0.1 - 2 % (leave-one-out) for stretch-type entries; grid in-plane
  shear needs the power-mixture form (exponent about 3.5, bending), leave-one-out error 5 %.
- Gyroid rho = 0.1 and 0.2 are not in this set (run them in Abaqus); its laws use rho = 0.3 - 0.7.

## Limitations

- Idealized walls with sharp corners: the elastic stress at re-entrant corners is singular, so the
  peak von Mises value grows with mesh refinement. Contour limits use the 99.5th percentile and
  the true maximum is reported separately. Stiffness converges normally.
- No print effects (bead shape, layer bonding, porosity, anisotropy of printed PLA).
- Linear elasticity only.
