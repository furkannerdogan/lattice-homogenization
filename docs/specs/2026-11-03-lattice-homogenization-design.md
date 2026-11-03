# Lattice homogenization: design (agreed 2026-11-03)

## Intent
Effective linear-elastic behaviour of five idealized infill/lattice unit cells, from periodic-RVE
finite element analysis in Abaqus (or CalculiX), for the poster and paper of the computational
homogenization group (Prof. H. Dal, METU). Experiments may follow later; they are not part of this.

## Scope
- Cells: grid, triangles, honeycomb, cubic (= SC plate lattice of Schneider & Kumar 2023), sheet gyroid
  (Schneider & Kumar Eq. 1).
- Cell size L = 20 mm for every pattern; density changed by wall thickness only; rho = 0.1 ... 0.7, step 0.1,
  the same list for all five patterns.
- Material: PLA, linear elastic, small strain, no NLGEOM. E = 2100 MPa (static tensile test of FDM PLA,
  Pszczolkowski et al., Materials 15 (2022) 149), nu = 0.35 (assumed). Results scale with E; nu matters.
- Idealized walls (CAD), no slicer toolpaths, no print defects.
- Periodic cells only. No finite specimens, no plates, no comparison with measurements.

## Analyses
- Six strain cases per cell and density (all reference-point DOFs prescribed): full C, 6x6.
- Virtual tests, stress-controlled, the unloaded directions stress-free (Dal, Transverse Isotropy, Table 2):
  UT, ET, simple shear. Prismatic cells (x = extrusion): UT_x, UT_y, UT_z, ET_yz, ET_xy, SS_xy, SS_yz;
  grid also UT at 45 deg in the section plane. Cubic, gyroid: UT_x, ET_xy, SS_xy.
  Lower triangle of the displacement gradient held at zero: no rigid rotation, shear tests are simple shear.
- Von Mises contour plots of the virtual tests.
- Engineering constants, directional E (polar plots), A^U, symmetry checks, cross-check of the virtual tests
  against C.
- Mesh refinement: rho = 0.1 and 0.4, four levels (2, 3, 4, 6 elements across the thinnest wall; gyroid
  1.5, 2, 2.5, 3), extrapolation f(h) = f0 + a h^p; production level = coarsest within 1 %.
- Density law: mixture form of the HD repository, C_ij = C_ij^s h(rho), h = rho + rho(1-rho) sum c_i rho^i,
  number of coefficients by leave-one-out cross-validation over the 7 densities; g = 1 - h (void matrix).
  Power law a rho^n reported for the scaling exponent.

## Approaches considered
- (A, chosen) Standalone repository: periodic meshes and complete input decks generated in Python, tested
  in CalculiX, run in Abaqus with `abaqus job=...`; Python post-processing. Same PBC algorithm and stress
  convention as the HD repository, so results are comparable and the step to (B) is small.
- (B) A 'lattice' geometry type inside the HD repository: GUI and job cache, but changes across study.py,
  hashing, workbench and GUI, untestable without Abaqus here. Possible later.
- (C) Native Abaqus CAE geometry in the HD repository: cannot make the gyroid; most code.

## Decisions taken by the builder (check these)
- FE frame for the prismatic cells: x = extrusion (HD fiber axis), so the transversely isotropic
  identification of the HD repository applies to honeycomb and triangles; CAD/STL frame: z = build direction.
- Elements: C3D20R for grid, triangles, honeycomb, cubic (structured or extruded quads); C3D10 for the gyroid
  (marching tetrahedra on a periodic grid, level corrected so the meshed rho is exact).
- Rectangular periodic cells for triangles and honeycomb: 20 x 34.64 mm section. Triangles shifted by
  sqrt3/4 L so that no wall lies on the cell boundary.
- CalculiX average stress from boundary-node forces (CalculiX reports no reaction at free reference nodes);
  Abaqus from the reference-point reactions as in the HD repository. The two are equal by virtual work;
  only the CalculiX path has been run here (no Abaqus licence in this environment).
