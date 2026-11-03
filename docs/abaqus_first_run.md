# First run in Abaqus: what to check

The decks were generated and solved here with CalculiX only. Before running the full set in
Abaqus, run one small deck and check these points.

1. Run one small deck, e.g. `abaqus job=grid_rho40_nt2 interactive` from the convergence set.
2. `.msg` / `.dat`: no errors. Expected warnings, harmless:
   - the three reference nodes RP_X, RP_Y, RP_Z have no elements (they only carry the equations);
   - Abaqus may report distorted or high-aspect elements for the gyroid (sliver tetrahedra at the
     surface, about 1 % of the elements) and for the thin extruded walls.
   If Abaqus stops with "no active degrees of freedom" at a reference node, tell us: the fix is to
   declare the RPs as `*Node` in an assembly or attach them with a dummy element.
3. Extract: `abaqus python abaqus_extract.py grid_rho40_nt2.odb --fields` (the script sits next to the decks)
   -> `grid_rho40_nt2_rp.json` (and `_fields.npz`). The script was written without access to
   Abaqus; if it fails, send the traceback.
4. Compare with CalculiX: the production and convergence decks were also solved here in CalculiX on
   identical meshes (`out/post/stiffness.csv`, `out/post_conv/`); `python scripts/run_ccx.py grid 0.4 --nt 2`
   repeats one.
   Abaqus and CalculiX should agree closely (same mesh, both C3D20R with 2x2x2 integration); a
   difference above about 0.1 % points to a setup problem.
   Then `python scripts/postprocess.py <deck folder>`.

## Viewing the results in Abaqus/Viewer

- Open the ODB, Result > Step/Frame: the steps are named E11 ... G12 (strain cases for C) and
  UT_*, ET_*, SS_* (virtual tests).
- Plot > Contours on deformed shape, field output S, invariant Mises. Use the virtual-test steps
  for the von Mises pictures; their applied stress is 1 MPa, so the contour values are directly the
  stress concentration factor of the wall material.
- Peak values sit at sharp re-entrant corners of the idealized walls and grow with mesh refinement
  (elastic singularity). Use the contour limits of `scripts/contours.py` (99.5th percentile) or
  set the maximum manually; do not report the peak as a material stress.
