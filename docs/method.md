# Method, in short (for the slides)

## 1. The unit cell stands for an infinite lattice
A printed infill repeats the same cell many times. Periodic boundary conditions make the one cell
behave as if identical copies surrounded it: opposite faces deform alike, apart from the overall
stretch or shear imposed on the whole cell,

    u(x + t_k) - u(x) = H t_k        (t_k: cell edge vector, H: average displacement gradient)

In the deck this is one `*Equation` per boundary node and direction, tied to three reference points
RP_X, RP_Y, RP_Z whose displacements are H t_k. The reaction forces at those points give the
average stress over the whole cell volume, holes included:

    sigma_ik = RF_k[i] L_k / V

So sigma is the stress of an imaginary uniform material that carries the same load as the lattice.

## 2. Six strain cases give the stiffness C
Each case imposes one strain component (0.2 %) and holds the other five at zero. The six resulting
average stresses are the six columns of the 6x6 matrix C (Voigt order 11, 22, 33, 23, 13, 12):

    sigma = C eps

Everything else - Young's and shear moduli, Poisson ratios in any direction, the polar plots, the
anisotropy index - is computed from C, without further FE runs.

## 3. Virtual tests: UT, ET, simple shear
These imitate laboratory tests on the homogenized material. One average stress state is prescribed
and the other stress components are zero ("stress-free" sides):

| Test | Applied average stress | Measured | Gives |
|---|---|---|---|
| UT (uniaxial tension) | sigma_11 = 1 MPa, rest 0 | eps_11, lateral eps | E, nu |
| ET (equibiaxial tension) | sigma_11 = sigma_22 = 1 MPa | eps_11, eps_22 | E_eb = sigma / eps |
| SS (simple shear) | tau_12 = 1 MPa | gamma_12 | G |

Simple shear: the cell's rotation is held at zero, so the top slides relative to the bottom
(u_1 = gamma x_2), as in a shear test. In linear elasticity these results follow exactly from C;
running them separately checks the setup (agreement to 1e-7 here) and gives the stress fields
for the von Mises plots.

## 4. Von Mises plots
The von Mises stress in the walls under a virtual test, divided by the applied average stress
(1 MPa), is the local stress concentration: how much more the PLA in the wall is loaded than the
homogenized material. Example check: UT along the extrusion axis of a prismatic cell gives
sigma_vM / sigma = 1 / rho exactly (the walls carry the load alone).
Sharp corners where idealized walls meet are elastic singularities: the peak value there grows
with mesh refinement and is not a material stress. The plots cap the colour scale at the 99.5th
percentile; the true maximum is stated.

## 5. Density law and g(rho)
For each independent stiffness entry the FE values at rho = 0.1 ... 0.7 are fitted with

    C_ij(rho) = C_ij^solid h(rho),   h(rho) = rho + rho (1 - rho)(c_0 + c_1 rho + ...)

h is 0 at rho = 0 (empty) and 1 at rho = 1 (solid). Doruk's interpolation function with a void
matrix is g = 1 - h. The number of coefficients is chosen by leave-one-out cross-validation:
every density is left out once, the law is fitted to the other six and predicts it; the
smallest prediction error wins. The power law C/C^s = a rho^n is reported for the exponent n
(n = 1: walls stretch, n = 2-3: walls bend).

## 6. Mesh refinement
Every cell is meshed with 2, 3, 4, 6 elements across the thinnest wall (gyroid 1.5 - 3) at
rho = 0.1 and 0.4. Each stiffness entry is fitted with f(h) = f_0 + a h^p; f_0 is the
mesh-independent estimate, and the production mesh is the coarsest level within 1 % of it.
