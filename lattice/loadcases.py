"""Load cases (FE frame; for the prismatic cells x is the extrusion axis).

Strain cases (give C): one Voigt strain component, all others zero. Displacement gradient H with
H[i, i] = eps for a normal case and H[i, j] = gamma (i < j, simple shear) for a shear case; every
reference-point DOF is prescribed, u(RP_k) = H t_k. Voigt order 11, 22, 33, 23, 13, 12.

Virtual tests (stress-controlled, the directions that are not loaded are stress-free, as in Dal,
Transverse Isotropy, Table 2): a prescribed average stress sigma. The lower triangle of H is held at
zero (H[i, k] = 0 for i > k), which removes the rigid rotation and makes the shear tests simple
shear, u_1 = gamma x_2. The upper triangle and the diagonal are free and loaded by the forces
RF_k[i] = sigma[i, k] V / L_k on the reference points.
"""
import numpy as np

VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
VOIGT_LABELS = ('11', '22', '33', '23', '13', '12')

STRAIN_CASES = ('E11', 'E22', 'E33', 'G23', 'G13', 'G12')
STRAIN = 0.002
STRESS = 1.0                                   # MPa (linear: the value only scales the output)


def strain_gradient(k, eps=STRAIN):
    H = np.zeros((3, 3))
    i, j = VOIGT_PAIRS[k]
    H[min(i, j), max(i, j)] = eps
    return H


def _s(**c):
    S = np.zeros((3, 3))
    idx = {'11': (0, 0), '22': (1, 1), '33': (2, 2), '23': (1, 2), '13': (0, 2), '12': (0, 1)}
    for k, v in c.items():
        i, j = idx[k[1:]]
        S[i, j] = S[j, i] = v
    return S


def virtual_tests(name):
    """{test name: (description, sigma (3x3) per unit STRESS)} for a pattern."""
    if name in ('cubic', 'gyroid'):
        return {
            'UT_x': ('uniaxial tension along x', _s(s11=1)),
            'ET_xy': ('equibiaxial tension in the x-y plane', _s(s11=1, s22=1)),
            'SS_xy': ('simple shear in the x-y plane', _s(s12=1)),
        }
    tests = {
        'UT_x': ('uniaxial tension along the extrusion axis x', _s(s11=1)),
        'UT_y': ('uniaxial tension in-plane along y', _s(s22=1)),
        'UT_z': ('uniaxial tension in-plane along z', _s(s33=1)),
        'ET_yz': ('equibiaxial tension in the section plane y-z', _s(s22=1, s33=1)),
        'ET_xy': ('equibiaxial tension in the plane x-y (extrusion + in-plane)', _s(s11=1, s22=1)),
        'SS_xy': ('simple shear in the x-y plane (longitudinal shear)', _s(s12=1)),
        'SS_yz': ('simple shear in the section plane y-z (transverse shear)', _s(s23=1)),
    }
    if name == 'grid':
        tests['UT_yz45'] = ('uniaxial tension in-plane at 45 deg between y and z', _s(s22=0.5, s33=0.5, s23=0.5))
    return tests
