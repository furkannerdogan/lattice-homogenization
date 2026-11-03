"""Post-processing of the homogenized stiffness C (Voigt 11 22 33 23 13 12, engineering shear).

engineering_constants   E_i, G_ij, nu_ij from S = C^-1
symmetry_checks         cubic / transversely isotropic (axis x) / tetragonal (axis x) relations
directional_E           E(n) = 1 / (n n : S : n n)    (Mandel form)
universal_anisotropy    A^U = 5 G_V / G_R + K_V / K_R - 6  (Ranganathan & Ostoja-Starzewski 2008)
predicted_test          strain of a virtual test predicted from C, for the cross-check with the FE test
"""
import numpy as np

from .loadcases import VOIGT_PAIRS

W = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])


def iso_C(E, nu):
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[[0, 1, 2], [0, 1, 2]] += 2 * mu
    C[3:, 3:] = np.eye(3) * mu
    return C


def engineering_constants(C):
    S = np.linalg.inv(C)
    E = 1 / np.diag(S)[:3]
    G = 1 / np.diag(S)[3:]
    return {'E1': E[0], 'E2': E[1], 'E3': E[2], 'G23': G[0], 'G13': G[1], 'G12': G[2],
            'nu12': -S[0, 1] * E[0], 'nu13': -S[0, 2] * E[0], 'nu23': -S[1, 2] * E[1],
            'nu21': -S[0, 1] * E[1], 'nu31': -S[0, 2] * E[2], 'nu32': -S[1, 2] * E[2]}


def _rel(a, b):
    return float(abs(a - b) / max(abs(a), abs(b), 1e-30))


def symmetry_checks(C, pattern):
    """Relative deviations from the symmetry the cell must have (FE frame; x = extrusion for the prismatic cells)."""
    c = lambda i, j: C[i - 1, j - 1]
    off = [C[i, j] for i in range(3) for j in range(3, 6)] + [C[3, 4], C[3, 5], C[4, 5]]
    out = {'normal_shear_coupling': float(np.max(np.abs(off)) / np.max(np.abs(C)))}
    if pattern in ('cubic', 'gyroid'):
        out.update(C22_C11=_rel(c(2, 2), c(1, 1)), C33_C11=_rel(c(3, 3), c(1, 1)), C13_C12=_rel(c(1, 3), c(1, 2)),
                   C23_C12=_rel(c(2, 3), c(1, 2)), C55_C44=_rel(c(5, 5), c(4, 4)), C66_C44=_rel(c(6, 6), c(4, 4)))
    else:
        out.update(C33_C22=_rel(c(3, 3), c(2, 2)), C13_C12=_rel(c(1, 3), c(1, 2)), C66_C55=_rel(c(6, 6), c(5, 5)))
        if pattern in ('triangles', 'honeycomb'):        # 6-fold axis: transversely isotropic about x
            out['C22_C23+2C44'] = _rel(c(2, 2), c(2, 3) + 2 * c(4, 4))
    return out


def independent_entries(pattern):
    """Independent stiffness entries of the cell's symmetry class (1-based Voigt indices)."""
    if pattern in ('cubic', 'gyroid'):
        return {'C11': (1, 1), 'C12': (1, 2), 'C44': (4, 4)}
    d = {'C11': (1, 1), 'C22': (2, 2), 'C12': (1, 2), 'C23': (2, 3), 'C44': (4, 4), 'C55': (5, 5)}
    if pattern in ('triangles', 'honeycomb'):
        d.pop('C44')                                     # C44 = (C22 - C23) / 2
    return d


def symmetrize(C, pattern):
    """Average entries that are equal by symmetry (removes mesh noise before fitting)."""
    C = C.copy()
    off = np.zeros((6, 6), bool)
    off[:3, 3:] = off[3:, :3] = True
    off[3, 4] = off[4, 3] = off[3, 5] = off[5, 3] = off[4, 5] = off[5, 4] = True
    C[off] = 0.0
    if pattern in ('cubic', 'gyroid'):
        groups = [[(0, 0), (1, 1), (2, 2)], [(0, 1), (0, 2), (1, 2)], [(3, 3), (4, 4), (5, 5)]]
    else:
        groups = [[(1, 1), (2, 2)], [(0, 1), (0, 2)], [(4, 4), (5, 5)]]
    for g in groups:
        v = np.mean([C[i, j] for i, j in g])
        for i, j in g:
            C[i, j] = C[j, i] = v
    if pattern in ('triangles', 'honeycomb'):
        C[3, 3] = 0.5 * (C[1, 1] - C[1, 2])
    return C


def mandel(C):
    return C * np.outer(W, W)


def directional_E(C, n):
    """n: (..., 3) unit vectors -> E(n)."""
    S = np.linalg.inv(mandel(C))
    n = np.asarray(n, dtype=float)
    N = np.stack([n[..., i] * n[..., j] for i, j in VOIGT_PAIRS], -1) * W
    return 1.0 / np.einsum('...i,ij,...j->...', N, S, N)


def directional_G(C, n, m):
    """Shear modulus for shear on the plane with normal n in direction m (n . m = 0)."""
    S = np.linalg.inv(mandel(C))
    M = np.stack([0.5 * (n[..., i] * m[..., j] + n[..., j] * m[..., i]) for i, j in VOIGT_PAIRS], -1) * W
    return 1.0 / (4 * np.einsum('...i,ij,...j->...', M, S, M))


def universal_anisotropy(C):
    S = np.linalg.inv(C)
    KV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[1, 2] + C[0, 2])) / 9
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - (C[0, 1] + C[1, 2] + C[0, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
    KR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[1, 2] + S[0, 2]))
    GR = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[1, 2] + S[0, 2]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    return float(5 * GV / GR + KV / KR - 6), dict(KV=KV, GV=GV, KR=KR, GR=GR)


def zener(C):
    return float(2 * C[3, 3] / (C[0, 0] - C[0, 1]))


def predicted_test(C, sigma_voigt):
    return np.linalg.solve(C, sigma_voigt)


def test_moduli(sig, eps, name):
    """Apparent moduli of a virtual test from its average stress and strain (Voigt, engineering shear)."""
    idx = {'x': 0, 'y': 1, 'z': 2}
    kind, plane = name.split('_')[0], name.split('_')[1]
    if kind == 'UT':
        if plane == 'yz45':
            n = np.array([0, 1, 1]) / np.sqrt(2)
            e = np.array([[eps[0], eps[5] / 2, eps[4] / 2], [eps[5] / 2, eps[1], eps[3] / 2],
                          [eps[4] / 2, eps[3] / 2, eps[2]]])
            return {'E': 1.0 / (n @ e @ n)}
        a = idx[plane]
        others = [b for b in range(3) if b != a]
        return {'E': sig[a] / eps[a], **{f'nu_{plane}{"xyz"[b]}': -eps[b] / eps[a] for b in others}}
    if kind == 'ET':
        a, b = idx[plane[0]], idx[plane[1]]
        return {'E_eb': sig[a] / (0.5 * (eps[a] + eps[b])), f'eps_{plane[0]}/eps_{plane[1]}': eps[a] / eps[b]}
    if kind == 'SS':
        k = {'yz': 3, 'xz': 4, 'xy': 5}[plane]
        return {'G': sig[k] / eps[k]}
    raise ValueError(name)
