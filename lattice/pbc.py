"""Periodic boundary conditions for a box cell [0, Lx] x [0, Ly] x [0, Lz] as *Equation constraints.

The algorithm is the one of the HD repository (hd/fe/abaqus/periodic_equations.py, H. Dal group),
copied here so this repository stands alone:
    u(x + t_k) - u(x) = u(RP_k),   t_k = L_k e_k,  u(RP_k) = H t_k
    1. every node on a 'minus' face is matched with its image on the 'plus' face (error if none);
    2. images linked this way (face: 2 nodes, edge: 4, corner: 8) form a class, master = lowest node;
    3. every other node of the class gets, per DOF:  u(node) - u(master) - sum_k m_k u(RP_k) = 0.
Average stress (virtual work):  sigma_ik = RF_k[i] L_k / V,  V = Lx Ly Lz (holes included).
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FacePair:
    name: str
    minus: object
    plus: object
    translation: tuple
    rp_combination: tuple


@dataclass(frozen=True)
class BoxCell:
    Lx: float
    Ly: float
    Lz: float
    rp_names = ('RP_X', 'RP_Y', 'RP_Z')

    @property
    def L(self):
        return (self.Lx, self.Ly, self.Lz)

    @property
    def volume(self):
        return self.Lx * self.Ly * self.Lz

    @property
    def centre(self):
        return (self.Lx / 2, self.Ly / 2, self.Lz / 2)

    @property
    def tol(self):
        return 1e-6 * max(self.L)

    def periodic_translations(self):
        return ((self.Lx, 0.0, 0.0), (0.0, self.Ly, 0.0), (0.0, 0.0, self.Lz))

    def boundary_mask(self, xyz, tol=None):
        tol = self.tol if tol is None else tol
        x = np.asarray(xyz, dtype=float)
        L = np.array(self.L)
        return ((np.abs(x) < tol) | (np.abs(x - L) < tol)).any(1)

    def face_pairs(self):
        t, L = self.tol, self.L
        out = []
        for a, nm in enumerate('xyz'):
            tr = [0.0, 0.0, 0.0]
            tr[a] = L[a]
            m = [0, 0, 0]
            m[a] = 1
            out.append(FacePair(nm, (lambda p, a=a: np.abs(p[:, a]) < t), (lambda p, a=a, La=L[a]: np.abs(p[:, a] - La) < t),
                                tuple(tr), tuple(m)))
        return tuple(out)


def match_face_pair(xyz, pair, tol):
    from scipy.spatial import cKDTree
    i_minus = np.nonzero(pair.minus(xyz))[0]
    i_plus = np.nonzero(pair.plus(xyz))[0]
    if len(i_minus) != len(i_plus):
        raise RuntimeError(f'PBC {pair.name}: node counts differ ({len(i_minus)} vs {len(i_plus)}); mesh is not periodic')
    d, j = cKDTree(xyz[i_plus]).query(xyz[i_minus] + np.asarray(pair.translation))
    if (d > tol).any():
        raise RuntimeError(f'PBC {pair.name}: opposite node coordinates do not match (max gap {d.max():.3g}); '
                           'mesh is not periodic')
    if len(set(j)) != len(j):
        raise RuntimeError(f'PBC {pair.name}: two nodes map to the same image; mesh is not periodic')
    return i_minus, i_plus[j]


def periodic_classes(xyz, geom):
    links = {}
    for pair in geom.face_pairs():
        m = np.asarray(pair.rp_combination, dtype=int)
        for a, b in zip(*match_face_pair(xyz, pair, geom.tol)):
            links.setdefault(a, []).append((b, m))
            links.setdefault(b, []).append((a, -m))
    key = lambda i: tuple(np.round(xyz[i] / geom.tol).astype(np.int64))
    classes, seen = [], set()
    for start in sorted(links, key=key):
        if start in seen:
            continue
        offset = {start: np.zeros(3, dtype=int)}
        queue = deque([start])
        while queue:
            n = queue.popleft()
            for nb, m in links[n]:
                if nb not in offset:
                    offset[nb] = offset[n] + m
                    queue.append(nb)
                elif (offset[nb] != offset[n] + m).any():
                    raise RuntimeError('PBC: inconsistent periodic images (mesh is not periodic)')
        seen.update(offset)
        master = min(offset, key=key)
        members = sorted((n for n in offset if n != master), key=key)
        classes.append((master, [(n, offset[n] - offset[master]) for n in members]))
    return classes


def equations(xyz, geom, labels, rp_labels):
    """*Equation text (node labels, not names) and the number of equations."""
    lines, n = [], 0
    for master, members in periodic_classes(xyz, geom):
        for node, m in members:
            for dof in (1, 2, 3):
                terms = [(labels[node], dof, 1.0), (labels[master], dof, -1.0)]
                terms += [(rp, dof, -float(mk)) for rp, mk in zip(rp_labels, m) if mk != 0]
                lines.append('*Equation')
                lines.append(str(len(terms)))
                for k in range(0, len(terms), 4):
                    lines.append(', '.join(f'{a}, {d}, {c:.1f}' for a, d, c in terms[k:k + 4]))
                n += 1
    return '\n'.join(lines) + '\n', n


def pin_node(xyz, geom):
    """Interior node closest to the cell centre (stops the rigid translation)."""
    interior = np.nonzero(~geom.boundary_mask(xyz))[0]
    if len(interior) == 0:
        raise RuntimeError('mesh has no interior node to pin')
    return int(interior[np.argmin(np.linalg.norm(xyz[interior] - np.asarray(geom.centre), axis=1))])


def average_stress(rf, geom):
    """rf: (3, 3) reaction forces of RP_X, RP_Y, RP_Z (rows) -> T[i, k] = RF_k[i] L_k / V."""
    rf = np.asarray(rf, dtype=float)
    return np.stack([rf[k] * geom.L[k] for k in range(3)], 1) / geom.volume


def gradient_from_rp(u, geom):
    """u: (3, 3) displacements of RP_X, RP_Y, RP_Z (rows) -> H[i, k] = u_k[i] / L_k."""
    u = np.asarray(u, dtype=float)
    return np.stack([u[k] / geom.L[k] for k in range(3)], 1)
