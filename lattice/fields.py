"""Stress fields: reading CalculiX .frd (or the npz written by scripts/abaqus_extract.py), von Mises
stress, outer surface of the mesh, and contour pictures of one cell or of a tiled array.

Peak values sit at the sharp re-entrant corners where idealized walls meet; there the elastic
stress is singular and the FE peak grows with mesh refinement. Contour limits therefore default to
the 99.5th percentile of the nodal values, and the true maximum is reported separately.
"""
import numpy as np

HEX_FACES = [((0, 1, 2, 3), (8, 9, 10, 11)), ((4, 5, 6, 7), (12, 13, 14, 15)), ((0, 1, 5, 4), (8, 17, 12, 16)),
             ((1, 2, 6, 5), (9, 18, 13, 17)), ((2, 3, 7, 6), (10, 19, 14, 18)), ((3, 0, 4, 7), (11, 16, 15, 19))]
TET_FACES = [((0, 1, 2), (4, 5, 6)), ((0, 1, 3), (4, 8, 7)), ((1, 2, 3), (5, 9, 8)), ((0, 2, 3), (6, 9, 7))]


def mises(s):
    """s: (..., 6) in the order xx, yy, zz, xy, yz, zx."""
    s = np.asarray(s, dtype=float)
    return np.sqrt(0.5 * ((s[..., 0] - s[..., 1]) ** 2 + (s[..., 1] - s[..., 2]) ** 2 + (s[..., 2] - s[..., 0]) ** 2)
                   + 3 * (s[..., 3] ** 2 + s[..., 4] ** 2 + s[..., 5] ** 2))


def parse_frd(path, n_nodes):
    """Nodal DISP (n, 3) and STRESS (n, 6: xx yy zz xy yz zx) per step: [(U, S), ...]."""
    steps, cur, block, data = [], {}, None, None
    with open(path) as f:
        for line in f:
            if line.startswith(' -4'):
                block = line.split()[1]
                data = np.zeros((n_nodes, 3 if block == 'DISP' else 6)) if block in ('DISP', 'STRESS') else None
            elif line.startswith(' -1') and data is not None:
                node = int(line[3:13])
                ncol = data.shape[1]
                vals = [float(line[13 + 12 * k:25 + 12 * k]) for k in range(ncol)]
                if node <= n_nodes:
                    data[node - 1] = vals
            elif line.startswith(' -3') and data is not None:
                cur[block] = data
                data = None
                if 'DISP' in cur and 'STRESS' in cur:
                    steps.append((cur['DISP'], cur['STRESS']))
                    cur = {}
    return steps


def surface(elements, etype, nodes=None):
    """Boundary faces of the mesh as triangles (midside nodes used, so quadratic faces look smooth).
    With nodes given, every triangle is wound counter-clockwise seen from outside (consistent normals)."""
    faces = HEX_FACES if etype.startswith('C3D20') else TET_FACES
    corners = np.vstack([elements[:, list(c)] for c, _ in faces])
    mids = np.vstack([elements[:, list(m)] for _, m in faces])
    owner = np.tile(np.arange(len(elements)), len(faces))
    key = np.sort(corners, 1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    outer = idx[cnt == 1]
    c, m, own = corners[outer], mids[outer], owner[outer]
    if nodes is not None:                                       # outward: away from the element centre
        ncorner = 8 if etype.startswith('C3D20') else 4
        centre = nodes[elements[own, :ncorner]].mean(1)
        nrm = np.cross(nodes[c[:, 1]] - nodes[c[:, 0]], nodes[c[:, 2]] - nodes[c[:, 0]])
        flip = np.einsum('ij,ij->i', nrm, nodes[c].mean(1) - centre) < 0
        if c.shape[1] == 4:
            c[flip] = c[flip][:, [0, 3, 2, 1]]
            m[flip] = m[flip][:, [3, 2, 1, 0]]
        else:
            c[flip] = c[flip][:, [0, 2, 1]]
            m[flip] = m[flip][:, [2, 1, 0]]
    if c.shape[1] == 4:                                         # quad8 -> 6 triangles
        tri = [(c[:, 0], m[:, 0], m[:, 3]), (c[:, 1], m[:, 1], m[:, 0]), (c[:, 2], m[:, 2], m[:, 1]),
               (c[:, 3], m[:, 3], m[:, 2]), (m[:, 0], m[:, 1], m[:, 2]), (m[:, 0], m[:, 2], m[:, 3])]
    else:                                                       # tri6 -> 4 triangles
        tri = [(c[:, 0], m[:, 0], m[:, 2]), (c[:, 1], m[:, 1], m[:, 0]), (c[:, 2], m[:, 2], m[:, 1]),
               (m[:, 0], m[:, 1], m[:, 2])]
    return np.vstack([np.stack(t, 1) for t in tri])


def tile_field(nodes, tris, values, U, H, box, n):
    """Tiled copies of a periodic cell with a periodic scalar field and displacement u(x + t) = u(x) + H t."""
    V, F, S, D = [], [], [], []
    k = 0
    for i in range(n[0]):
        for j in range(n[1]):
            for l in range(n[2]):
                t = np.array([i * box[0], j * box[1], l * box[2]])
                V.append(nodes + t)
                D.append(U + H @ t)
                F.append(tris + k * len(nodes))
                S.append(values)
                k += 1
    V, F, S, D = np.concatenate(V), np.concatenate(F), np.concatenate(S), np.concatenate(D)
    keep = np.ones(len(F), dtype=bool)                          # drop faces on internal cell planes
    for a in range(3):
        c = V[F][:, :, a]
        q = np.round(c / box[a])
        on = (np.abs(c - q * box[a]) < 1e-6 * box[a]).all(1) & (q == q[:, :1]).all(1)
        keep &= ~(on & (q[:, 0] > 0) & (q[:, 0] < n[a]))
    return V, F[keep], S, D


def contour_png(nodes, elements, etype, values, png, U=None, H=None, box=None, n=(1, 1, 1), deform=0.0,
                vrange=None, to_cad=None, **kw):
    """Banded contour picture. to_cad: function mapping FE to CAD coordinates (view with the build axis up)."""
    from .render import render
    tris = surface(elements, etype, nodes)
    U = np.zeros_like(nodes) if U is None else U
    H = np.zeros((3, 3)) if H is None else H
    V, F, S, D = tile_field(nodes, tris, values, U, H, box, n)
    X = V + deform * D
    if to_cad is not None:
        X = to_cad(X)
    if vrange is None:
        vrange = (0.0, float(np.percentile(values, 99.5)))
    render(X, F, png, scalars=S, vrange=vrange, banded=True, **kw)
    return vrange
