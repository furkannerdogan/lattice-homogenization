"""Periodic FE meshes of the unit cells, FE frame, mm.

    grid, cubic          structured hexahedra (tensor grid, graded away from the walls)
    triangles, honeycomb 2D quadrilaterals of the section (gmsh, periodic curves), extruded along x
    gyroid               tetrahedra cut from a periodic Freudenthal background grid (marching tetrahedra,
                         consistent splitting of prisms and pyramids, Dompierre et al. 1999)

Linear elements are built first and then made quadratic with straight-edge midside nodes:
C3D20R (Abaqus node order) for hexahedra, C3D10 for tetrahedra. Opposite faces of the cell
have matching nodes by construction; lattice.pbc checks it again when the equations are written.

Mesh density: n_t elements across the thinnest wall (h = t_min / n_t); the extrusion direction of
the prismatic cells has n_ext layers (the periodic fluctuation does not vary along it).
"""
import threading
from dataclasses import dataclass, field

import numpy as np

from .geometry import (L_DEFAULT, PRISMATIC, S3, cad_box, fe_box, gyroid_g, gyroid_level, smallest_feature,
                       wall_segments, wall_thickness)


@dataclass
class Mesh:
    nodes: np.ndarray            # (n, 3) FE frame, mm
    elements: np.ndarray         # (m, 20) C3D20R or (m, 10) C3D10, 0-based node indices
    etype: str
    box: tuple                   # (Lx, Ly, Lz) FE frame
    info: dict = field(default_factory=dict)

    @property
    def volume_fraction(self):
        return element_volumes(self).sum() / np.prod(self.box)


# ---------------------------------------------------------------------------------------------
# quadratic elements
# ---------------------------------------------------------------------------------------------
HEX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
TET_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


def make_quadratic(nodes, elems, edges):
    """Adds one straight midside node per element edge, shared between elements."""
    a = elems[:, [e[0] for e in edges]].ravel()
    b = elems[:, [e[1] for e in edges]].ravel()
    key = np.minimum(a, b).astype(np.int64) * (len(nodes) + 1) + np.maximum(a, b)
    uniq, inv = np.unique(key, return_inverse=True)
    lo, hi = uniq // (len(nodes) + 1), uniq % (len(nodes) + 1)
    mid = 0.5 * (nodes[lo] + nodes[hi])
    new = len(nodes) + inv.reshape(len(elems), len(edges))
    return np.vstack([nodes, mid]), np.hstack([elems, new])


def hex_volume_sign(nodes, h):
    x = nodes[h[:, :8]]
    c = x.mean(1)
    j = np.einsum('ij,ij->i', np.cross(x[:, 1] - x[:, 0], x[:, 3] - x[:, 0]), x[:, 4] - x[:, 0])
    return np.sign(j), c


def fix_hex_orientation(nodes, h):
    s, _ = hex_volume_sign(nodes, h)
    bad = s < 0
    h[bad] = h[bad][:, [4, 5, 6, 7, 0, 1, 2, 3]]          # swap top and bottom face
    return h


def tet_volumes(nodes, t):
    x = nodes[t[:, :4]]
    return np.einsum('ij,ij->i', np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), x[:, 3] - x[:, 0]) / 6.0


def hex_volumes(nodes, h):
    """Volume of linear (straight-edged) hexahedra: 6 tetrahedra per hex."""
    tets = [(0, 1, 2, 5), (0, 2, 7, 5), (0, 2, 3, 7), (0, 5, 7, 4), (2, 7, 5, 6)]
    return sum(tet_volumes(nodes, h[:, list(t)]) for t in tets)


def element_volumes(m):
    return hex_volumes(m.nodes, m.elements) if m.etype == 'C3D20R' else tet_volumes(m.nodes, m.elements)


# ---------------------------------------------------------------------------------------------
# structured hexahedra: grid and cubic
# ---------------------------------------------------------------------------------------------
def graded(length, h0, ratio=1.25, hmax=None):
    """Points from 0 to length, size h0 at 0 growing by `ratio` up to hmax (fine at the start)."""
    hmax = hmax or length
    sizes, s = [], 0.0
    while s < length - 1e-12:
        h = min(h0 * ratio ** len(sizes), hmax)
        sizes.append(h)
        s += h
    sizes = np.array(sizes) * length / sum(sizes)
    return np.concatenate([[0.0], np.cumsum(sizes)])


def wall_axis(L, t, n_t, ratio, hmax):
    """Breakpoints of one axis with a centred wall of thickness t; returns (coords, index range of the wall)."""
    a, b = L / 2 - t / 2, L / 2 + t / 2
    n_t = int(round(n_t))
    # cosine spacing across the wall: smaller elements at the wall faces, where the re-entrant
    # corners of the junctions are; the outer intervals start with the same size
    wall = a + t * 0.5 * (1 - np.cos(np.pi * np.arange(n_t + 1) / n_t))
    h0 = wall[1] - wall[0]
    left = a - graded(a, h0, ratio, hmax)[::-1]            # fine near the wall
    right = b + graded(L - b, h0, ratio, hmax)
    x = np.concatenate([left, wall[1:], right[1:]])
    i0 = len(left) - 1
    return x, (i0, i0 + n_t)


def tensor_hex(xs, ys, zs, keep):
    """Hexahedra of a tensor grid; keep(i, j, k) boolean arrays of cell indices."""
    nx, ny, nz = len(xs), len(ys), len(zs)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    nid = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    I, J, K = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing='ij')
    m = keep(I, J, K)
    i, j, k = I[m], J[m], K[m]
    h = np.stack([nid[i, j, k], nid[i + 1, j, k], nid[i + 1, j + 1, k], nid[i, j + 1, k],
                  nid[i, j, k + 1], nid[i + 1, j, k + 1], nid[i + 1, j + 1, k + 1], nid[i, j + 1, k + 1]], 1)
    used = np.unique(h)
    remap = -np.ones(len(nodes), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[h]


def cubic_mesh(rho, n_t, L=L_DEFAULT, ratio=1.3, hmax_factor=6.0):
    t = wall_thickness('cubic', rho, L)
    x, (a, b) = wall_axis(L, t, n_t, ratio, hmax_factor * t / n_t)
    inside = lambda i: (i >= a) & (i < b)
    nodes, h = tensor_hex(x, x, x, lambda I, J, K: inside(I) | inside(J) | inside(K))
    return nodes, h, (L, L, L)


def grid_mesh(rho, n_t, n_ext, L=L_DEFAULT, ratio=1.3, hmax_factor=6.0):
    """FE frame: extrusion along x, section in (y, z)."""
    t = wall_thickness('grid', rho, L)
    s, (a, b) = wall_axis(L, t, n_t, ratio, hmax_factor * t / n_t)
    xe = np.linspace(0.0, L, n_ext + 1)
    inside = lambda i: (i >= a) & (i < b)
    nodes, h = tensor_hex(xe, s, s, lambda I, J, K: inside(J) | inside(K))
    return nodes, h, (L, L, L)


# ---------------------------------------------------------------------------------------------
# triangles, honeycomb: periodic 2D quadrilaterals (gmsh) extruded along x
# ---------------------------------------------------------------------------------------------
GMSH_LOCK = threading.Lock()


def section_quads(name, rho, h, L=L_DEFAULT):
    """All-quadrilateral periodic mesh of the section in CAD (x, y); returns (points (n, 2), quads (m, 4))."""
    with GMSH_LOCK:                                     # gmsh keeps one global model: one caller at a time
        return _section_quads(name, rho, h, L)


def _section_quads(name, rho, h, L):
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize(interruptible=False)            # no SIGINT handler: works outside the main thread
    gmsh.option.setNumber('General.Verbosity', 1)
    gmsh.model.add(f'{name}_section')
    occ = gmsh.model.occ
    t = wall_thickness(name, rho, L)
    W, D, _ = cad_box(name, L)
    rects = []
    for x0, y0, x1, y1 in wall_segments(name):
        x0, y0, x1, y1 = x0 * L, y0 * L, x1 * L, y1 * L
        r = occ.addRectangle(0, -t / 2, 0, np.hypot(x1 - x0, y1 - y0), t)
        occ.rotate([(2, r)], 0, 0, 0, 0, 0, 1, np.arctan2(y1 - y0, x1 - x0))
        occ.translate([(2, r)], x0, y0, 0)
        rects.append((2, r))
    fused, _ = occ.fuse(rects[:1], rects[1:])
    box = occ.addRectangle(0, 0, 0, W, D)
    occ.intersect(fused, [(2, box)])
    occ.synchronize()
    eps = 1e-6 * L
    # periodic curves: x = 0 -> x = W, y = 0 -> y = D
    for axis, shift in ((0, W), (1, D)):
        lo = gmsh.model.getEntitiesInBoundingBox(*(np.array([-eps, -eps, -eps, W + eps, D + eps, eps])
                                                   - np.array([0, 0, 0, W if axis == 0 else 0,
                                                               D if axis == 1 else 0, 0])), 1)
        for _, c in lo:
            bb = np.array(gmsh.model.getBoundingBox(1, c))
            tr = np.zeros(3)
            tr[axis] = shift
            match = gmsh.model.getEntitiesInBoundingBox(*(bb[:3] + tr - eps), *(bb[3:] + tr + eps), 1)
            if len(match) != 1:
                raise RuntimeError(f'{name}: no periodic partner for curve {c}')
            T = [1, 0, 0, tr[0], 0, 1, 0, tr[1], 0, 0, 1, tr[2], 0, 0, 0, 1]
            gmsh.model.mesh.setPeriodic(1, [match[0][1]], [c], T)
    # size 2h (halved by the subdivision into quads), refined towards the corners of the section
    # (wall junctions: re-entrant corners) down to 2h/3 within one wall thickness
    pts = [tg for _, tg in gmsh.model.getEntities(0)]
    fd = gmsh.model.mesh.field.add('Distance')
    gmsh.model.mesh.field.setNumbers(fd, 'PointsList', pts)
    ft = gmsh.model.mesh.field.add('Threshold')
    gmsh.model.mesh.field.setNumber(ft, 'InField', fd)
    gmsh.model.mesh.field.setNumber(ft, 'SizeMin', 2 * h / 3)
    gmsh.model.mesh.field.setNumber(ft, 'SizeMax', 2 * h)
    gmsh.model.mesh.field.setNumber(ft, 'DistMin', 0.0)
    gmsh.model.mesh.field.setNumber(ft, 'DistMax', t)
    gmsh.model.mesh.field.setAsBackgroundMesh(ft)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.Algorithm', 8)
    gmsh.option.setNumber('Mesh.RecombineAll', 1)
    gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
    gmsh.option.setNumber('Mesh.SubdivisionAlgorithm', 1)
    gmsh.model.mesh.generate(2)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)[:, :2]
    pos = {tg: i for i, tg in enumerate(tags)}
    et, _, en = gmsh.model.mesh.getElements(2)
    quads = []
    for typ, conn in zip(et, en):
        if typ != 3:
            raise RuntimeError(f'{name}: non-quadrilateral element type {typ} in the section mesh')
        quads.append(np.vectorize(pos.get)(conn.reshape(-1, 4)))
    gmsh.model.remove()
    quads = np.vstack(quads)
    used = np.unique(quads)
    remap = -np.ones(len(xyz), dtype=np.int64)
    remap[used] = np.arange(len(used))
    pts, quads = xyz[used], remap[quads]
    # counter-clockwise
    p = pts[quads]
    area = 0.5 * np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, 1) - np.roll(p[:, :, 0], -1, 1) * p[:, :, 1], 1)
    quads[area < 0] = quads[area < 0][:, ::-1]
    return pts, quads


def extrude(pts, quads, H, n_ext):
    """Section (CAD x, y) extruded along CAD z -> hexahedra in the FE frame (x_fe = z_cad)."""
    n = len(pts)
    zs = np.linspace(0.0, H, n_ext + 1)
    nodes = np.vstack([np.column_stack([np.full(n, z), pts[:, 0], pts[:, 1]]) for z in zs])
    h = np.vstack([np.hstack([quads + k * n, quads + (k + 1) * n]) for k in range(n_ext)])
    return nodes, h


# ---------------------------------------------------------------------------------------------
# gyroid: marching tetrahedra on a periodic Freudenthal grid
# ---------------------------------------------------------------------------------------------
FREUDENTHAL = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
# cube corner c = i + 2 j + 4 k  (bit 0: x, bit 1: y, bit 2: z)


def _split_prism(p, gid):
    """Prism p = (a0, a1, a2, b0, b1, b2) (ai -- bi are the lateral edges) into 3 tetrahedra whose
    quadrilateral diagonals start at the smallest global id (Dompierre et al. 1999)."""
    rot = {0: (0, 1, 2, 3, 4, 5), 1: (1, 2, 0, 4, 5, 3), 2: (2, 0, 1, 5, 3, 4),
           3: (3, 5, 4, 0, 2, 1), 4: (4, 3, 5, 1, 0, 2), 5: (5, 4, 3, 2, 1, 0)}
    g = gid[p]                                              # (m, 6)
    imin = np.argmin(g, 1)
    out = []
    for r, perm in rot.items():
        sel = imin == r
        if not sel.any():
            continue
        q = p[sel][:, list(perm)]
        gq = gid[q]
        # quad (1, 2, 5, 4): diagonal 1-5 if min(g1, g5) < min(g2, g4)
        d15 = np.minimum(gq[:, 1], gq[:, 5]) < np.minimum(gq[:, 2], gq[:, 4])
        a = q[d15]
        out += [a[:, [0, 1, 2, 5]], a[:, [0, 1, 5, 4]], a[:, [0, 4, 5, 3]]]
        b = q[~d15]
        out += [b[:, [0, 1, 2, 4]], b[:, [0, 4, 2, 5]], b[:, [0, 4, 5, 3]]]
    return np.vstack(out) if out else np.zeros((0, 4), dtype=np.int64)


def _split_pyramid(p, gid):
    """Pyramid p = (q0, q1, q2, q3, apex), quad base in cyclic order -> 2 tetrahedra, base diagonal
    from the smallest global id."""
    g = gid[p[:, :4]]
    d02 = np.minimum(g[:, 0], g[:, 2]) < np.minimum(g[:, 1], g[:, 3])
    a, b = p[d02], p[~d02]
    return np.vstack([a[:, [0, 1, 2, 4]], a[:, [0, 2, 3, 4]], b[:, [0, 1, 3, 4]], b[:, [1, 2, 3, 4]]])


def gyroid_mesh(rho, h, L=L_DEFAULT, snap=0.2, c=None):
    c = gyroid_level(rho) if c is None else c
    N = int(np.ceil(L / h))
    g1 = np.linspace(0.0, L, N + 1)
    X, Y, Z = np.meshgrid(g1, g1, g1, indexing='ij')
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    nid = np.arange((N + 1) ** 3).reshape(N + 1, N + 1, N + 1)
    ii, jj, kk = [a.ravel() for a in np.meshgrid(np.arange(N + 1), np.arange(N + 1), np.arange(N + 1), indexing='ij')]
    wrap = (ii % N) * N * N + (jj % N) * N + (kk % N)
    # level-set values on the periodic grid, copied to the images: bit-identical on opposite faces
    gw = g1[:N]
    Xw, Yw, Zw = np.meshgrid(gw, gw, gw, indexing='ij')
    phi = (c - np.abs(gyroid_g(Xw, Yw, Zw, L))).ravel()[wrap]
    I, J, K = [a.ravel() for a in np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')]
    corner = np.stack([nid[I + (cc & 1), J + ((cc >> 1) & 1), K + ((cc >> 2) & 1)] for cc in range(8)], 1)
    tets = np.vstack([corner[:, list(t)] for t in FREUDENTHAL])
    # snap: a node whose cut point on some edge lies within `snap` (fraction of the edge) gets phi = 0,
    # so the surface passes through the node instead of leaving a sliver. Decided on the periodic
    # (wrapped) grid, so a node and its periodic images are snapped alike.
    e = np.vstack([tets[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    a, b = e[:, 0], e[:, 1]
    sc = phi[a] * phi[b] < 0
    a, b = a[sc], b[sc]
    fa = np.abs(phi[a]) / (np.abs(phi[a]) + np.abs(phi[b]))
    snapped_w = np.unique(np.concatenate([wrap[a[fa < snap]], wrap[b[(1 - fa) < snap]]]))
    # never snap nodes on the cell faces: a face node at phi = 0 can be touched by the solid from one
    # side only and would then have no periodic partner
    wi, wj, wk = snapped_w // (N * N), (snapped_w // N) % N, snapped_w % N
    snapped_w = snapped_w[(wi != 0) & (wj != 0) & (wk != 0)]
    phi[np.isin(wrap, snapped_w)] = 0.0
    s = phi[tets]
    pos, neg = (s > 0).sum(1), (s < 0).sum(1)
    keep_whole = tets[(neg == 0) & (pos > 0)]
    mixed = tets[(neg > 0) & (pos > 0)]
    # cut points on the edges with a sign change
    ms = phi[mixed]
    new_pts, cut_index = [], {}
    base = len(nodes)

    def cut(a, b):                                           # vectorized: arrays of node ids
        key = np.minimum(a, b).astype(np.int64) * (base + 1) + np.maximum(a, b)
        return key

    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = mixed[:, i], mixed[:, j]
            sc = (phi[a] * phi[b]) < 0
            pairs.append(cut(a[sc], b[sc]))
    keys = np.unique(np.concatenate(pairs))
    ka, kb = keys // (base + 1), keys % (base + 1)
    # cut point: root of |g| = c on the edge (regula falsi, Illinois), from the linear guess
    xa, xb = nodes[ka], nodes[kb]
    gb_sign = np.sign(gyroid_g(*xb.T, L))
    ga_sign = np.sign(gyroid_g(*xa.T, L))
    target_sign = np.where(phi[ka] < 0, ga_sign, gb_sign)      # sign of g on the outside end
    f = lambda w: (gyroid_g(*(xa + w[:, None] * (xb - xa)).T, L) * target_sign) - c
    lo, hi = np.zeros(len(keys)), np.ones(len(keys))
    flo, fhi = f(lo), f(hi)
    w = flo / (flo - fhi)
    for _ in range(30):
        fw = f(w)
        left = np.sign(fw) == np.sign(flo)
        lo = np.where(left, w, lo); flo = np.where(left, fw, flo * 0.5)
        hi = np.where(left, hi, w); fhi = np.where(left, fhi * 0.5, fw)
        w = np.clip(lo - flo * (hi - lo) / (fhi - flo), 0.0, 1.0)
    # cut points closer than `snap` to a face node (which is never snapped) are held at that distance,
    # so no sliver forms at the cell faces; identical on periodic images
    on_face = (ii == 0) | (ii == N) | (jj == 0) | (jj == N) | (kk == 0) | (kk == N)
    w = np.where(on_face[ka] & (w < snap), snap, w)
    w = np.where(on_face[kb] & (w > 1 - snap), 1 - snap, w)
    cut_xyz = xa + w[:, None] * (xb - xa)
    all_nodes = np.vstack([nodes, cut_xyz])
    cid = lambda a, b: base + np.searchsorted(keys, cut(a, b))
    gid = np.arange(len(all_nodes))                          # global ids for the diagonal rule

    out = [keep_whole]
    # sort each mixed tet's vertices into positive (P), zero (Z), negative (M)
    order = np.argsort(-np.sign(ms), axis=1, kind='stable')  # positives first, then zeros, then negatives
    v = np.take_along_axis(mixed, order, 1)
    P, Mn = (ms > 0).sum(1), (ms < 0).sum(1)
    Zc = 4 - P - Mn
    for p_, m_ in [(1, 3), (1, 2), (1, 1), (2, 2), (2, 1), (3, 1)]:
        z_ = 4 - p_ - m_
        sel = (P == p_) & (Mn == m_)
        if not sel.any():
            continue
        q = v[sel]
        ps, zs, ns = q[:, :p_], q[:, p_:p_ + z_], q[:, p_ + z_:]
        if p_ == 1:                                          # a tetrahedron: p, zeros, cuts p--n
            verts = [ps[:, 0]] + [zs[:, k] for k in range(z_)] + [cid(ps[:, 0], ns[:, k]) for k in range(m_)]
            out.append(np.stack(verts, 1))
        elif p_ == 2 and m_ == 2:                            # prism: (p0, c00, c01) -- (p1, c10, c11)
            pr = np.stack([ps[:, 0], cid(ps[:, 0], ns[:, 0]), cid(ps[:, 0], ns[:, 1]),
                           ps[:, 1], cid(ps[:, 1], ns[:, 0]), cid(ps[:, 1], ns[:, 1])], 1)
            out.append(_split_prism(pr, gid))
        elif p_ == 2 and m_ == 1:                            # pyramid: base (p0, p1, c1, c0), apex z
            py = np.stack([ps[:, 0], ps[:, 1], cid(ps[:, 1], ns[:, 0]), cid(ps[:, 0], ns[:, 0]), zs[:, 0]], 1)
            out.append(_split_pyramid(py, gid))
        elif p_ == 3:                                        # prism: (p0, p1, p2) -- (c0, c1, c2)
            pr = np.stack([ps[:, 0], ps[:, 1], ps[:, 2],
                           cid(ps[:, 0], ns[:, 0]), cid(ps[:, 1], ns[:, 0]), cid(ps[:, 2], ns[:, 0])], 1)
            out.append(_split_prism(pr, gid))
    tets = np.vstack(out)
    vol = tet_volumes(all_nodes, tets)
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    vol = np.abs(vol)
    tets = tets[vol > 1e-12 * h ** 3]
    used = np.unique(tets)
    remap = -np.ones(len(all_nodes), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return all_nodes[used], remap[tets], (L, L, L)


def gyroid_mesh_exact(rho, h, L=L_DEFAULT, tol=2e-4, snap=0.2):
    """gyroid_mesh with the level c corrected so that the meshed volume fraction is rho
    (secant steps, then bisection if the secant does not settle: snapping makes vf(c) slightly jumpy)."""
    vf = lambda m: np.abs(tet_volumes(m[0], m[1])).sum() / L ** 3
    c0 = gyroid_level(rho)
    m0 = gyroid_mesh(rho, h, L, c=c0, snap=snap)
    r0 = vf(m0)
    if abs(r0 - rho) < tol:
        return (*m0, c0)
    c1 = c0 * (1 + 0.5 * (rho - r0) / rho)
    m1 = gyroid_mesh(rho, h, L, c=c1, snap=snap)
    r1 = vf(m1)
    for _ in range(6):
        if abs(r1 - rho) < tol or r1 == r0:
            break
        c0, r0, c1 = c1, r1, c1 + (rho - r1) * (c1 - c0) / (r1 - r0)
        m1 = gyroid_mesh(rho, h, L, c=c1, snap=snap)
        r1 = vf(m1)
    if abs(r1 - rho) < tol:
        return (*m1, c1)
    lo, hi = gyroid_level(rho) * 0.85, gyroid_level(rho) * 1.15        # vf increases with c
    best = (abs(r1 - rho), m1, c1)
    for _ in range(30):
        c = 0.5 * (lo + hi)
        m = gyroid_mesh(rho, h, L, c=c, snap=snap)
        r = vf(m)
        if abs(r - rho) < best[0]:
            best = (abs(r - rho), m, c)
        if abs(r - rho) < tol:
            break
        lo, hi = (c, hi) if r < rho else (lo, c)
    return (*best[1], best[2])


# ---------------------------------------------------------------------------------------------
def build_mesh(name, rho, n_t=2, n_ext=4, L=L_DEFAULT):
    """Periodic quadratic mesh of one cell. n_t: elements across the thinnest wall."""
    tmin = smallest_feature(name, rho, L)
    h = tmin / n_t
    if name == 'cubic':
        nodes, h8, box = cubic_mesh(rho, n_t, L)
    elif name == 'grid':
        nodes, h8, box = grid_mesh(rho, n_t, n_ext, L)
    elif name in ('triangles', 'honeycomb'):
        pts, quads = section_quads(name, rho, h, L)
        nodes, h8 = extrude(pts, quads, cad_box(name, L)[2], n_ext)
        box = fe_box(name, L)
    elif name == 'gyroid':
        best = None
        for snap in (0.2, 0.17, 0.23, 0.14, 0.26):          # vf(c) jumps on coarse grids: try a few snaps
            nodes, t4, box, c = gyroid_mesh_exact(rho, h, L, snap=snap)
            err = abs(np.abs(tet_volumes(nodes, t4)).sum() / np.prod(box) - rho)
            if best is None or err < best[0]:
                best = (err, nodes, t4, box, c)
            if err < 2e-4:
                break
        _, nodes, t4, box, c = best
        q_nodes, t10 = make_quadratic(nodes, t4, TET_EDGES)
        return Mesh(q_nodes, t10, 'C3D10', box, dict(pattern=name, rho=rho, n_t=n_t, h=h, t_min=tmin,
                                                     level_c=c, level_c_exact=gyroid_level(rho)))
    else:
        raise ValueError(name)
    h8 = fix_hex_orientation(nodes, h8)
    q_nodes, h20 = make_quadratic(nodes, h8, HEX_EDGES)
    return Mesh(q_nodes, h20, 'C3D20R', box, dict(pattern=name, rho=rho, n_t=n_t, n_ext=n_ext, h=h, t_min=tmin))


def quality(m):
    """Element quality summary: tets -> radius ratio (1 = regular), hexes -> scaled Jacobian at corners."""
    if m.etype == 'C3D10':
        x = m.nodes[m.elements[:, :4]]
        V = np.abs(tet_volumes(m.nodes, m.elements))
        faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        A = sum(0.5 * np.linalg.norm(np.cross(x[:, b] - x[:, a], x[:, c] - x[:, a]), axis=1) for a, b, c in faces)
        r_in = 3 * V / A
        e = [np.linalg.norm(x[:, j] - x[:, i], axis=1) for i, j in TET_EDGES]
        # circumradius via formula R = sqrt((aA+bB+cC)(aA+bB-cC)(aA-bB+cC)(-aA+bB+cC)) / (24 V)
        a, b, c = e[0] * e[5], e[1] * e[3], e[2] * e[4]
        R = np.sqrt(np.abs((a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c))) / (24 * V)
        q = 3 * r_in / R
        return dict(min=float(q.min()), p1=float(np.percentile(q, 1)), median=float(np.median(q)),
                    below_0_1=int((q < 0.1).sum()), n=len(q))
    x = m.nodes[m.elements[:, :8]]
    nb = {0: (1, 3, 4), 1: (2, 0, 5), 2: (3, 1, 6), 3: (0, 2, 7), 4: (7, 5, 0), 5: (4, 6, 1), 6: (5, 7, 2), 7: (6, 4, 3)}
    sj = []
    for k, (a, b, c) in nb.items():
        u, v, w = x[:, a] - x[:, k], x[:, b] - x[:, k], x[:, c] - x[:, k]
        d = np.einsum('ij,ij->i', np.cross(u, v), w)
        sj.append(d / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) * np.linalg.norm(w, axis=1)))
    sj = np.min(sj, 0)
    return dict(min=float(sj.min()), p1=float(np.percentile(sj, 1)), median=float(np.median(sj)),
                below_0_2=int((sj < 0.2).sum()), n=len(sj))
