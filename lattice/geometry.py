"""The five unit cells: sizes, frames and wall thickness for a relative density rho.

CAD frame (STL/STEP, ready for a slicer): x, y in the print plane, z = build direction.
    grid       L x L x L          two wall families (along x and y), walls centred in the cell
    triangles  L x sqrt3 L x L    three wall families at 0/60/120 deg, triangle side L
    honeycomb  L x sqrt3 L x L    regular hexagons, flat-to-flat distance L
    cubic      L x L x L          three orthogonal plates, centred (= SC plate lattice, Schneider & Kumar Eq. 4)
    gyroid     L x L x L          sheet gyroid |g| <= c, g of Schneider & Kumar Eq. 1
The prismatic cells (grid, triangles, honeycomb) are extruded along z.

FE frame (meshes, input decks): x = extrusion axis of the prismatic cells, as the fiber axis of the HD
repository, so (x, y, z)_fe = (z, x, y)_cad. Cubic and gyroid use the CAD frame.

Wall thickness t (all in units of L, walls centred on the ideal lines):
    grid       rho = 1 - (1 - t)^2
    honeycomb  rho = 1 - (1 - t)^2              (inner hexagon flat-to-flat 1 - t)
    triangles  rho = 1 - (1 - sqrt3 t)^2        (inner triangle inradius 1/(2 sqrt3) - t/2)
    cubic      rho = 1 - (1 - t)^3
    gyroid     c = rho-quantile of |g| over the cell (no closed form)
"""
import numpy as np

S3 = np.sqrt(3.0)
L_DEFAULT = 20.0                       # mm
RHOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
PATTERNS = ('grid', 'triangles', 'honeycomb', 'cubic', 'gyroid')
PRISMATIC = ('grid', 'triangles', 'honeycomb')


def cad_box(name, L=L_DEFAULT):
    """Cell dimensions (x, y, z) in the CAD frame."""
    if name in ('triangles', 'honeycomb'):
        return (L, S3 * L, L)
    return (L, L, L)


def cad_to_fe(xyz, name):
    """CAD coordinates -> FE coordinates (prismatic: x_fe = z_cad)."""
    xyz = np.asarray(xyz, dtype=float)
    return xyz[..., [2, 0, 1]] if name in PRISMATIC else xyz


def fe_to_cad(xyz, name):
    """FE coordinates -> CAD coordinates (inverse of cad_to_fe)."""
    xyz = np.asarray(xyz, dtype=float)
    return xyz[..., [1, 2, 0]] if name in PRISMATIC else xyz


def fe_box(name, L=L_DEFAULT):
    b = cad_box(name, L)
    return (b[2], b[0], b[1]) if name in PRISMATIC else b


def wall_thickness(name, rho, L=L_DEFAULT):
    """Wall (plate) thickness in mm. Gyroid: mean sheet thickness rho V / A_mid (A_mid = 3.0919 L^2)."""
    rho = float(rho)
    if name in ('grid', 'honeycomb'):
        t = 1.0 - np.sqrt(1.0 - rho)
    elif name == 'triangles':
        t = (1.0 - np.sqrt(1.0 - rho)) / S3
    elif name == 'cubic':
        t = 1.0 - (1.0 - rho) ** (1.0 / 3.0)
    elif name == 'gyroid':
        t = rho / 3.0919
    else:
        raise ValueError(name)
    return float(t * L)


def density(name, t, L=L_DEFAULT):
    """Inverse of wall_thickness for the walled cells (t in mm)."""
    s = t / L
    if name in ('grid', 'honeycomb'):
        return 1.0 - (1.0 - s) ** 2
    if name == 'triangles':
        return 1.0 - (1.0 - S3 * s) ** 2
    if name == 'cubic':
        return 1.0 - (1.0 - s) ** 3
    raise ValueError(name)


def smallest_feature(name, rho, L=L_DEFAULT):
    """Thinnest solid feature in mm, used for mesh sizes: the wall, or for the gyroid the
    thinner of sheet and channel (sheet thickness of the complementary density)."""
    if name == 'gyroid':
        return min(wall_thickness('gyroid', rho, L), wall_thickness('gyroid', 1.0 - rho, L))
    return wall_thickness(name, rho, L)


def gyroid_g(x, y, z, L=L_DEFAULT):
    """Schneider & Kumar (2023) Eq. 1: cos(kx) sin(ky) + cos(ky) sin(kz) + cos(kz) sin(kx), k = 2 pi / L."""
    k = 2.0 * np.pi / L
    return (np.cos(k * x) * np.sin(k * y) + np.cos(k * y) * np.sin(k * z)
            + np.cos(k * z) * np.sin(k * x))


_GYROID_SAMPLES = None


def gyroid_level(rho, N=200):
    """Level c with vol{|g| <= c} = rho, from |g| at N^3 cell-centred points (midpoint rule)."""
    global _GYROID_SAMPLES
    if _GYROID_SAMPLES is None or _GYROID_SAMPLES[0] != N:
        c = (np.arange(N) + 0.5) / N
        X, Y = np.meshgrid(c, c, indexing='ij')
        vals = [np.abs(gyroid_g(X, Y, np.full_like(X, z), 1.0)).ravel() for z in c]
        _GYROID_SAMPLES = (N, np.sort(np.concatenate(vals)))
    s = _GYROID_SAMPLES[1]
    return float(s[min(len(s) - 1, int(round(rho * len(s))))])


def wall_segments(name):
    """Centre lines of the walls of a prismatic cell, unit cell size, CAD (x, y), with enough
    periodic images to cover the cell after clipping."""
    if name == 'grid':
        return [(-0.5, 0.5, 1.5, 0.5), (0.5, -0.5, 0.5, 1.5)]
    if name == 'triangles':                             # shifted by sqrt3/4 so no wall lies on the cell boundary
        d = S3 / 4
        s = [(-1.0, y + d, 2.0, y + d) for y in (-S3 / 2, 0.0, S3 / 2, S3)]
        for c in range(-2, 4):                          # slopes +-sqrt3 through (c, d)
            s.append((c - 1.5, d - S3 * 1.5, c + 1.5, d + S3 * 1.5))
            s.append((c + 1.5, d - S3 * 1.5, c - 1.5, d + S3 * 1.5))
        return s
    if name == 'honeycomb':                             # Voronoi cells of a triangular lattice, spacing 1
        pts = [(i + 0.5 * (j % 2), j * S3 / 2) for i in range(-1, 3) for j in range(-1, 4)]
        s = []
        for px, py in pts:
            v = [(px + np.cos(np.pi / 6 + k * np.pi / 3) / S3, py + np.sin(np.pi / 6 + k * np.pi / 3) / S3)
                 for k in range(6)]
            s += [(v[k][0], v[k][1], v[(k + 1) % 6][0], v[(k + 1) % 6][1]) for k in range(6)]
        return s
    raise ValueError(name)
