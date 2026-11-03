"""CAD unit cells in the CAD frame (mm): STEP + STL for the walled cells (OpenCASCADE through gmsh),
STL for the gyroid (marching cubes of c - |g|, faces closed on the cell boundary)."""
import os

import gmsh
import numpy as np
import trimesh
from skimage import measure

from .geometry import L_DEFAULT, cad_box, gyroid_g, gyroid_level, wall_segments, wall_thickness


def _init():
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', 1)


def _wall_box(x0, y0, x1, y1, t, H, L):
    occ = gmsh.model.occ
    x0, y0, x1, y1 = (c * L for c in (x0, y0, x1, y1))
    length = np.hypot(x1 - x0, y1 - y0)
    b = occ.addBox(0, -t / 2, 0, length, t, H)
    occ.rotate([(3, b)], 0, 0, 0, 0, 0, 1, np.arctan2(y1 - y0, x1 - x0))
    occ.translate([(3, b)], x0, y0, 0)
    return (3, b)


def build_occ(name, rho, L=L_DEFAULT):
    """Builds the solid in the current gmsh model; returns (entities, box, t)."""
    occ = gmsh.model.occ
    t = wall_thickness(name, rho, L)
    W, D, H = cad_box(name, L)
    if name == 'cubic':
        h = L / 2
        walls = [occ.addBox(h - t / 2, -L, -L, t, 3 * L, 3 * L), occ.addBox(-L, h - t / 2, -L, 3 * L, t, 3 * L),
                 occ.addBox(-L, -L, h - t / 2, 3 * L, 3 * L, t)]
        walls = [(3, w) for w in walls]
    else:
        walls = [_wall_box(*s, t, H, L) for s in wall_segments(name)]
    out, _ = occ.fuse(walls[:1], walls[1:])
    box = occ.addBox(0, 0, 0, W, D, H)
    res, _ = occ.intersect(out, [(3, box)])
    occ.synchronize()
    return res, (W, D, H), t


def walled_cell(name, rho, out_dir, L=L_DEFAULT, step=True):
    """Writes {name}_rho{NN}.step/.stl; returns a dict of checks."""
    _init()
    tag = f'{name}_rho{int(round(rho * 100)):02d}'
    gmsh.model.add(tag)
    res, box, t = build_occ(name, rho, L)
    vol = sum(gmsh.model.occ.getMass(*e) for e in res)
    if step:
        gmsh.write(os.path.join(out_dir, tag + '.step'))
    gmsh.option.setNumber('Mesh.MeshSizeMax', L / 8)
    gmsh.model.mesh.generate(2)
    stl = os.path.join(out_dir, tag + '.stl')
    gmsh.option.setNumber('Mesh.Binary', 1)
    gmsh.write(stl)
    gmsh.model.remove()
    m = trimesh.load(stl)
    V = float(np.prod(box))
    return dict(file=tag, t_mm=t, rho_cad=vol / V, rho_stl=m.volume / V, watertight=bool(m.is_watertight),
                triangles=len(m.faces), box_mm=box)


def gyroid_cell(rho, out_dir, L=L_DEFAULT, h=0.15):
    """Sheet gyroid |g| <= c by marching cubes (grid spacing h mm), closed on the cell faces."""
    c = gyroid_level(rho)
    N = int(np.ceil(L / h))
    g = np.linspace(0.0, L, N + 1)
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    f = c - np.abs(gyroid_g(X, Y, Z, L))                       # > 0 inside the sheet
    fp = np.pad(f, 1, constant_values=-1e3)                    # the cap falls on the face plane
    v, fa, _, _ = measure.marching_cubes(fp, 0.0, spacing=(L / N,) * 3)
    v = np.clip(v - L / N, 0.0, L)
    m = trimesh.Trimesh(v, fa[:, ::-1], process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    if m.volume < 0:
        m.invert()
    tag = f'gyroid_rho{int(round(rho * 100)):02d}'
    m.export(os.path.join(out_dir, tag + '.stl'))
    return dict(file=tag, level_c=c, t_mm=wall_thickness('gyroid', rho, L), rho_stl=m.volume / L ** 3,
                watertight=bool(m.is_watertight), triangles=len(m.faces), box_mm=(L, L, L))
