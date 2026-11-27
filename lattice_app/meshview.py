"""Mesh preview without solving: build the periodic mesh, return its outer surface (linear faces),
the element edges on that surface and quality numbers. Used by the mesh tab to show what n_t does."""
import base64
import time

import numpy as np

from lattice import mesh
from lattice.fields import HEX_FACES, TET_FACES
from lattice.geometry import PRISMATIC, fe_to_cad, smallest_feature

_cache = {}


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode('ascii')


def preview(pattern, rho, n_t, n_ext=4):
    key = (pattern, round(rho, 4), round(n_t, 3), n_ext)
    if key in _cache:
        return _cache[key]
    t0 = time.time()
    m = mesh.build_mesh(pattern, rho, n_t=n_t, n_ext=n_ext)
    t_build = time.time() - t0
    faces = HEX_FACES if m.etype.startswith('C3D20') else TET_FACES
    corners = np.vstack([m.elements[:, list(c)] for c, _ in faces])
    k = np.sort(corners, 1)
    _, idx, cnt = np.unique(k, axis=0, return_index=True, return_counts=True)
    outer = corners[idx[cnt == 1]]
    if outer.shape[1] == 4:
        tris = np.vstack([outer[:, [0, 1, 2]], outer[:, [0, 2, 3]]])
        edges = np.vstack([outer[:, [0, 1]], outer[:, [1, 2]], outer[:, [2, 3]], outer[:, [3, 0]]])
    else:
        tris = outer
        edges = np.vstack([outer[:, [0, 1]], outer[:, [1, 2]], outer[:, [2, 0]]])
    edges = np.unique(np.sort(edges, 1), axis=0)
    used, inv = np.unique(np.concatenate([tris.ravel(), edges.ravel()]), return_inverse=True)
    tris = inv[:tris.size].reshape(tris.shape)
    edges = inv[tris.size:].reshape(edges.shape)
    X = m.nodes[used]
    box = np.array(m.box, float)
    if pattern in PRISMATIC:
        X, box = fe_to_cad(X, pattern), fe_to_cad(box, pattern)
    q = mesh.quality(m)
    vols = mesh.element_volumes(m)
    out = dict(pattern=pattern, rho=rho, n_t=n_t, etype=m.etype, nodes=len(m.nodes), elements=len(m.elements),
               h=float(m.info['h']), t_min=float(smallest_feature(pattern, rho)),
               volume_fraction=float(m.volume_fraction), box=box.tolist(), build_s=round(t_build, 2),
               quality={k: (float(v) if np.ndim(v) == 0 else v) for k, v in q.items()},
               size=dict(min=float(np.cbrt(vols.min())), median=float(np.cbrt(np.median(vols))),
                         max=float(np.cbrt(vols.max()))),
               X=_b64(X, np.float32), T=_b64(tris, np.uint32), Ed=_b64(edges, np.uint32),
               dof_estimate=3 * len(m.nodes))
    if len(_cache) > 12:
        _cache.pop(next(iter(_cache)))
    _cache[key] = out
    return out
