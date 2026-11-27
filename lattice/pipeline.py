"""One cell, end to end: mesh -> CalculiX deck -> ccx -> C, virtual tests, quality.

Files in job_dir (tag = <pattern>_rho<NN>_nt<n>):
    tag.inp, tag.json      deck and step description (inp.write_deck)
    tag_mesh.npz           nodes, elements, etype, box (for the field viewer and VTU export)
    tag.dat, tag.frd       ccx output (frd only with field output)
    tag_result.json        everything the registry stores
"""
import json
import os
import time

import numpy as np

from . import inp, mesh, results, solver

MESH_VERSION = 1          # bump when mesh.py changes the meshes it builds (invalidates cached results)


def tag(pattern, rho, n_t):
    return f'{pattern}_rho{int(round(rho * 100)):02d}_nt{("%g" % n_t).replace(".", "p")}'


def build(pattern, rho, n_t, n_ext=4):
    return mesh.build_mesh(pattern, rho, n_t=n_t, n_ext=n_ext)


def run_cell(pattern, rho, n_t, job_dir, ccx, E=inp.E_PLA, nu=inp.NU_PLA, n_ext=4, threads=2, field=True,
             log=print, cancel=None, tests=True):
    """tests=False: only the six strain cases (C), no virtual tests."""
    os.makedirs(job_dir, exist_ok=True)
    name = tag(pattern, rho, n_t)
    path = os.path.join(job_dir, name + '.inp')
    t0 = time.time()
    log(f'mesh {pattern} rho={rho} n_t={n_t} ...')
    m = build(pattern, rho, n_t, n_ext)
    log(f'mesh: {len(m.nodes)} nodes, {len(m.elements)} {m.etype}, volume fraction {m.volume_fraction:.4f}')
    meta = inp.write_deck(m, path, 'ccx', E=E, nu=nu, field_output=field, tests=None if tests else {})
    np.savez_compressed(path[:-4] + '_mesh.npz', nodes=m.nodes, elements=m.elements, etype=m.etype,
                        box=np.array(m.box))
    t1 = time.time()
    log(f'deck written ({t1 - t0:.1f} s); running {os.path.basename(ccx)} with {threads} thread(s)')
    solver.run(ccx, job_dir, name, threads=threads, log=log, cancel=cancel)
    t2 = time.time()
    st = results.parse_ccx_dat(path[:-4] + '.dat', meta, m.nodes)
    C, chk = results.stiffness(meta, st)
    vt = results.virtual_test_results(meta, st)
    res = dict(pattern=pattern, rho=float(rho), n_t=float(n_t), n_ext=n_ext, L=float(m.info.get('L', 20.0)),
               h=float(m.info['h']), t_min=float(m.info['t_min']), nodes=len(m.nodes), elements=len(m.elements),
               etype=m.etype, volume_fraction=float(meta['volume_fraction']), E=E, nu=nu, C=C.tolist(),
               checks=chk, source='calculix', mesh_version=MESH_VERSION,
               tests={k: dict(sigma=v['sigma_voigt'].tolist(), eps=v['eps_voigt'].tolist()) for k, v in vt.items()},
               steps=[s['name'] for s in meta['steps']], time_mesh=t1 - t0, time_solve=t2 - t1,
               quality=mesh.quality(m), files=dict(inp=name + '.inp', frd=name + '.frd' if field else None,
                                                   mesh=name + '_mesh.npz'))
    with open(path[:-4] + '_result.json', 'w') as f:
        json.dump(res, f, indent=1, default=float)
    log(f'done: mesh {t1 - t0:.1f} s, solve {t2 - t1:.1f} s')
    return res
