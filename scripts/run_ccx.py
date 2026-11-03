"""Build a mesh + CalculiX deck for one cell, run ccx, print C and the virtual tests.
python scripts/run_ccx.py honeycomb 0.3 --nt 2 [--next 4] [--out out/ccx]"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice import inp, mesh, results            # noqa: E402


def run(name, rho, n_t, n_ext=4, out='out/ccx', field=True, threads=2, quiet=False, tests=True):
    os.makedirs(out, exist_ok=True)
    tag = f'{name}_rho{int(round(rho * 100)):02d}_nt{("%g" % n_t).replace(".", "p")}'
    t0 = time.time()
    m = mesh.build_mesh(name, rho, n_t=n_t, n_ext=n_ext)
    path = os.path.join(out, tag + '.inp')
    meta = inp.write_deck(m, path, 'ccx', field_output=field, tests=None if tests else {})
    np.savez_compressed(path[:-4] + '_mesh.npz', nodes=m.nodes, elements=m.elements, etype=m.etype, box=m.box)
    t1 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), CCX_NPROC_EQUATION_SOLVER=str(threads))
    r = subprocess.run(['ccx', '-i', path[:-4]], capture_output=True, text=True, env=env)
    t2 = time.time()
    if 'Job finished' not in r.stdout:
        raise RuntimeError(r.stdout[-3000:])
    st = results.parse_ccx_dat(path[:-4] + '.dat', meta, m.nodes)
    C, chk = results.stiffness(meta, st)
    vt = results.virtual_test_results(meta, st)
    res = dict(pattern=name, rho=rho, n_t=n_t, n_ext=n_ext, nodes=len(m.nodes), elements=len(m.elements),
               etype=m.etype, volume_fraction=meta['volume_fraction'], C=C.tolist(), checks=chk,
               tests={k: dict(sigma=v['sigma_voigt'].tolist(), eps=v['eps_voigt'].tolist()) for k, v in vt.items()},
               time_mesh=t1 - t0, time_solve=t2 - t1, quality=mesh.quality(m))
    json.dump(res, open(path[:-4] + '_result.json', 'w'), indent=1)
    if not quiet:
        np.set_printoptions(precision=2, suppress=True, linewidth=140)
        print(f'{tag}: {len(m.nodes)} nodes, {len(m.elements)} {m.etype}, vf {meta["volume_fraction"]:.4f}, '
              f'mesh {t1 - t0:.1f} s, solve {t2 - t1:.1f} s, checks {chk}')
        print(C)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pattern')
    ap.add_argument('rho', type=float)
    ap.add_argument('--nt', type=float, default=2)
    ap.add_argument('--next', type=int, default=4)
    ap.add_argument('--out', default='out/ccx')
    a = ap.parse_args()
    run(a.pattern, a.rho, a.nt, a.next, a.out)
