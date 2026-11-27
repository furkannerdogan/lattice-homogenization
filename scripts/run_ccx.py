"""Build a mesh + CalculiX deck for one cell, run ccx, print C (FE frame) and the checks.
python scripts/run_ccx.py honeycomb 0.3 --nt 2 [--next 4] [--out out/ccx] [--ccx PATH]
The app (python -m lattice_app) does the same through its job queue and keeps a registry; this script
is for quick runs without the app."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice import pipeline, solver            # noqa: E402


def run(name, rho, n_t, n_ext=4, out='out/ccx', field=True, threads=2, quiet=False, ccx=None, tests=True):
    exe = solver.find_ccx(ccx)
    if not exe:
        raise SystemExit('ccx not found: pass --ccx PATH or set LATTICE_CCX')
    res = pipeline.run_cell(name, rho, n_t, out, exe, n_ext=n_ext, threads=threads, field=field,
                            log=(lambda s: None) if quiet else print, tests=tests)
    if not quiet:
        np.set_printoptions(precision=2, suppress=True, linewidth=140)
        print(f"{res['nodes']} nodes, {res['elements']} {res['etype']}, vf {res['volume_fraction']:.4f}, "
              f"checks {res['checks']}")
        print(np.array(res['C']))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pattern')
    ap.add_argument('rho', type=float)
    ap.add_argument('--nt', type=float, default=2)
    ap.add_argument('--next', type=int, default=4)
    ap.add_argument('--out', default='out/ccx')
    ap.add_argument('--ccx')
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    run(a.pattern, a.rho, a.nt, a.next, a.out, threads=a.threads, ccx=a.ccx)
