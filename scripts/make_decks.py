"""Abaqus (and optionally CalculiX) decks for the study.

    python scripts/make_decks.py production                   # 5 patterns x 7 rho at the production level
    python scripts/make_decks.py convergence                  # 5 patterns x rho 0.1, 0.4 x 4 levels
    python scripts/make_decks.py one gyroid 0.3 --nt 2 [--flavour ccx]

Files: out/decks/<set>/<pattern>_rho<NN>_nt<k>.inp + .json (+ run_all.bat / run_all.sh).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice import inp, mesh                                    # noqa: E402
from lattice.geometry import PATTERNS, RHOS                      # noqa: E402

LEVELS = {'hex': (2, 3, 4, 6), 'gyroid': (1.5, 2, 2.5, 3)}
# production mesh level per pattern: coarsest level within about 1 % of the mesh-converged stiffness
# (CalculiX study at rho = 0.1 and 0.4, out/post/convergence.json); grid in-plane shear converges slowest
PRODUCTION_NT = {'grid': 8, 'triangles': 3, 'honeycomb': 3, 'cubic': 3, 'gyroid': 2}


def tag(name, rho, nt):
    return f'{name}_rho{int(round(rho * 100)):02d}_nt{("%g" % nt).replace(".", "p")}'


def make(name, rho, nt, out, flavour='abaqus', n_ext=4):
    m = mesh.build_mesh(name, rho, n_t=nt, n_ext=n_ext)
    t = tag(name, rho, nt)
    meta = inp.write_deck(m, os.path.join(out, t + '.inp'), flavour)
    print(f'{t}: {meta["nodes"]} nodes, {meta["elements"]} {meta["etype"]}, {meta["equations"]} equations, '
          f'vf {meta["volume_fraction"]:.4f}', flush=True)
    return t


def scripts(out, jobs, cpus=4):
    import shutil
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'abaqus_extract.py'), out)
    with open(os.path.join(out, 'run_all.bat'), 'w') as f:
        f.write('@echo off\r\nrem Runs every deck in this folder one after the other, then extracts the results.\r\n')
        for j in jobs:
            f.write(f'call abaqus job={j} cpus={cpus} interactive\r\n')
            f.write(f'call abaqus python abaqus_extract.py {j}.odb --fields\r\n')
    with open(os.path.join(out, 'run_all.sh'), 'w') as f:
        f.write('#!/bin/sh\n# Runs every deck in this folder one after the other, then extracts the results.\n')
        for j in jobs:
            f.write(f'abaqus job={j} cpus={cpus} interactive\n')
            f.write(f'abaqus python abaqus_extract.py {j}.odb --fields\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('set', choices=['production', 'convergence', 'one'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--nt', type=float, default=None)
    ap.add_argument('--flavour', default='abaqus')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join('out', 'decks', a.set)
    os.makedirs(out, exist_ok=True)
    jobs = []
    if a.set == 'one':
        jobs.append(make(a.args[0], float(a.args[1]), a.nt or 2, out, a.flavour))
    elif a.set == 'production':
        for name in PATTERNS:
            for rho in RHOS:
                nt = a.nt if a.nt is not None else PRODUCTION_NT[name]
                jobs.append(make(name, rho, nt, out, a.flavour))
    else:
        for name in PATTERNS:
            for rho in (0.1, 0.4):
                for nt in LEVELS['gyroid' if name == 'gyroid' else 'hex']:
                    jobs.append(make(name, rho, nt, out, a.flavour))
    if a.flavour == 'abaqus':
        scripts(out, jobs)
