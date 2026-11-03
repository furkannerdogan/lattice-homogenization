"""Input decks for Abaqus/Standard and CalculiX (flat input, no parts or assembly, units mm, N, MPa).

One deck per cell, density and mesh, with one step per load case:
    E11 ... G12          strain cases (all reference-point DOFs prescribed)      -> C
    UT_*, ET_*, SS_*     virtual tests (stress-controlled, other stresses zero)  -> UT/ET/SS responses, plots
Every step restates its boundary conditions and loads with OP=NEW, so the steps are independent
(linear, NLGEOM off). A JSON file next to the deck describes the steps for the post-processing.
"""
import json
import os

import numpy as np

from . import loadcases as lc
from .pbc import BoxCell, equations, pin_node

E_PLA, NU_PLA = 2100.0, 0.35      # MPa; Pszczolkowski et al. (2022) static tensile test of FDM PLA; nu assumed


def _lines(arr, fmt, per_line=16):
    out = []
    for row in arr:
        items = [fmt(v) for v in row]
        for k in range(0, len(items), per_line):
            chunk = ', '.join(items[k:k + per_line])
            out.append(chunk + (',' if k + per_line < len(items) else ''))
    return out


def write_deck(m, path, flavour='abaqus', E=E_PLA, nu=NU_PLA, tests=None, field_output=True):
    """Writes path (.inp) and path[:-4] + '.json'. flavour: 'abaqus' or 'ccx'."""
    geom = BoxCell(*m.box)
    n = len(m.nodes)
    labels = np.arange(1, n + 1)
    rp = [n + 1, n + 2, n + 3]
    eq_text, n_eq = equations(m.nodes, geom, labels, rp)
    pin = int(labels[pin_node(m.nodes, geom)])
    name = m.info.get('pattern', 'cell')
    tests = lc.virtual_tests(name) if tests is None else tests
    abq = flavour == 'abaqus'

    L = []
    L += ['*Heading', f"** {name} rho={m.info.get('rho')} n_t={m.info.get('n_t')} {m.etype}, units mm N MPa",
          f'** cell {geom.Lx:.6f} x {geom.Ly:.6f} x {geom.Lz:.6f} (FE frame; prismatic cells: x = extrusion)']
    if abq:
        L += ['*Preprint, echo=NO, model=NO, history=NO, contact=NO']
    L += ['*Node']
    L += [f'{i}, {x:.10g}, {y:.10g}, {z:.10g}' for i, (x, y, z) in zip(labels, m.nodes)]
    rp_xyz = [(1.5 * geom.Lx, 0.5 * geom.Ly, 0.5 * geom.Lz), (0.5 * geom.Lx, 1.5 * geom.Ly, 0.5 * geom.Lz),
              (0.5 * geom.Lx, 0.5 * geom.Ly, 1.5 * geom.Lz)]
    L += [f'{r}, {x:.10g}, {y:.10g}, {z:.10g}' for r, (x, y, z) in zip(rp, rp_xyz)]
    L += [f'*Element, type={m.etype}, elset=SOLID']
    el = np.column_stack([np.arange(1, len(m.elements) + 1), m.elements + 1])
    L += _lines(el, str)
    L += ['*Nset, nset=RP_X', str(rp[0]), '*Nset, nset=RP_Y', str(rp[1]), '*Nset, nset=RP_Z', str(rp[2]),
          '*Nset, nset=NRP', f'{rp[0]}, {rp[1]}, {rp[2]}', '*Nset, nset=PIN', str(pin)]
    bnd = labels[geom.boundary_mask(m.nodes)]
    if not abq:                                   # CalculiX reports no reaction at the free RP nodes
        L += ['*Nset, nset=NBND'] + _lines(bnd.reshape(-1, 1)[:len(bnd) // 16 * 16].reshape(-1, 16), str) \
            + ([', '.join(map(str, bnd[len(bnd) // 16 * 16:]))] if len(bnd) % 16 else [])
    L += ['*Material, name=PLA', '*Elastic', f'{E:.6g}, {nu:.6g}',
          '*Solid Section, elset=SOLID, material=PLA', ',']
    L += [f'** Periodic boundary conditions: {n_eq} equations (algorithm of hd/fe/abaqus/periodic_equations.py)']
    L += [eq_text.rstrip('\n')]

    def outputs():
        o = []
        if abq:
            o += ['*Node Print, nset=NRP, totals=NO, summary=NO', 'U, RF']
            if field_output:
                o += ['*Output, field', '*Node Output', 'U, RF', '*Element Output, directions=YES', 'S']
        else:
            o += ['*Node Print, nset=NRP', 'U', '*Node Print, nset=NBND', 'RF']
            if field_output:
                o += ['*Node File', 'U', '*El File', 'S']
        return o

    steps = []
    for k, case in enumerate(lc.STRAIN_CASES):
        H = lc.strain_gradient(k)
        L += [f'*Step, name={case}, nlgeom=NO' if abq else '*Step', '*Static', '*Boundary, op=NEW', 'PIN, 1, 3, 0.']
        for kk in range(3):
            u = H[:, kk] * geom.L[kk]
            L += [f'{rp[kk]}, {i + 1}, {i + 1}, {u[i]:.10g}' for i in range(3)]
        L += outputs() + ['*End Step']
        steps.append(dict(name=case, kind='strain', voigt=k, H=H.tolist()))
    for tname, (desc, S) in tests.items():
        S = S * lc.STRESS
        L += [f'*Step, name={tname}, nlgeom=NO' if abq else '*Step', '*Static', '*Boundary, op=NEW', 'PIN, 1, 3, 0.',
              f'{rp[0]}, 2, 3, 0.', f'{rp[1]}, 3, 3, 0.', '*Cload, op=NEW']
        loads = [(rp[kk], i + 1, S[i, kk] * geom.volume / geom.L[kk]) for kk in range(3) for i in range(kk + 1)]
        loads = [l for l in loads if abs(l[2]) > 0]
        L += [f'{a}, {d}, {v:.10g}' for a, d, v in loads]
        L += outputs() + ['*End Step']
        steps.append(dict(name=tname, kind='stress', description=desc, sigma=S.tolist()))

    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    meta = dict(pattern=name, flavour=flavour, box=list(geom.L), volume=geom.volume, etype=m.etype,
                nodes=n, elements=len(m.elements), equations=n_eq, rp_labels=rp, pin=pin, E=E, nu=nu,
                strain=lc.STRAIN, stress=lc.STRESS, steps=steps,
                mesh={k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.info.items()},
                volume_fraction=float(m.volume_fraction))
    with open(path[:-4] + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    return meta
