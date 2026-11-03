"""Reading solver output and turning it into homogenized quantities.

Sources: CalculiX .dat (*NODE PRINT of U and RF at the reference points), Abaqus .dat (*NODE PRINT),
or the JSON written by scripts/abaqus_extract.py from an Abaqus ODB. All give, per step, the
displacements and reaction forces of RP_X, RP_Y, RP_Z.
"""
import json
import re

import numpy as np

from .loadcases import VOIGT_PAIRS
from .pbc import BoxCell, average_stress, gradient_from_rp

_NUM = r'[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?'


def parse_ccx_dat(path, meta, nodes):
    """[{U: (3,3), RF: (3,3)} per step], rows = RP_X, RP_Y, RP_Z.
    CalculiX gives no reaction force at the free reference nodes, so the average stress is taken from
    the forces at the boundary nodes, sigma_ij = sum_n f_i x_j / V, and converted to equivalent RP
    forces RF_k[i] = sigma_ik V / L_k (same convention as Abaqus). nodes: (n, 3) coordinates, label = index + 1."""
    rp = meta['rp_labels']
    txt = open(path).read()
    parts = re.split(r'\n\s*(displacements|forces) \(\w+,\w+,\w+\) for set (\w+) and time\s+' + _NUM, txt)
    U, T = [], []
    geom = BoxCell(*meta['box'])
    for kind, nset, body in zip(parts[1::3], parts[2::3], parts[3::3]):
        lab, vals = [], []
        for line in body.strip().splitlines():
            p = line.split()
            if len(p) == 4 and p[0].isdigit():
                lab.append(int(p[0])); vals.append([float(v) for v in p[1:]])
            elif p:
                break
        lab, vals = np.array(lab), np.array(vals)
        if kind == 'displacements' and nset == 'NRP':
            d = dict(zip(lab, vals))
            U.append(np.array([d[r] for r in rp]))
        elif kind == 'forces' and nset == 'NBND':
            T.append(np.einsum('ni,nj->ij', vals, nodes[lab - 1]) / geom.volume)
    if len(U) != len(meta['steps']) or len(T) != len(meta['steps']):
        raise RuntimeError(f'{path}: {len(U)} displacement and {len(T)} force blocks for {len(meta["steps"])} steps')
    RF = [np.stack([t[:, k] * geom.volume / geom.L[k] for k in range(3)], 0) for t in T]
    return [dict(U=u, RF=f) for u, f in zip(U, RF)]


def parse_abaqus_dat(path, meta):
    """Abaqus .dat with *Node Print, nset=NRP (U, RF): one table per step, in step order."""
    rp = meta['rp_labels']
    out = []
    for step in re.split(r'\n\s*S T E P\s+\d+', open(path).read())[1:]:
        rows = {}
        for m in re.finditer(r'^\s*(\d+)\s+(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM + r')\s+(' + _NUM +
                             r')\s+(' + _NUM + r')\s+(' + _NUM + r')\s*$', step, re.M):
            v = [float(x.replace('D', 'E')) for x in m.groups()[1:]]
            rows[int(m.group(1))] = v                     # last table of the step wins (end of step)
        if all(r in rows for r in rp):
            a = np.array([rows[r] for r in rp])
            out.append(dict(U=a[:, :3], RF=a[:, 3:]))
    if len(out) != len(meta['steps']):
        raise RuntimeError(f'{path}: found {len(out)} complete RP tables for {len(meta["steps"])} steps')
    return out


def load_odb_json(path, meta):
    d = json.load(open(path))
    by = {s['name'].upper(): s for s in d['steps']}
    out = []
    for s in meta['steps']:
        r = by[s['name'].upper()]
        out.append(dict(U=np.array(r['U']), RF=np.array(r['RF'])))
    return out


def stiffness(meta, steps):
    """C (6x6, Voigt 11 22 33 23 13 12, engineering shear) from the six strain steps, plus checks."""
    geom = BoxCell(*meta['box'])
    C = np.zeros((6, 6))
    asym = []
    for s, r in zip(meta['steps'], steps):
        if s['kind'] != 'strain':
            continue
        T = average_stress(r['RF'], geom)
        asym.append(np.abs(T - T.T).max() / max(np.abs(T).max(), 1e-30))
        sig = np.array([T[min(i, j), max(i, j)] for i, j in VOIGT_PAIRS])   # upper triangle (repo convention)
        C[:, s['voigt']] = sig / meta['strain']
    sym = np.abs(C - C.T).max() / np.abs(C).max()
    return 0.5 * (C + C.T), dict(major_symmetry_error=float(sym), stress_symmetry_error=float(max(asym)))


def virtual_test_results(meta, steps):
    """Per stress-controlled step: average stress and strain (Voigt, engineering shear)."""
    geom = BoxCell(*meta['box'])
    out = {}
    for s, r in zip(meta['steps'], steps):
        if s['kind'] != 'stress':
            continue
        T = average_stress(r['RF'], geom)
        H = gradient_from_rp(r['U'], geom)
        eps = 0.5 * (H + H.T)
        out[s['name']] = dict(
            description=s.get('description', ''),
            sigma_applied=np.array(s['sigma']),
            sigma=0.5 * (T + T.T), sigma_asym=float(np.abs(T - T.T).max()),
            eps=eps, H=H,
            eps_voigt=np.array([eps[i, j] * (1 if i == j else 2) for i, j in VOIGT_PAIRS]),
            sigma_voigt=np.array([0.5 * (T[i, j] + T[j, i]) for i, j in VOIGT_PAIRS]))
    return out
