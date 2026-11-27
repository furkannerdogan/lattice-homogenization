"""Solid materials (isotropic, linear elastic). Units: MPa.

Linear elasticity: the homogenized C of a cell scales with E_s at fixed nu_s, so a result computed for
one material serves every material with the same Poisson ratio (C_new = C * E_new / E_old). A different
nu_s needs new FE runs; the app shows such a material only where matching runs exist.

Only PLA is filled in. Add a material in the app (Settings > Materials) or in
<workspace>/materials.json with a source for every value.
"""
import json
import os

BUILTIN = {
    'PLA': dict(name='PLA', E=2100.0, nu=0.35, source='E: Pszczolkowski et al. (2022), static tensile test of FDM '
                                                       'PLA; nu: assumed (typical 0.33-0.36)'),
}


def load(workspace=None):
    mats = dict(BUILTIN)
    path = os.path.join(workspace, 'materials.json') if workspace else None
    if path and os.path.exists(path):
        for m in json.load(open(path)):
            mats[m['name']] = dict(name=m['name'], E=float(m['E']), nu=float(m['nu']), source=m.get('source', ''),
                                   user=True)
    return mats


def save_user(workspace, mats):
    user = [dict(name=m['name'], E=m['E'], nu=m['nu'], source=m.get('source', '')) for m in mats.values()
            if m.get('user')]
    with open(os.path.join(workspace, 'materials.json'), 'w') as f:
        json.dump(user, f, indent=1)


def same_nu(a, b, tol=1e-6):
    return abs(a - b) < tol
