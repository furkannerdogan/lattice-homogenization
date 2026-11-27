"""Stress fields of an app CalculiX run, prepared for the browser (three.js) and for ParaView.

The .frd is parsed once into fields.npz (float32 U and S per step). The surface is triangulated with
midside nodes (fields.surface) and sent in the print frame (z = build direction) together with the
macroscopic displacement gradient H of each step, so the browser can tile periodic copies:
x -> x + t, u -> u + H t.
"""
import base64
import json
import os

import numpy as np

from lattice import fields, results, vtu
from lattice.geometry import PRISMATIC, fe_to_cad
from lattice.pbc import BoxCell, gradient_from_rp

_P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)    # x_print = P x_fe for the prismatic cells


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode('ascii')


def _paths(rec):
    d = rec.get('job_dir')
    if not d or not os.path.isdir(d):
        raise FileNotFoundError('field data only exist for runs made in this app (with field output on)')
    f = rec.get('files', {})
    if not f.get('frd'):
        raise FileNotFoundError('this run was made without field output')
    return d, os.path.join(d, f['mesh']), os.path.join(d, f['frd']), os.path.join(d, f['inp'][:-4])


def load(rec):
    d, mesh_p, frd_p, base = _paths(rec)
    z = np.load(mesh_p)
    nodes, el, etype, box = z['nodes'], z['elements'], str(z['etype']), z['box']
    cache = os.path.join(d, 'fields.npz')
    if os.path.exists(cache):
        c = np.load(cache)
        U, S, H = c['U'], c['S'], c['H']
    else:
        steps = fields.parse_frd(frd_p, len(nodes))
        U = np.stack([s[0] for s in steps]).astype(np.float32)
        S = np.stack([s[1] for s in steps]).astype(np.float32)
        meta = json.load(open(base + '.json'))
        st = results.parse_ccx_dat(base + '.dat', meta, nodes)
        geom = BoxCell(*meta['box'])
        H = np.stack([gradient_from_rp(s['U'], geom) for s in st])
        np.savez_compressed(cache, U=U, S=S, H=H)
    return dict(nodes=nodes, elements=el, etype=etype, box=box, U=U, S=S, H=H, steps=rec['steps'])


def payload(rec, step):
    f = load(rec)
    k = f['steps'].index(step)
    tris = fields.surface(f['elements'], f['etype'], f['nodes'])
    used, inv = np.unique(tris, return_inverse=True)
    tris = inv.reshape(tris.shape)
    X, U, H, box = f['nodes'][used], f['U'][k][used], f['H'][k], f['box']
    vm = fields.mises(f['S'][k][used])
    vm_all = fields.mises(f['S'][k])
    if rec['pattern'] in PRISMATIC:
        X, U = fe_to_cad(X, rec['pattern']), fe_to_cad(U, rec['pattern'])
        H = _P @ H @ _P.T
        box = fe_to_cad(box, rec['pattern'])
    return dict(step=step, steps=f['steps'], n_vertices=len(X), n_triangles=len(tris), box=list(map(float, box)),
                H=H.tolist(), X=_b64(X, np.float32), U=_b64(U, np.float32), T=_b64(tris, np.uint32),
                mises=_b64(vm, np.float32), p995=float(np.percentile(vm_all, 99.5)), vmax=float(vm_all.max()),
                vmean=float(vm_all.mean()), umax=float(np.abs(U).max()))


def export_vtu(rec):
    f = load(rec)
    path = os.path.join(rec['job_dir'], os.path.basename(rec['files']['inp'])[:-4] + '.vtu')
    if not os.path.exists(path):
        vtu.write_run(path, f['nodes'], f['elements'], f['etype'], list(zip(f['U'], f['S'])), f['steps'])
    return path
