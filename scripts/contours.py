"""Von Mises contour pictures of the virtual tests of one cell (tiled array, build direction up).

    python scripts/contours.py RESULT_DIR PATTERN RHO [--tile 2,3,2] [--out out/figures]

Reads <tag>_fields.npz (Abaqus, scripts/abaqus_extract.py) or <tag>.frd + <tag>_mesh.npz
(CalculiX, scripts/run_ccx.py). Values are von Mises stress divided by the applied stress
(1 MPa), i.e. the local stress concentration of the wall material.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice import fields                                          # noqa: E402
from lattice.geometry import PRISMATIC, fe_to_cad                   # noqa: E402
from lattice.render import jet                                       # noqa: E402

LABEL = {'UT': 'uniaxial tension', 'ET': 'equibiaxial tension', 'SS': 'simple shear'}


def load(d, pattern, rho):
    tags = sorted(glob.glob(os.path.join(d, f'{pattern}_rho{int(round(rho * 100)):02d}_nt*.json')))
    tags = [t[:-5] for t in tags if not t.endswith(('_rp.json', '_result.json'))]
    if not tags:
        raise SystemExit(f'no results for {pattern} rho {rho} in {d}')
    base = tags[-1]
    meta = json.load(open(base + '.json'))
    if os.path.exists(base + '_fields.npz'):
        z = np.load(base + '_fields.npz')
        nodes, elements, etype = z['nodes'], z['elements'], str(z['etype'])
        S = {str(n): s for n, s in zip(z['steps'], z['S'])}
        steps = [(st['name'], S[st['name'].upper()] if st['name'].upper() in S else S[st['name']]) for st in meta['steps']]
    else:
        z = np.load(base + '_mesh.npz')
        nodes, elements, etype = z['nodes'], z['elements'], str(z['etype'])
        fr = fields.parse_frd(base + '.frd', len(nodes))
        steps = [(st['name'], s) for st, (u, s) in zip(meta['steps'], fr)]
    kinds = {st['name']: st['kind'] for st in meta['steps']}
    return base, meta, nodes, elements, etype, [(n, s) for n, s in steps if kinds[n] == 'stress']


def main(d, pattern, rho, tile, out):
    base, meta, nodes, elements, etype, steps = load(d, pattern, rho)
    os.makedirs(os.path.join(out, 'panels'), exist_ok=True)
    tile = tile or ((2, 3, 2) if pattern in PRISMATIC else (2, 2, 2))
    ncol = min(len(steps), 4)
    nrow = int(np.ceil(len(steps) / ncol))
    fig = plt.figure(figsize=(3.3 * ncol, 3.9 * nrow + 0.5), facecolor='white')
    top_m = 0.5 / (3.9 * nrow + 0.5)                       # figure fraction kept for the title
    rh = (1 - top_m) / nrow
    bands = 12
    cmap = ListedColormap(jet((np.arange(bands) + 0.5) / bands))
    for k, (name, S) in enumerate(steps):
        vm = fields.mises(S) / meta['stress']
        top = float(np.percentile(vm, 99.5))
        png = os.path.join(out, 'panels', f'{os.path.basename(base)}_{name}_mises.png')
        fields.contour_png(nodes, elements, etype, vm, png, box=tuple(meta['box']), n=tile, vrange=(0.0, top),
                           to_cad=lambda X: fe_to_cad(X, pattern), width=820, height=820)
        y0 = 1 - top_m - (k // ncol + 1) * rh
        ax = fig.add_axes([(k % ncol) / ncol, y0 + 0.22 * rh, 1 / ncol, 0.70 * rh])
        ax.imshow(plt.imread(png))
        ax.axis('off')
        kind, plane = name.split('_')
        ax.set_title(f'{name}: {LABEL[kind]} ({plane})', fontsize=9.5, color='#0b0b0b')
        cax = fig.add_axes([(k % ncol) / ncol + 0.12 / ncol, y0 + 0.15 * rh, 0.76 / ncol, 0.03 * rh])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.linspace(0, top, bands + 1), bands), cmap=cmap),
                          cax=cax, orientation='horizontal')
        cb.set_ticks([0, top / 2, top])
        cb.ax.tick_params(labelsize=7, colors='#52514e')
        cb.set_label(f'von Mises / applied stress (max {vm.max():.1f})', fontsize=7.5, color='#52514e')
    fig.suptitle(f'{pattern.capitalize()}, rho = {rho:.1f}: von Mises stress in the virtual tests '
                 f'(build direction up)', fontsize=11, y=1 - 0.25 * top_m, va='center')
    p = os.path.join(out, f'mises_{pattern}_rho{int(round(rho * 100)):02d}.png')
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print(p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('pattern')
    ap.add_argument('rho', type=float)
    ap.add_argument('--tile', default=None)
    ap.add_argument('--out', default='out/figures')
    a = ap.parse_args()
    main(a.dir, a.pattern, a.rho, tuple(int(v) for v in a.tile.split(',')) if a.tile else None, a.out)
