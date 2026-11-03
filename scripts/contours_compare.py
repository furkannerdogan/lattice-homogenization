"""One figure comparing the five cells: rows UT / ET / SS, columns patterns, at one density.

    python scripts/contours_compare.py RESULT_DIR RHO [--out out/figures]

Prismatic cells (x = extrusion): UT_y, ET_yz, SS_yz (loads in the print plane, the weak plane);
cubic and gyroid: UT_x, ET_xy, SS_xy. Colour scale per panel: von Mises / applied stress, capped at
the 99.5th percentile (sharp-corner singularities).
"""
import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contours import load                                            # noqa: E402
from lattice import fields                                          # noqa: E402
from lattice.geometry import PATTERNS, PRISMATIC, fe_to_cad          # noqa: E402
from lattice.render import jet                                       # noqa: E402

ROWS = [('UT', {'p': 'UT_y', 'c': 'UT_x'}, 'Uniaxial tension'),
        ('ET', {'p': 'ET_yz', 'c': 'ET_xy'}, 'Equibiaxial tension'),
        ('SS', {'p': 'SS_yz', 'c': 'SS_xy'}, 'Simple shear')]
NAMES = {'grid': 'Grid', 'triangles': 'Triangles', 'honeycomb': 'Honeycomb', 'cubic': 'Cubic', 'gyroid': 'Gyroid'}


def main(d, rho, out):
    pats = []
    for p in PATTERNS:
        try:
            pats.append((p, load(d, p, rho)))
        except SystemExit:
            print(f'no {p} at rho {rho}')
    os.makedirs(os.path.join(out, 'panels'), exist_ok=True)
    bands = 12
    cmap = ListedColormap(jet((np.arange(bands) + 0.5) / bands))
    nc, nr = len(pats), len(ROWS)
    fig = plt.figure(figsize=(3.0 * nc, 3.45 * nr + 0.5), facecolor='white')
    top = 0.5 / (3.45 * nr + 0.5)
    rh = (1 - top) / nr
    for j, (p, (base, meta, nodes, elements, etype, steps)) in enumerate(pats):
        S = dict(steps)
        tile = (2, 3, 2) if p in PRISMATIC else (2, 2, 2)
        for i, (kind, names, label) in enumerate(ROWS):
            name = names['p' if p in PRISMATIC else 'c']
            vm = fields.mises(S[name]) / meta['stress']
            hi = float(np.percentile(vm, 99.5))
            png = os.path.join(out, 'panels', f'{os.path.basename(base)}_{name}_mises.png')
            if not os.path.exists(png):
                fields.contour_png(nodes, elements, etype, vm, png, box=tuple(meta['box']), n=tile, vrange=(0, hi),
                                   to_cad=lambda X, p=p: fe_to_cad(X, p), width=820, height=820)
            y0 = 1 - top - (i + 1) * rh
            ax = fig.add_axes([j / nc, y0 + 0.2 * rh, 1 / nc, 0.74 * rh])
            ax.imshow(plt.imread(png))
            ax.axis('off')
            ax.set_title(f'{NAMES[p]}: {name}' if i == 0 else name, fontsize=9.5)
            if j == 0:
                fig.text(0.004, y0 + 0.57 * rh, label, rotation=90, va='center', fontsize=10.5)
            cax = fig.add_axes([j / nc + 0.14 / nc, y0 + 0.13 * rh, 0.72 / nc, 0.03 * rh])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.linspace(0, hi, bands + 1), bands), cmap=cmap),
                              cax=cax, orientation='horizontal')
            cb.set_ticks([0, hi / 2, hi])
            cb.ax.tick_params(labelsize=7, colors='#52514e')
            cb.set_label(f'$\\sigma_{{vM}}/\\bar\\sigma$ (max {vm.max():.0f})', fontsize=7.5, color='#52514e')
    fig.suptitle(f'Von Mises stress in the virtual tests, rho = {rho:.1f} (applied average stress 1 MPa; '
                 f'loads in the print plane; build direction up)', fontsize=11, y=1 - 0.3 * top, va='center')
    path = os.path.join(out, f'mises_compare_rho{int(round(rho * 100)):02d}.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('rho', type=float)
    ap.add_argument('--out', default='out/figures')
    a = ap.parse_args()
    main(a.dir, a.rho, a.out)
