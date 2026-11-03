"""Pictures of the CAD cells: out/figures/cells_overview.png (5 patterns x 7 rho, one unit cell each)
and out/figures/arrays.png (3x3x3-type arrays at rho = 0.1, 0.4, 0.7)."""
import os
import sys

import matplotlib
import numpy as np
import trimesh

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice.geometry import PATTERNS, RHOS, cad_box                 # noqa: E402
from lattice.render import render, tile                              # noqa: E402

CAD, FIG = 'out/cad', 'out/figures'
TMP = os.path.join(FIG, 'panels')
os.makedirs(TMP, exist_ok=True)
TILES = {'grid': (3, 3, 3), 'triangles': (3, 2, 3), 'honeycomb': (3, 2, 3), 'cubic': (3, 3, 3), 'gyroid': (3, 3, 3)}
NAMES = {'grid': 'Grid', 'triangles': 'Triangles', 'honeycomb': 'Honeycomb', 'cubic': 'Cubic', 'gyroid': 'Gyroid'}


def load(name, rho, faces):
    m = trimesh.load(f'{CAD}/{name}_rho{int(round(rho * 100)):02d}.stl')
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    return m


def panel(name, rho, array, px):
    png = f'{TMP}/{name}_{int(rho * 100):02d}_{"array" if array else "cell"}.png'
    if not os.path.exists(png):
        m = load(name, rho, 60000 if array else 150000)
        V, F = (tile(m.vertices, m.faces, cad_box(name), TILES[name]) if array else (m.vertices, m.faces))
        walled = name != 'gyroid'
        render(V, F, png, flat=walled, edges=walled, width=px, height=px)
    return plt.imread(png)


def sheet(rows, cols, get, row_lab, col_lab, out, px):
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(1.9 * len(cols), 1.95 * len(rows)), facecolor='white')
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axs[i, j]
            ax.imshow(get(r, c))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(col_lab(c), fontsize=11)
            if j == 0:
                ax.set_ylabel(row_lab(r), fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.95, bottom=0.005, wspace=0.02, hspace=0.04)
    fig.savefig(out, dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    sheet(PATTERNS, RHOS, lambda n, r: panel(n, r, False, 520), lambda n: NAMES[n],
          lambda r: rf'$\rho$ = {r:.1f}', f'{FIG}/cells_overview.png', 520)
    sheet(PATTERNS, (0.1, 0.4, 0.7), lambda n, r: panel(n, r, True, 900), lambda n: NAMES[n],
          lambda r: rf'$\rho$ = {r:.1f}', f'{FIG}/arrays.png', 900)
    print('done')
