"""Read-only web version of the explorer (no server, no CalculiX), e.g. for GitHub Pages.

python scripts/build_static.py [--out site] [--no-mesh]

Writes <out>/index.html and <out>/static/ (the app's own front end, switched to static mode) and
<out>/data/: per source the print-frame C(rho) on the slider grid from the density laws, the FE points,
trend curves, laws and refinement studies; mesh previews for the FE densities and refinement levels.
The browser computes E(n), polar sections, constants and virtual tests from C (lattice_app/static/science.js,
checked against lattice.post in tests/test_static.py).
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lattice.geometry import PATTERNS, RHOS                    # noqa: E402
from lattice_app import analysis, meshview, store              # noqa: E402

RHO_GRID = np.round(np.arange(0.05, 0.95 + 1e-9, 0.005), 4)
LEVELS = {'gyroid': (1.5, 2, 2.5, 3)}
LEVELS_HEX = (2, 3, 4, 6)
MAX_MESH_NODES = 500_000                                       # skip bigger previews (file size)
SOURCES = ('abaqus', 'calculix-ref')
PLA = dict(name='PLA', E=2100.0, nu=0.35)


def upper(C):
    return [float('%.6g' % C[i, j]) for i in range(6) for j in range(i, 6)]


def rnd(x):
    """Round floats in nested data to 6 significant digits (file size)."""
    if isinstance(x, float):
        return float('%.6g' % x)
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    return x


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rnd(obj), separators=(',', ':')))


def build(out, mesh=True):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / 'lattice_app' / 'static', out / 'static',
                    ignore=shutil.ignore_patterns('__pycache__'))
    html = (ROOT / 'lattice_app' / 'static' / 'index.html').read_text(encoding='utf-8')
    (out / 'index.html').write_text(html.replace('<body>', '<body data-static="1">', 1), encoding='utf-8')
    (out / '.nojekyll').write_text('')

    reg = store.Registry(out / '_ws')
    ana = analysis.Analysis(reg)
    sources = [s for s in reg.sources() if s['id'] in SOURCES]
    dump(out / 'data' / 'state.json', dict(
        version=__import__('lattice_app').__version__, workspace='(read-only web version)', patterns=list(PATTERNS),
        rhos=list(RHOS), production_nt=analysis.PRODUCTION_NT, rho_range=analysis.RHO_RANGE,
        materials={'PLA': dict(PLA, source=__import__('lattice.materials', fromlist=['BUILTIN']).BUILTIN['PLA']['source'])},
        sources=sources, settings=dict(source='abaqus', material='PLA'), solver=dict(ok=False, message='read-only'),
        cgx=None, no_solve=list(PATTERNS), static=True, rho_grid=dict(start=0.05, step=0.005, n=len(RHO_GRID))))

    for src in SOURCES:
        d = out / 'data' / src
        grid = {}
        for p in PATTERNS:
            if ana.fitted(p, src, PLA['nu']) is None:
                continue
            Cs = []
            for r in RHO_GRID:
                try:
                    Cs.append(upper(ana.C_at(p, float(r), src, PLA['E'], PLA['nu'])))
                except ValueError:
                    Cs.append(None)
            fe = []
            for rec in ana.points(p, src, PLA['nu']):
                e = ana.explore(p, rec['rho'], src, PLA)
                fe.append(dict(rho=rec['rho'], n_t=rec['n_t'], nodes=rec.get('nodes'),
                               C_print=upper(np.array(e['fe']['C_print'])), law_vs_fe=e['fe']['law_vs_fe']))
            grid[p] = dict(C=Cs, fe=fe)
        dump(d / 'cgrid.json', dict(patterns=grid))
        dump(d / 'trends.json', ana.trends(src, PLA))
        for p in grid:
            dump(d / f'laws_{p}.json', ana.law_detail(p, src, PLA))
        sets = ana.convergence_sets(src, PLA['nu'])
        dump(d / 'conv_sets.json', sets)
        for s in sets:
            dump(d / f"conv_{s['pattern']}_{s['rho']}.json", ana.convergence(s['pattern'], s['rho'], src, PLA['nu']))
        print(src, 'done')

    index = {}
    if mesh:
        for p in PATTERNS:
            lv = LEVELS.get(p, LEVELS_HEX + ((8,) if p == 'grid' else ()))
            for r in RHOS:
                for nt in lv:
                    m = meshview.preview(p, r, nt)
                    meshview._cache.clear()
                    if m['nodes'] > MAX_MESH_NODES:
                        print(f'skip mesh {p} {r} {nt}: {m["nodes"]} nodes')
                        break
                    m['quality']['metric'] = 'radius ratio' if m['etype'] == 'C3D10' else 'scaled Jacobian'
                    dump(out / 'data' / 'mesh' / f'{p}_{r}_{nt:g}.json', m)
                    index.setdefault(p, {}).setdefault(str(r), []).append(nt)
                print('mesh', p, r, index.get(p, {}).get(str(r)))
    dump(out / 'data' / 'mesh_index.json', index)
    shutil.rmtree(out / '_ws', ignore_errors=True)
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    print(f'{out}: {size / 1e6:.1f} MB')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'site'))
    ap.add_argument('--no-mesh', action='store_true')
    a = ap.parse_args()
    build(a.out, mesh=not a.no_mesh)
