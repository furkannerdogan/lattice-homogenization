"""Post-processing of finished runs.

    python scripts/postprocess.py RESULT_DIR [RESULT_DIR ...] [--out out/post]

A result is recognised by its files next to the deck's <tag>.json:
    <tag>_rp.json      Abaqus (scripts/abaqus_extract.py)
    <tag>.dat          Abaqus .dat (fallback; *Node Print of the RPs) or CalculiX .dat
    <tag>_result.json  CalculiX run by scripts/run_ccx.py
Outputs (out/post): stiffness.csv, results.json, convergence.json + convergence_*.png,
laws.json + laws_*.png, constants_vs_rho.png, polar_E.png, virtual_tests.csv.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice import laws, post, results                                 # noqa: E402
from lattice.geometry import PATTERNS                                    # noqa: E402

COLORS = {'grid': '#2a78d6', 'triangles': '#eb6834', 'honeycomb': '#1baf7a', 'cubic': '#eda100', 'gyroid': '#e87ba4'}
MARKERS = {'grid': 's', 'triangles': '^', 'honeycomb': 'h', 'cubic': 'D', 'gyroid': 'o'}
INK, INK2, MUTED, GRIDC = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
RAMP = ['#86b6ef', '#3987e5', '#184f95']
NAMES = {'grid': 'Grid', 'triangles': 'Triangles', 'honeycomb': 'Honeycomb', 'cubic': 'Cubic', 'gyroid': 'Gyroid'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRIDC, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                     'figure.facecolor': 'white', 'axes.facecolor': '#fcfcfb'})
TAG = re.compile(r'(?P<p>[a-z]+)_rho(?P<r>\d\d)_nt(?P<n>[\dp]+)$')


def load(dirs):
    recs = []
    for d in dirs:
        for meta_path in sorted(glob.glob(os.path.join(d, '*.json'))):
            base = meta_path[:-5]
            m = TAG.search(os.path.basename(base))
            if not m:
                continue
            meta = json.load(open(meta_path))
            rec = dict(pattern=m['p'], rho=int(m['r']) / 100, n_t=float(m['n'].replace('p', '.')), meta=meta,
                       source=None)
            try:
                if os.path.exists(base + '_rp.json'):
                    steps, rec['source'] = results.load_odb_json(base + '_rp.json', meta), 'abaqus-odb'
                elif os.path.exists(base + '_result.json'):
                    r = json.load(open(base + '_result.json'))
                    rec.update(C=np.array(r['C']), source='calculix', nodes=r['nodes'], etype=r['etype'],
                               tests_raw=r.get('tests', {}))
                    recs.append(rec)
                    continue
                elif os.path.exists(base + '.dat') and meta['flavour'] == 'abaqus':
                    steps, rec['source'] = results.parse_abaqus_dat(base + '.dat', meta), 'abaqus-dat'
                else:
                    continue
            except Exception as e:                                       # noqa: BLE001
                print(f'skip {base}: {e}')
                continue
            C, chk = results.stiffness(meta, steps)
            vt = results.virtual_test_results(meta, steps)
            rec.update(C=C, checks=chk, nodes=meta['nodes'], etype=meta['etype'],
                       tests_raw={k: dict(sigma=v['sigma_voigt'].tolist(), eps=v['eps_voigt'].tolist())
                                  for k, v in vt.items()})
            recs.append(rec)
    return recs


def solid_C(meta):
    return post.iso_C(meta['E'], meta['nu'])


def finest(recs):
    """One record per (pattern, rho): the finest mesh."""
    best = {}
    for r in recs:
        k = (r['pattern'], r['rho'])
        if k not in best or r['n_t'] > best[k]['n_t']:
            best[k] = r
    return best


def write_tables(best, out):
    rows, tests = [], []
    for (p, rho), r in sorted(best.items(), key=lambda kv: (PATTERNS.index(kv[0][0]), kv[0][1])):
        C = r['C']
        ec = post.engineering_constants(C)
        Es = r['meta']['E']
        au, _ = post.universal_anisotropy(C)
        row = dict(pattern=p, rho=rho, n_t=r['n_t'], nodes=r['nodes'], source=r['source'])
        row.update({f'C{i + 1}{j + 1}': round(C[i, j], 4) for i in range(6) for j in range(i, 6)
                    if (i < 3 and j < 3) or i == j})
        row.update({k: round(v / Es, 6) if k[0] in 'EG' else round(v, 5) for k, v in ec.items()})
        row.update(A_U=round(au, 5), zener=round(post.zener(C), 4) if p in ('cubic', 'gyroid') else '')
        row.update({f'chk_{k}': round(v, 5) for k, v in post.symmetry_checks(C, p).items()})
        rows.append(row)
        for name, t in r.get('tests_raw', {}).items():
            sig, eps = np.array(t['sigma']), np.array(t['eps'])
            fe = post.test_moduli(sig, eps, name)
            pred = post.test_moduli(sig, post.predicted_test(C, sig), name)
            for k in fe:
                tests.append(dict(pattern=p, rho=rho, test=name, quantity=k, fe=round(fe[k], 6),
                                  from_C=round(pred[k], 6), rel_diff=f'{abs(fe[k] - pred[k]) / abs(pred[k]):.2e}'))
    for name, data in (('stiffness.csv', rows), ('virtual_tests.csv', tests)):
        if data:
            keys = list(dict.fromkeys(k for d in data for k in d))
            with open(os.path.join(out, name), 'w', newline='') as f:
                w = csv.DictWriter(f, keys)
                w.writeheader()
                w.writerows(data)
    return rows, tests


def convergence(recs, out):
    groups = {}
    for r in recs:
        groups.setdefault((r['pattern'], r['rho']), []).append(r)
    res = {}
    for (p, rho), g in sorted(groups.items()):
        if len(g) < 2:
            continue
        g = sorted(g, key=lambda r: r['n_t'])
        h = np.array([1.0 / r['n_t'] for r in g])
        ent = post.independent_entries(p)
        Cf = post.symmetrize(g[-1]['C'], p)
        d = {}
        for name, (i, j) in ent.items():
            f = np.array([post.symmetrize(r['C'], p)[i - 1, j - 1] for r in g])
            e = laws.extrapolate(h, f)
            if e['p'] is not None and not (0.6 < e['p'] < 6):      # erratic, not asymptotic: use the finest mesh
                e.update(f0=float(f[-1]), unreliable_p=True, rel_err=((f - f[-1]) / f[-1]).tolist())
            # error on the scale of the matrix: |f - f0| / sqrt(C_ii C_jj) (near-zero couplings do not dominate)
            scale = np.sqrt(abs(Cf[i - 1, i - 1] * Cf[j - 1, j - 1]))
            e['err_scaled'] = (np.abs(f - e['f0']) / scale).tolist()
            e['values'] = f.tolist()
            d[name] = e
        worst = np.max([d[k]['err_scaled'] for k in d], axis=0)
        ok = [g[k]['n_t'] for k in range(len(g)) if worst[k] <= 0.01]
        res[f'{p}|{rho}'] = dict(pattern=p, rho=rho, n_t=[r['n_t'] for r in g], nodes=[r['nodes'] for r in g],
                                 entries=d, worst_rel_err=worst.tolist(), coarsest_within_1pct=min(ok) if ok else None)
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for k, (name, e) in enumerate(d.items()):
            ax.plot([r['n_t'] for r in g], 100 * np.maximum(e['err_scaled'], 1e-6), marker='o', ms=4, lw=1.6,
                    color=list(COLORS.values())[k % 5] if k < 5 else '#6250d6', label=name)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls='--')
        ax.set_yscale('log')
        ax.set_xlabel('elements across the thinnest wall, $n_t$')
        ax.set_ylabel(r'$|C_{ij}-C_{ij}^{0}|\,/\sqrt{C_{ii}C_{jj}}$ (%)')
        ax.set_title(f'{NAMES[p]}, $\\rho$ = {rho:.1f}', color=INK, fontsize=10, loc='left')
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'convergence_{p}_rho{int(rho * 100):02d}.png'), dpi=200)
        plt.close(fig)
    json.dump(res, open(os.path.join(out, 'convergence.json'), 'w'), indent=1)
    return res


def fit_laws(best, out):
    res = {}
    for p in PATTERNS:
        rs = sorted([r for (q, _), r in best.items() if q == p], key=lambda r: r['rho'])
        if len(rs) < 4:
            continue
        rho = np.array([r['rho'] for r in rs])
        Cs = solid_C(rs[0]['meta'])
        ent = post.independent_entries(p)
        res[p] = {}
        fig, ax = plt.subplots(figsize=(4.4, 3.3))
        rr = np.linspace(0.0, 1.0, 201)
        for k, (name, (i, j)) in enumerate(ent.items()):
            y = np.array([post.symmetrize(r['C'], p)[i - 1, j - 1] for r in rs]) / Cs[i - 1, j - 1]
            law = laws.select_law(rho, y)
            pw = laws.power_law(rho, y)
            res[p][name] = dict(rho=rho.tolist(), normalized=y.tolist(), solid_value=Cs[i - 1, j - 1], law=law,
                                power=pw, g_equals_1_minus_h=True)
            col = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'][k]
            ax.plot(rr[1:], laws.evaluate(law, rr[1:]), color=col, lw=1.6)
            ax.plot(rho, y, ls='none', marker='o', ms=4.5, color=col, mec='white', mew=0.8, label=name)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=':')
        ax.set_xlim(0.05, 1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(max(1e-4, 0.5 * min(min(v['normalized']) for v in res[p].values())), 1.2)
        ax.set_xticks([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
        ax.set_xticklabels(['0.1', '0.2', '0.3', '0.5', '0.7', '1'])
        ax.set_xlabel(r'relative density $\rho$')
        ax.set_ylabel(r'$C_{ij}\,/\,C^{s}_{ij}$ = h($\rho$)')
        ax.set_title(f'{NAMES[p]}: FE (dots) and fitted law (lines)', color=INK, fontsize=10, loc='left')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'laws_{p}.png'), dpi=200)
        plt.close(fig)
    json.dump(res, open(os.path.join(out, 'laws.json'), 'w'), indent=1)
    return res


def constants_plot(best, out):
    qs = [('E1', 'E$_x$ / E$_s$'), ('E2', 'E$_y$ / E$_s$'), ('G12', 'G$_{xy}$ / E$_s$'), ('G23', 'G$_{yz}$ / E$_s$')]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.0))
    for ax, (q, lab) in zip(axs, qs):
        for p in PATTERNS:
            rs = sorted([r for (pp, _), r in best.items() if pp == p], key=lambda r: r['rho'])
            if not rs:
                continue
            v = [post.engineering_constants(r['C'])[q] / r['meta']['E'] for r in rs]
            ax.plot([r['rho'] for r in rs], v, marker=MARKERS[p], ms=4.5, lw=1.6, color=COLORS[p], mec='white',
                    mew=0.6, label=NAMES[p])
        ax.set_yscale('log')
        ax.set_xlabel(r'$\rho$')
        ax.set_title(lab, color=INK, fontsize=10, loc='left')
    axs[0].legend(fontsize=8)
    axs[0].text(0.03, 0.97, 'grid, triangles, honeycomb coincide:\nE$_x$ = $\\rho$ E$_s$ (walls along x)', transform=axs[0].transAxes,
                va='top', fontsize=7.5, color=INK2)
    fig.text(0.5, -0.02, 'FE frame: x = extrusion axis of grid, triangles, honeycomb; cube axes for cubic and gyroid',
             ha='center', color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'constants_vs_rho.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def anisotropy_plot(best, out):
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for p in PATTERNS:
        rs = sorted([r for (pp, _), r in best.items() if pp == p], key=lambda r: r['rho'])
        if not rs:
            continue
        ax.plot([r['rho'] for r in rs], [post.universal_anisotropy(r['C'])[0] for r in rs], marker=MARKERS[p], ms=4.5,
                lw=1.6, color=COLORS[p], mec='white', mew=0.6, label=NAMES[p])
    ax.set_yscale('log')
    ax.set_xlabel(r'relative density $\rho$')
    ax.set_title('Universal anisotropy index $A^U$ (0 = isotropic)', color=INK, fontsize=10, loc='left')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'anisotropy_vs_rho.png'), dpi=200)
    plt.close(fig)


def polar_plot(best, out, rhos=(0.1, 0.4, 0.7)):
    """E(n)/max in the three coordinate planes, per pattern, for three densities."""
    pats = [p for p in PATTERNS if any(q == p for q, _ in best)]
    planes = [('x-y', lambda a: np.stack([np.cos(a), np.sin(a), 0 * a], -1)),
              ('y-z', lambda a: np.stack([0 * a, np.cos(a), np.sin(a)], -1)),
              ('x-z', lambda a: np.stack([np.cos(a), 0 * a, np.sin(a)], -1))]
    a = np.linspace(0, 2 * np.pi, 721)
    fig, axs = plt.subplots(len(pats), 3, figsize=(7.2, 2.35 * len(pats)), subplot_kw=dict(projection='polar'))
    axs = np.atleast_2d(axs)
    for i, p in enumerate(pats):
        for j, (pl, f) in enumerate(planes):
            ax = axs[i, j]
            for k, rho in enumerate(rhos):
                r = best.get((p, rho))
                if r is None:
                    continue
                E = post.directional_E(r['C'], f(a))
                ax.plot(a, E / E.max(), color=RAMP[k], lw=1.6, label=rf'$\rho$={rho:.1f}')
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0.5, 1.0])
            ax.set_yticklabels([])
            ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
            ax.set_xticklabels([pl[0], pl[2], '', ''], color=MUTED)
            if j == 0:
                ax.set_ylabel(NAMES[p], labelpad=22, color=INK, fontsize=10)
            if i == 0:
                ax.set_title(f'{pl} plane', color=INK, fontsize=10, pad=12)
    axs[0, 2].legend(loc='upper left', bbox_to_anchor=(1.05, 1.1), fontsize=8)
    fig.text(0.5, 0.0, 'Directional Young modulus E(n), each curve divided by its own maximum (shape only)',
             ha='center', color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'polar_E.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--out', default='out/post')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = load(a.dirs)
    print(f'{len(recs)} results')
    best = finest(recs)
    write_tables(best, a.out)
    conv = convergence(recs, a.out)
    for k, v in conv.items():
        print(f"convergence {k}: worst |err| per level {np.round(np.array(v['worst_rel_err']) * 100, 2)} % "
              f"at n_t {v['n_t']}; coarsest within 1 %: {v['coarsest_within_1pct']}")
    lw = fit_laws(best, a.out)
    for p, d in lw.items():
        for name, e in d.items():
            m = e['law'][e['law']['chosen']]
            print(f"law {p} {name}: {e['law']['chosen']} d = {m['d']}, LOO rms {100 * m['loo_rms']:.2f} % "
                  f"(mixture {100 * e['law']['mixture']['loo_rms']:.2f} %), power n = {e['power']['n']:.2f}")
    constants_plot(best, a.out)
    anisotropy_plot(best, a.out)
    polar_plot(best, a.out)
    json.dump([dict(pattern=r['pattern'], rho=r['rho'], n_t=r['n_t'], source=r['source'], C=np.asarray(r['C']).tolist())
               for r in recs], open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
