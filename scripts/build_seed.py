"""Bundle post-processed results as an app dataset (lattice_app/data/<source>.json).
python scripts/build_seed.py <source> <post_dir> [<post_dir> ...]
post_dir: output folder of scripts/postprocess.py (results.json, convergence.json, stiffness.csv).
Duplicate (pattern, rho, n_t) entries are merged (the runs are identical decks)."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice.inp import E_PLA, NU_PLA            # noqa: E402


def main(source, dirs):
    recs, nodes = {}, {}
    for d in dirs:
        conv = os.path.join(d, 'convergence.json')
        if os.path.exists(conv):
            for c in json.load(open(conv)).values():
                for nt, n in zip(c['n_t'], c['nodes']):
                    nodes[(c['pattern'], c['rho'], float(nt))] = n
        tab = os.path.join(d, 'stiffness.csv')
        if os.path.exists(tab):
            for row in csv.DictReader(open(tab)):
                nodes[(row['pattern'], float(row['rho']), float(row['n_t']))] = int(row['nodes'])
        for r in json.load(open(os.path.join(d, 'results.json'))):
            k = (r['pattern'], float(r['rho']), float(r['n_t']))
            recs[k] = dict(pattern=k[0], rho=k[1], n_t=k[2], E=E_PLA, nu=NU_PLA, C=r['C'], nodes=nodes.get(k))
    for k, r in recs.items():
        r['nodes'] = r['nodes'] or nodes.get(k)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lattice_app', 'data',
                       source + '.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = sorted(recs.values(), key=lambda r: (r['pattern'], r['rho'], r['n_t']))
    with open(out, 'w') as f:
        json.dump(data, f, indent=0)
    print(f'{out}: {len(data)} records, {sum(r["nodes"] is None for r in data)} without node count')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
