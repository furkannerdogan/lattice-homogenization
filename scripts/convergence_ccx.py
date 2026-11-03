"""Mesh-convergence runs in CalculiX: patterns x rho x n_t (no field output). Appends to out/ccx_conv/summary.jsonl"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ccx import run
pats = sys.argv[1].split(',')
rhos = [float(r) for r in sys.argv[2].split(',')]
nts = [float(n) for n in sys.argv[3].split(',')]
next_ = int(sys.argv[4]) if len(sys.argv) > 4 else 4
out = 'out/ccx_conv'
os.makedirs(out, exist_ok=True)
for p in pats:
    for r in rhos:
        for nt in nts:
            try:
                res = run(p, r, nt if p == 'gyroid' else int(nt), next_, out, field=False, quiet=True, tests=False)
                res.pop('tests', None)
                with open(f'{out}/summary.jsonl', 'a') as f:
                    f.write(json.dumps(res) + '\n')
                print(p, r, nt, res['nodes'], f"{res['time_solve']:.0f}s", 'C11', round(res['C'][0][0], 3), 'C44', round(res['C'][3][3], 4), flush=True)
            except Exception as e:
                print(p, r, nt, 'FAILED', str(e)[-500:], flush=True)
