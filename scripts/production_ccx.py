"""Reference runs in CalculiX at the production mesh level (with virtual tests and field output).
python scripts/production_ccx.py grid:8,triangles:3,honeycomb:3,cubic:3 0.1,0.2,...  -> out/ccx_prod"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ccx import run
spec = [s.split(':') for s in sys.argv[1].split(',')]
rhos = [float(r) for r in sys.argv[2].split(',')]
out = sys.argv[3] if len(sys.argv) > 3 else 'out/ccx_prod'
for p, nt in spec:
    nt = float(nt) if p == 'gyroid' else int(nt)
    for r in rhos:
        try:
            res = run(p, r, nt, 4, out, field=True, quiet=True)
            print(p, r, nt, res['nodes'], f"{res['time_solve']:.0f}s", flush=True)
        except Exception as e:
            print(p, r, nt, 'FAILED', str(e)[-400:], flush=True)
