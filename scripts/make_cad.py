"""All CAD unit cells: 5 patterns x rho 0.1..0.7, L = 20 mm -> out/cad/*.stl, *.step, cad_summary.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice import cad                                    # noqa: E402
from lattice.geometry import PATTERNS, RHOS               # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else 'out/cad'
os.makedirs(OUT, exist_ok=True)
rows = []
for name in PATTERNS:
    for rho in RHOS:
        d = cad.gyroid_cell(rho, OUT) if name == 'gyroid' else cad.walled_cell(name, rho, OUT)
        d = {k: (float(v) if hasattr(v, 'dtype') else v) for k, v in d.items()}
        d['box_mm'] = [round(float(b), 4) for b in d['box_mm']]
        d.update(pattern=name, rho=rho)
        rows.append(d)
        print(f"{d['file']:16s} t = {d['t_mm']:.3f} mm  rho_stl = {d['rho_stl']:.4f}  watertight = {d['watertight']}",
              flush=True)
json.dump(rows, open(os.path.join(OUT, 'cad_summary.json'), 'w'), indent=1)
