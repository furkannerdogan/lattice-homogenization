"""CalculiX run -> ParaView .vtu (quadratic elements; U, S and von Mises per step).
python scripts/to_paraview.py <run_dir>/<tag>.inp   (needs <tag>_mesh.npz, <tag>.frd, <tag>.json next to it)
Open the .vtu in ParaView, colour by Mises_<step>, Warp By Vector with U_<step>."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice import fields, vtu            # noqa: E402


def convert(inp):
    base = inp[:-4] if inp.endswith('.inp') else inp
    z = np.load(base + '_mesh.npz')
    meta = json.load(open(base + '.json'))
    steps = fields.parse_frd(base + '.frd', len(z['nodes']))
    out = base + '.vtu'
    vtu.write_run(out, z['nodes'], z['elements'], str(z['etype']), steps, [s['name'] for s in meta['steps']])
    return out


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(convert(p))
