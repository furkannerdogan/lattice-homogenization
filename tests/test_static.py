"""The read-only web version computes derived quantities in JavaScript (lattice_app/static/science.js).
Check them against lattice.post / lattice_app.analysis. Skipped without Node.js."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lattice import post
from lattice_app import analysis, store

NODE = shutil.which('node')
SCI = (Path(__file__).resolve().parent.parent / 'lattice_app' / 'static' / 'science.js').as_uri()


@pytest.mark.skipif(NODE is None, reason='node not installed')
def test_science_js_matches_python(tmp_path):
    ana = analysis.Analysis(store.Registry(tmp_path))
    cases = [(p, r) for p in ('grid', 'triangles', 'honeycomb', 'cubic', 'gyroid') for r in (0.15, 0.4, 0.85)]
    Cs = [ana.C_at(p, r, 'abaqus', 2100.0, 0.35) for p, r in cases]
    script = f"""
import * as S from '{SCI}';
const Cs = {json.dumps([C.tolist() for C in Cs])};
console.log(JSON.stringify(Cs.map((C) => ({{c: S.constants(C), au: S.anisotropyU(C), s: S.surface(C, 19, 37),
  p: S.polar(C, 'xz', 73), t: S.virtualTests(C), hs: S.hsUpper(0.3, 2100, 0.35)}}))));
"""
    f = tmp_path / 'check.mjs'
    f.write_text(script)
    js = json.loads(subprocess.run([NODE, str(f)], capture_output=True, text=True, check=True).stdout)
    for (p, r), C, o in zip(cases, Cs, js):
        ec = post.engineering_constants(C)
        for k, v in ec.items():
            assert np.isclose(o['c'][k], v, rtol=1e-9, atol=1e-12), (p, r, k)
        assert np.isclose(o['au'], post.universal_anisotropy(C)[0], rtol=1e-8, atol=1e-10)
        s = analysis.Analysis.surface(C, 19, 37)
        assert np.allclose(o['s']['E'], s['E'], rtol=1e-9)
        pl = analysis.Analysis.polar(C, 'xz', 73)
        assert np.allclose(o['p']['E'], pl['E'], rtol=1e-9) and np.allclose(o['p']['G'], pl['G'], rtol=1e-9)
        for a, b in zip(o['t'], analysis.Analysis.virtual_tests(C)):
            assert a['test'] == b['test']
            for k in b['values']:
                assert np.isclose(a['values'][k], b['values'][k], rtol=1e-9), (p, r, a['test'], k)
    hs = analysis.hs_upper(0.3, 2100.0, 0.35)
    assert all(np.isclose(js[0]['hs'][k], float(hs[k])) for k in ('K', 'G', 'E'))
