"""FastAPI server: JSON API under /api, the browser UI under /."""
import csv
import io
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lattice import materials, post, solver
from lattice.geometry import PATTERNS, RHOS, smallest_feature

from . import __version__, analysis, config, fieldview, meshview
from .jobs import Jobs
from .store import Registry

STATIC = Path(__file__).resolve().parent / 'static'
# gyroid: refinement is oscillatory and fine meshes are very large; results come from the bundled data only
NO_SOLVE = {'gyroid'}


class JobIn(BaseModel):
    pattern: str
    rho: float
    n_t: list[float]
    material: str = 'PLA'
    field: bool | None = None
    force: bool = False


class SettingsIn(BaseModel):
    ccx_path: str | None = None
    cgx_path: str | None = None
    threads: int | None = None
    field_output: bool | None = None
    source: str | None = None
    material: str | None = None


class MaterialIn(BaseModel):
    name: str
    E: float
    nu: float
    source: str = ''


def create_app():
    ws = config.workspace()
    reg = Registry(ws)
    ana = analysis.Analysis(reg)
    jobs = Jobs(ws, reg)
    app = FastAPI(title='Lattice homogenization', version=__version__)
    trend_cache = {}
    lock = threading.Lock()

    def mats():
        return materials.load(str(ws))

    def mat(name):
        m = mats().get(name)
        if not m:
            raise HTTPException(404, f'unknown material {name}')
        return m

    def check_pattern(p):
        if p not in PATTERNS:
            raise HTTPException(404, f'unknown pattern {p}')

    # ------------------------------------------------------------------ state, settings
    @app.get('/api/state')
    def state():
        s = config.load()
        return dict(version=__version__, workspace=str(ws), patterns=list(PATTERNS), no_solve=sorted(NO_SOLVE), rhos=list(RHOS),
                    production_nt=analysis.PRODUCTION_NT, rho_range=analysis.RHO_RANGE, materials=mats(),
                    sources=reg.sources(), settings=s, solver=solver.check(s.get('ccx_path') or None),
                    cgx=solver.find_cgx(s.get('cgx_path') or None))

    @app.post('/api/settings')
    def set_settings(body: SettingsIn):
        upd = {k: v for k, v in body.model_dump().items() if v is not None}
        if upd.get('ccx_path') and not os.path.isfile(upd['ccx_path']):
            raise HTTPException(400, f"file not found: {upd['ccx_path']}")
        s = config.save(upd)
        return dict(settings=s, solver=solver.check(s.get('ccx_path') or None),
                    cgx=solver.find_cgx(s.get('cgx_path') or None))

    @app.get('/api/settings/detect')
    def detect():
        return solver.all_candidates()

    @app.post('/api/materials')
    def add_material(body: MaterialIn):
        if not (0 < body.E and -1 < body.nu < 0.5):
            raise HTTPException(400, 'need E > 0 and -1 < nu < 0.5')
        m = mats()
        if body.name in materials.BUILTIN:
            raise HTTPException(400, 'built-in material; choose another name')
        m[body.name] = dict(name=body.name, E=body.E, nu=body.nu, source=body.source, user=True)
        materials.save_user(str(ws), m)
        return mats()

    @app.delete('/api/materials/{name}')
    def del_material(name: str):
        m = mats()
        if name in m and m[name].get('user'):
            del m[name]
            materials.save_user(str(ws), m)
        return mats()

    # ------------------------------------------------------------------ science
    @app.get('/api/explore')
    def explore(pattern: str, rho: float, source: str = 'abaqus', material: str = 'PLA'):
        check_pattern(pattern)
        if not 0.01 <= rho <= 0.99:
            raise HTTPException(400, 'rho must be in [0.01, 0.99]')
        return ana.explore(pattern, rho, source, mat(material))

    @app.get('/api/trends')
    def trends(source: str = 'abaqus', material: str = 'PLA'):
        m = mat(material)
        k = (source, m['E'], m['nu'], reg.version)
        with lock:
            if k not in trend_cache:
                trend_cache.clear()
                trend_cache[k] = ana.trends(source, m)
        return trend_cache[k]

    @app.get('/api/laws')
    def laws_(pattern: str, source: str = 'abaqus', material: str = 'PLA'):
        check_pattern(pattern)
        return ana.law_detail(pattern, source, mat(material))

    @app.get('/api/convergence/sets')
    def conv_sets(source: str = 'abaqus', material: str = 'PLA'):
        return ana.convergence_sets(source, mat(material)['nu'])

    @app.get('/api/convergence')
    def conv(pattern: str, rho: float, source: str = 'abaqus', material: str = 'PLA'):
        check_pattern(pattern)
        return ana.convergence(pattern, rho, source, mat(material)['nu'])

    @app.get('/api/mesh')
    def mesh_preview(pattern: str, rho: float, n_t: float):
        check_pattern(pattern)
        if n_t <= 0 or n_t > 12:
            raise HTTPException(400, 'n_t in (0, 12]')
        out = meshview.preview(pattern, rho, n_t)
        out['quality']['metric'] = 'radius ratio' if out['etype'] == 'C3D10' else 'scaled Jacobian'
        return out

    @app.get('/api/geometry')
    def geometry(pattern: str, rho: float):
        check_pattern(pattern)
        return dict(t_min=smallest_feature(pattern, rho))

    # ------------------------------------------------------------------ records
    @app.get('/api/records')
    def records(source: str | None = None):
        out = []
        for r in reg.all(source=source):
            out.append({k: r.get(k) for k in ('key', 'source', 'origin', 'pattern', 'rho', 'n_t', 'h', 'nodes', 'E',
                                              'nu', 'material', 'created', 'job_id', 'time_solve', 'etype')}
                       | dict(has_fields=bool(r.get('files', {}).get('frd')) and bool(r.get('job_dir'))))
        return sorted(out, key=lambda r: (r['source'], r['pattern'], r['rho'], r['n_t']))

    @app.get('/api/records/{key}')
    def record(key: str):
        r = reg.get(key)
        if not r:
            raise HTTPException(404, 'no such record')
        return r

    @app.delete('/api/records/{key}')
    def delete_record(key: str):
        if not reg.delete(key):
            raise HTTPException(400, 'bundled records cannot be deleted')
        return dict(ok=True)

    @app.get('/api/export/stiffness.csv')
    def export_csv(source: str = 'abaqus'):
        buf = io.StringIO()
        w = csv.writer(buf)
        head = ['source', 'pattern', 'rho', 'n_t', 'nodes', 'E_s', 'nu_s', 'frame'] + \
               [f'C{i + 1}{j + 1}' for i in range(6) for j in range(i, 6)]
        w.writerow(head)
        for r in sorted(reg.all(source=source), key=lambda r: (r['pattern'], r['rho'], r['n_t'])):
            C = post.to_print_frame(np.array(r['C']), r['pattern'])
            w.writerow([r['source'], r['pattern'], r['rho'], r['n_t'], r.get('nodes'), r['E'], r['nu'], 'print'] +
                       [f'{C[i, j]:.6g}' for i in range(6) for j in range(i, 6)])
        return Response(buf.getvalue(), media_type='text/csv',
                        headers={'Content-Disposition': f'attachment; filename=stiffness_{source}.csv'})

    # ------------------------------------------------------------------ jobs
    @app.get('/api/jobs')
    def list_jobs():
        return jobs.list()

    @app.get('/api/jobs/{jid}')
    def get_job(jid: str):
        j = jobs.get(jid)
        if not j:
            raise HTTPException(404, 'no such job')
        return j

    @app.post('/api/jobs')
    def submit(body: JobIn):
        check_pattern(body.pattern)
        if body.pattern in NO_SOLVE:
            raise HTTPException(400, f'{body.pattern}: CalculiX runs are switched off; the app shows the bundled results')
        m = mat(body.material)
        group = None if len(body.n_t) == 1 else f'refine-{body.pattern}-{body.rho}'
        return [jobs.submit(body.pattern, body.rho, nt, m, field=body.field, group=group, force=body.force)
                for nt in sorted(body.n_t)]

    @app.post('/api/jobs/{jid}/cancel')
    def cancel(jid: str):
        return dict(ok=jobs.cancel(jid))

    # ------------------------------------------------------------------ fields
    def field_rec(key):
        r = reg.get(key)
        if not r:
            raise HTTPException(404, 'no such record')
        return r

    @app.get('/api/fields/{key}')
    def field(key: str, step: str = 'E11'):
        r = field_rec(key)
        try:
            return fieldview.payload(r, step)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(404, str(e))

    @app.get('/api/fields/{key}/vtu')
    def field_vtu(key: str):
        r = field_rec(key)
        try:
            p = fieldview.export_vtu(r)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        return FileResponse(p, filename=os.path.basename(p), media_type='application/octet-stream')

    @app.post('/api/fields/{key}/open')
    def open_folder(key: str, tool: str = 'folder'):
        """Opens the run folder (Explorer) or the .frd in cgx on this computer."""
        r = field_rec(key)
        d = r.get('job_dir')
        if not d or not os.path.isdir(d):
            raise HTTPException(404, 'no run folder')
        if tool == 'cgx':
            cgx = solver.find_cgx(config.load().get('cgx_path') or None)
            if not cgx:
                raise HTTPException(400, 'cgx not found: set its path in Settings')
            subprocess.Popen([cgx, r['files']['frd'], r['files']['inp']], cwd=d)
        elif os.name == 'nt':
            os.startfile(d)                                           # noqa: S606
        else:
            subprocess.Popen(['xdg-open', d])
        return dict(ok=True, dir=d)

    def warm():                                                       # fits + trend curves for the defaults
        try:
            st = config.load()
            trends(st.get('source', 'abaqus'), st.get('material', 'PLA'))
        except Exception:                                              # noqa: BLE001
            pass
    threading.Thread(target=warm, daemon=True).start()

    # ------------------------------------------------------------------ UI
    app.mount('/static', StaticFiles(directory=STATIC), name='static')

    @app.get('/', response_class=HTMLResponse)
    def index():
        return (STATIC / 'index.html').read_text(encoding='utf-8')

    return app
