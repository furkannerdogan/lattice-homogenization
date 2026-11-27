"""Job queue: CalculiX runs in a background thread, one at a time (ccx itself uses several threads).

A job = one cell (pattern, rho, n_t) for one material. A refinement study is submitted as several jobs
with a common group id. Before a job runs, its registry key is checked: an existing CalculiX record
with the same key is reused (status 'cached'). Job folders: <workspace>/jobs/<id>/.
"""
import json
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path

from lattice import pipeline, solver

from . import config
from .store import make_key


class Jobs:
    def __init__(self, workspace, registry, on_done=None):
        self.dir = Path(workspace) / 'jobs'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.reg = registry
        self.on_done = on_done
        self.q = queue.Queue()
        self.jobs = {}
        self.lock = threading.Lock()
        for f in sorted(self.dir.glob('*/job.json')):
            try:
                j = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            if j['status'] in ('queued', 'running'):
                j['status'], j['error'] = 'failed', 'interrupted (server stopped)'
                f.write_text(json.dumps(j, indent=1))
            self.jobs[j['id']] = j
        threading.Thread(target=self._worker, daemon=True).start()

    def _save(self, j):
        d = self.dir / j['id']
        d.mkdir(exist_ok=True)
        (d / 'job.json').write_text(json.dumps(j, indent=1, default=float))

    def submit(self, pattern, rho, n_t, material, n_ext=4, field=None, group=None, force=False):
        s = config.load()
        field = s['field_output'] if field is None else field
        key = make_key('calculix', pattern, rho, n_t, material['E'], material['nu'], n_ext,
                       mesh_version=pipeline.MESH_VERSION)
        j = dict(id=time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6], pattern=pattern, rho=float(rho),
                 n_t=float(n_t), n_ext=n_ext, material=material, field=bool(field), group=group, key=key,
                 status='queued', created=time.strftime('%Y-%m-%d %H:%M:%S'), log=[], error=None)
        if not force and self.reg.get(key):
            j['status'], j['log'] = 'cached', ['identical run already in the registry; not run again']
        with self.lock:
            self.jobs[j['id']] = j
        self._save(j)
        if j['status'] == 'queued':
            self.q.put(j['id'])
        return j

    def cancel(self, jid):
        j = self.jobs.get(jid)
        if not j:
            return False
        if j['status'] == 'queued':
            j['status'] = 'cancelled'
            self._save(j)
        elif j['status'] == 'running':
            j['cancel'] = True
        return True

    def list(self):
        return sorted(({k: v for k, v in j.items() if k != 'log'} | dict(last=(j['log'][-1] if j['log'] else ''))
                       for j in self.jobs.values()), key=lambda j: j['created'], reverse=True)

    def get(self, jid):
        return self.jobs.get(jid)

    def _worker(self):
        while True:
            jid = self.q.get()
            j = self.jobs.get(jid)
            if not j or j['status'] != 'queued':
                continue
            s = config.load()
            ccx = solver.find_ccx(s.get('ccx_path') or None)
            j['status'], j['started'] = 'running', time.strftime('%Y-%m-%d %H:%M:%S')
            self._save(j)

            def log(line, j=j):
                j['log'].append(line)
                if len(j['log']) > 400:
                    j['log'] = j['log'][:20] + ['...'] + j['log'][-300:]

            try:
                if not ccx:
                    raise RuntimeError('CalculiX not found: set the ccx path in Settings')
                m = j['material']
                res = pipeline.run_cell(j['pattern'], j['rho'], j['n_t'], str(self.dir / jid), ccx, E=m['E'],
                                        nu=m['nu'], n_ext=j['n_ext'], threads=int(s['threads']), field=j['field'],
                                        log=log, cancel=lambda j=j: j.get('cancel', False))
                res.update(job_id=jid, job_dir=str(self.dir / jid), material=m['name'], key=j['key'])
                self.reg.add(res)
                j['status'] = 'done'
                if self.on_done:
                    self.on_done(j)
            except Exception as e:                                       # noqa: BLE001
                j['status'] = 'cancelled' if str(e) == 'cancelled' else 'failed'
                j['error'] = str(e)[-2000:]
                log(traceback.format_exc()[-2000:])
            j['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
            j.pop('cancel', None)
            self._save(j)
