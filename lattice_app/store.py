"""Results registry.

A record is one homogenized stiffness: pattern, rho, mesh level n_t, solid E and nu, source, C (Voigt,
FE frame, MPa). Sources:
    abaqus        bundled dataset, Abaqus/Standard runs of the decks from scripts/make_decks.py
    calculix-ref  bundled dataset, CalculiX 2.21 runs from the same decks (lattice_app/data/calculix-ref.json)
    calculix      runs made from this app (one JSON per record in <workspace>/records)
Bundled records are read-only. The key is a hash of everything that defines the FE problem, so a job
whose key exists is not run again (cache).
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

from lattice.geometry import L_DEFAULT, smallest_feature

DATA = Path(__file__).resolve().parent / 'data'
SOURCES = {'abaqus': 'Abaqus/Standard (bundled)', 'calculix-ref': 'CalculiX reference (bundled)',
           'calculix': 'CalculiX runs from this app'}


def make_key(source, pattern, rho, n_t, E, nu, n_ext=4, L=L_DEFAULT, mesh_version=1):
    kind = 'calculix' if source.startswith('calculix') else source
    s = json.dumps([kind, pattern, round(float(rho), 4), round(float(n_t), 3), int(n_ext), round(float(L), 4),
                    round(float(E), 3), round(float(nu), 5), int(mesh_version)])
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def normalize(rec, source=None, origin='local'):
    r = dict(rec)
    r['source'] = source or r.get('source')
    if r['source'] in ('abaqus-odb', 'abaqus-dat'):
        r['source'] = 'abaqus'
    r.setdefault('E', 2100.0)
    r.setdefault('nu', 0.35)
    r.setdefault('L', L_DEFAULT)
    r.setdefault('n_ext', 4)
    r['rho'], r['n_t'] = float(r['rho']), float(r['n_t'])
    r.setdefault('h', smallest_feature(r['pattern'], r['rho'], r['L']) / r['n_t'])
    r['origin'] = origin
    r['key'] = r.get('key') or make_key(r['source'], r['pattern'], r['rho'], r['n_t'], r['E'], r['nu'],
                                        r['n_ext'], r['L'], r.get('mesh_version', 1))
    return r


class Registry:
    def __init__(self, workspace):
        self.dir = Path(workspace) / 'records'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.version = 0
        self.records = {}
        for f in sorted(DATA.glob('*.json')):
            src = f.stem
            for r in json.loads(f.read_text()):
                r = normalize(r, src, 'bundled')
                self.records[r['key']] = r
        for f in sorted(self.dir.glob('*.json')):
            try:
                r = normalize(json.loads(f.read_text()), origin='local')
                self.records[r['key']] = r
            except Exception:                                   # noqa: BLE001
                continue

    def all(self, source=None, nu=None, pattern=None):
        out = []
        for r in self.records.values():
            if source and r['source'] != source:
                continue
            if nu is not None and abs(r['nu'] - nu) > 1e-6:
                continue
            if pattern and r['pattern'] != pattern:
                continue
            out.append(r)
        return out

    def get(self, key):
        return self.records.get(key)

    def add(self, rec):
        r = normalize(rec, origin='local')
        r.setdefault('created', time.strftime('%Y-%m-%d %H:%M:%S'))
        with self.lock:
            (self.dir / f"{r['key']}.json").write_text(json.dumps(r, indent=1, default=float))
            self.records[r['key']] = r
            self.version += 1
        return r

    def delete(self, key):
        r = self.records.get(key)
        if not r or r['origin'] == 'bundled':
            return False
        with self.lock:
            p = self.dir / f'{key}.json'
            if p.exists():
                os.remove(p)
            del self.records[key]
            self.version += 1
        return True

    def sources(self):
        cnt = {}
        for r in self.records.values():
            cnt[r['source']] = cnt.get(r['source'], 0) + 1
        return [dict(id=s, label=SOURCES.get(s, s), count=cnt.get(s, 0)) for s in
                list(SOURCES) + sorted(set(cnt) - set(SOURCES))]
