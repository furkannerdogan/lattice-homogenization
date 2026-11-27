"""Workspace location and settings.

Workspace (results registry, jobs, settings, user materials):
    LATTICE_WORKSPACE env var, else <repo>/workspace for a source checkout, else ~/lattice-workspace.
Settings file: <workspace>/settings.json. The solver path can also come from LATTICE_CCX.
"""
import json
import os
from pathlib import Path

DEFAULTS = dict(ccx_path='', cgx_path='', threads=max(1, min(4, (os.cpu_count() or 2) // 2)), field_output=True,
                source='abaqus', material='PLA', host='127.0.0.1', port=8765)


def repo_root():
    p = Path(__file__).resolve().parent.parent
    return p if (p / 'pyproject.toml').exists() else None


def workspace():
    w = os.environ.get('LATTICE_WORKSPACE')
    if w:
        p = Path(w)
    elif repo_root():
        p = repo_root() / 'workspace'
    else:
        p = Path.home() / 'lattice-workspace'
    for sub in ('', 'records', 'jobs'):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p


def load():
    s = dict(DEFAULTS)
    f = workspace() / 'settings.json'
    if f.exists():
        try:
            s.update(json.loads(f.read_text()))
        except json.JSONDecodeError:
            pass
    return s


def save(updates):
    s = load()
    s.update({k: v for k, v in updates.items() if k in DEFAULTS})
    (workspace() / 'settings.json').write_text(json.dumps(s, indent=1))
    return s
