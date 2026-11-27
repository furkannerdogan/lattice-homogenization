"""Command line.

python -m lattice_app serve [--port 8765] [--no-browser]   start the app (browser opens at http://127.0.0.1:8765)
python -m lattice_app config --ccx PATH [--cgx PATH] [--threads N]
python -m lattice_app config                             show settings
python -m lattice_app doctor                             check Python packages, ccx, workspace
python -m lattice_app run grid 0.3 --nt 2 4 8            run CalculiX jobs without the browser
"""
import argparse
import json
import sys
import threading
import time
import webbrowser


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m lattice_app')
    sub = ap.add_subparsers(dest='cmd')
    s = sub.add_parser('serve')
    s.add_argument('--host', default=None)
    s.add_argument('--port', type=int, default=None)
    s.add_argument('--no-browser', action='store_true')
    c = sub.add_parser('config')
    c.add_argument('--ccx')
    c.add_argument('--cgx')
    c.add_argument('--threads', type=int)
    sub.add_parser('doctor')
    r = sub.add_parser('run')
    r.add_argument('pattern')
    r.add_argument('rho', type=float)
    r.add_argument('--nt', type=float, nargs='+', default=[2])
    r.add_argument('--material', default='PLA')
    r.add_argument('--no-field', action='store_true')
    a = ap.parse_args(argv)

    from . import config
    if a.cmd == 'config':
        upd = {k: v for k, v in dict(ccx_path=a.ccx, cgx_path=a.cgx, threads=a.threads).items() if v is not None}
        s = config.save(upd) if upd else config.load()
        from lattice import solver
        print(json.dumps(s, indent=1))
        print('ccx:', solver.check(s.get('ccx_path') or None))
        return
    if a.cmd == 'doctor':
        return doctor()
    if a.cmd == 'run':
        return run_cli(a)
    if a.cmd in (None, 'serve'):
        import uvicorn

        from .server import create_app
        st = config.load()
        host = getattr(a, 'host', None) or st['host']
        port = getattr(a, 'port', None) or st['port']
        url = f'http://{host}:{port}'
        print(f'Lattice homogenization app: {url}   (workspace: {config.workspace()})   Ctrl+C to stop')
        if not getattr(a, 'no_browser', False):
            threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
        uvicorn.run(create_app(), host=host, port=port, log_level='warning')


def doctor():
    import importlib
    import platform
    ok = True
    print(f'Python {platform.python_version()} ({sys.executable})')
    for mod, need in (('numpy', True), ('scipy', True), ('fastapi', True), ('uvicorn', True),
                      ('gmsh', False), ('trimesh', False), ('skimage', False), ('matplotlib', False)):
        try:
            m = importlib.import_module(mod)
            print(f'  {mod:11s} {getattr(m, "__version__", "")}')
        except Exception as e:                                          # noqa: BLE001
            ok &= not need
            print(f'  {mod:11s} MISSING{" (required)" if need else " (optional: CAD/figures)"}: {e}')
    from lattice import solver

    from . import config
    s = config.load()
    chk = solver.check(s.get('ccx_path') or None)
    print('ccx:', chk['path'], chk['version'] or '', '' if chk['ok'] else chk['message'])
    print('cgx:', solver.find_cgx(s.get('cgx_path') or None))
    print('workspace:', config.workspace())
    print('all ok' if ok and chk['ok'] else 'see above')


def run_cli(a):
    from lattice import materials, pipeline, solver

    from . import config
    from .store import Registry, make_key
    s = config.load()
    ccx = solver.find_ccx(s.get('ccx_path') or None)
    if not ccx:
        sys.exit('ccx not found; python -m lattice_app config --ccx PATH')
    ws = config.workspace()
    reg = Registry(ws)
    m = materials.load(str(ws))[a.material]
    for nt in a.nt:
        key = make_key('calculix', a.pattern, a.rho, nt, m['E'], m['nu'], mesh_version=pipeline.MESH_VERSION)
        if reg.get(key):
            print(f'{a.pattern} rho={a.rho} n_t={nt}: cached')
            continue
        jid = time.strftime('%Y%m%d-%H%M%S-cli')
        d = ws / 'jobs' / jid
        res = pipeline.run_cell(a.pattern, a.rho, nt, str(d), ccx, E=m['E'], nu=m['nu'], threads=int(s['threads']),
                                field=not a.no_field, log=print)
        res.update(job_id=jid, job_dir=str(d), material=m['name'], key=key)
        reg.add(res)


if __name__ == '__main__':
    main()
