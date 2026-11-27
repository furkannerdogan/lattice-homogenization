"""CalculiX (ccx) executable: locate, check, run.

Search order for the solver path:
    1. explicit path (argument, app settings or `python -m lattice_app config --ccx PATH`)
    2. environment variable LATTICE_CCX (or CCX)
    3. executables on PATH with a known name
    4. common install folders (C:\\calculix, PrePoMax Solver folder, /usr/bin, ...)
Known names, in order: ccx_i4_static, ccx_static, ccx, ccx_2.23 ... ccx_2.17, ccx_MT, ccx_i8_static
(with .exe on Windows). The i4 builds use 32-bit integers: fine up to roughly 2e9 matrix entries,
far above the cell sizes used here.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import time

WINDOWS = sys.platform.startswith('win')
CCX_NAMES = ['ccx_i4_static', 'ccx_static', 'ccx'] + [f'ccx_2.{k}' for k in range(23, 16, -1)] + \
            ['ccx_MT', 'ccx_i8_static', 'ccx_i8']
CGX_NAMES = ['cgx_STATIC', 'cgx_static', 'cgx']


def _dirs():
    if WINDOWS:
        home = os.path.expanduser('~')
        roots = [r'C:\calculix', r'C:\CalculiX', r'C:\Program Files\calculix', r'C:\Program Files\CalculiX',
                 r'C:\Program Files (x86)\bConverged\CalculiX\ccx', r'C:\bConverged\CalculiX\ccx',
                 os.path.join(home, 'calculix'), os.path.join(home, 'Desktop', 'calculix'),
                 os.path.join(home, 'Downloads', 'calculix')]
        roots += glob.glob(os.path.join(home, '**', 'PrePoMax*', 'Solver'), recursive=False)
        roots += glob.glob(r'C:\PrePoMax*\Solver') + glob.glob(os.path.join(home, 'PrePoMax*', 'Solver'))
        out = []
        for r in roots:                                    # the zip may unpack into a subfolder (bin, ccx, ...)
            out += [r, os.path.join(r, 'bin'), os.path.join(r, 'ccx')]
            out += glob.glob(os.path.join(r, '*'))
        return [d for d in out if os.path.isdir(d)]
    return [d for d in ('/usr/bin', '/usr/local/bin', '/opt/calculix/bin', os.path.expanduser('~/calculix'),
                        os.path.expanduser('~/bin')) if os.path.isdir(d)]


def _candidates(names):
    ext = ['.exe', ''] if WINDOWS else ['']
    for n in names:
        p = shutil.which(n)
        if p:
            yield p
    for d in _dirs():
        for n in names:
            for e in ext:
                p = os.path.join(d, n + e)
                if os.path.isfile(p):
                    yield p


def find_ccx(explicit=None):
    """Path of the ccx executable or None."""
    for p in (explicit, os.environ.get('LATTICE_CCX'), os.environ.get('CCX')):
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return next(_candidates(CCX_NAMES), None)


def find_cgx(explicit=None):
    for p in (explicit, os.environ.get('LATTICE_CGX')):
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return next(_candidates(CGX_NAMES), None)


def all_candidates():
    """Every ccx/cgx found (for the settings page)."""
    return dict(ccx=list(dict.fromkeys(_candidates(CCX_NAMES))), cgx=list(dict.fromkeys(_candidates(CGX_NAMES))))


def version(ccx):
    """'2.22' etc. or raises. `ccx -v` prints the version and exits."""
    r = subprocess.run([ccx, '-v'], capture_output=True, text=True, timeout=30,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    m = re.search(r'[Vv]ersion\s+([\d.]+)', r.stdout + r.stderr)
    if not m:
        raise RuntimeError(f'{ccx} -v gave no version: {(r.stdout + r.stderr)[-300:]}')
    return m.group(1)


def check(ccx=None):
    """dict(ok, path, version, message) for the settings page and `doctor`."""
    path = find_ccx(ccx)
    if not path:
        return dict(ok=False, path=None, version=None,
                    message='ccx not found. Set its full path in Settings or with '
                            '`python -m lattice_app config --ccx C:\\calculix\\ccx_i4_static.exe`.')
    try:
        return dict(ok=True, path=path, version=version(path), message='ok')
    except Exception as e:                                          # noqa: BLE001
        return dict(ok=False, path=path, version=None, message=str(e))


def run(ccx, job_dir, job_name, threads=2, log=None, cancel=None, timeout=None):
    """Runs `ccx -i job_name` in job_dir. log: callable(str) for every output line; cancel: callable() -> bool.
    Returns the full output. Raises RuntimeError if ccx does not report 'Job finished'."""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), CCX_NPROC_EQUATION_SOLVER=str(threads),
               CCX_NPROC_STIFFNESS=str(threads), CCX_NPROC_RESULTS=str(threads))
    p = subprocess.Popen([ccx, '-i', job_name], cwd=job_dir, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    out, t0 = [], time.time()
    for line in p.stdout:
        out.append(line)
        if log:
            log(line.rstrip())
        if (cancel and cancel()) or (timeout and time.time() - t0 > timeout):
            p.kill()
            p.wait()
            raise RuntimeError('cancelled' if cancel and cancel() else 'timeout')
    p.wait()
    text = ''.join(out)
    if 'Job finished' not in text:
        raise RuntimeError('ccx did not finish:\n' + text[-3000:])
    return text
