"""Mesh convergence of homogenized stiffness entries.

Element size h: the nominal size at the thinnest wall, h = t_min / n_t (mesh.build_mesh). Every other
size in a mesh (grading, cap in the cell interior, gyroid grid spacing) is a fixed multiple of h, so the
meshes of one cell are geometrically similar and h ratios are refinement ratios.

Asymptotic range: p from the three finest meshes is compared with p from the next coarser triple
(gci_prev); similar values mean the error already behaves like a h^p.

Two estimates:
    gci()        Celik et al. (2008), J. Fluids Eng. 130, 078001: three finest meshes, observed order p from
                 the fixed-point iteration, Richardson extrapolation, GCI_fine = Fs e_a / (r^p - 1) with
                 Fs = 1.25 (Roache 1998, three or more meshes). Unequal ratios are allowed.
    fit_all()    least squares f = f0 + a h^p over every level (laws.extrapolate), for a check.

Why stiffness converges although stresses do not: re-entrant wall junctions are corner singularities,
sigma ~ r^(lambda - 1) with 0 < lambda < 1. The peak stress grows without bound under refinement, but the
strain energy, and with it every C_ij (C_ij = energy of a unit average strain), is an integral of
sigma : eps over the cell and stays finite. The singularity lowers the observed order of C_ij below the
smooth-solution value of quadratic elements (2p = 4 for energy) and makes it depend on the corner angle.
"""
import numpy as np

from .laws import extrapolate


def _observed_p(r21, r32, e21, e32, iters=200):
    s = np.sign(e32 / e21) if e21 != 0 else 1.0
    if e21 == 0 or e32 == 0:
        return None, s
    p = abs(np.log(abs(e32 / e21))) / np.log(r21)
    for _ in range(iters):
        q = np.log((r21 ** p - s) / (r32 ** p - s)) if (r21 ** p - s) > 0 and (r32 ** p - s) > 0 else 0.0
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = p_new
    return float(p), float(s)


def gci(h, f, Fs=1.25):
    """h, f: element sizes and values for >= 3 meshes (any order). Uses the three finest.
    Returns dict(p, s, f_ext, e_a21, e_ext21, gci21, gci32, asymptotic, r21, r32, oscillatory) or None.
    Note: with p observed from the same three meshes, asymptotic = GCI32 / (r21^p GCI21) equals f1/f2
    identically, so it says nothing here. study() checks the asymptotic range by comparing p from the
    three finest meshes with p from the next coarser triple instead."""
    h, f = np.asarray(h, float), np.asarray(f, float)
    o = np.argsort(h)
    h, f = h[o], f[o]
    if len(h) < 3 or np.any(np.diff(h[:3]) <= 0):
        return None
    h1, h2, h3 = h[:3]
    f1, f2, f3 = f[:3]
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    p, s = _observed_p(r21, r32, e21, e32)
    if p is None or p == 0:
        return dict(p=p, s=s, f_ext=float(f1), e_a21=0.0, e_ext21=0.0, gci21=0.0, gci32=0.0, asymptotic=None,
                    r21=r21, r32=r32, oscillatory=False)
    f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    e_a21 = abs((f1 - f2) / f1)
    e_a32 = abs((f2 - f3) / f2)
    gci21 = Fs * e_a21 / (r21 ** p - 1)
    gci32 = Fs * e_a32 / (r32 ** p - 1)
    asym = gci32 / (r21 ** p * gci21) if gci21 > 0 else None
    return dict(p=p, s=s, f_ext=float(f_ext), e_a21=float(e_a21), e_ext21=float(abs((f_ext - f1) / f_ext)),
                gci21=float(gci21), gci32=float(gci32), asymptotic=None if asym is None else float(asym),
                r21=float(r21), r32=float(r32), oscillatory=bool(s < 0))


def fit_all(h, f):
    return extrapolate(h, f)


def study(levels, entries, Fs=1.25):
    """levels: list of dict(h, n_t, nodes, C) for one cell; entries: {name: (i, j)} 1-based.
    Per entry: values, GCI (three finest), least-squares fit; plus the error of every level against
    the extrapolated value."""
    levels = sorted(levels, key=lambda d: -d['h'])                 # coarse -> fine
    h = np.array([d['h'] for d in levels])
    out = dict(h=h.tolist(), n_t=[d['n_t'] for d in levels], nodes=[d.get('nodes') for d in levels], entries={})
    for name, (i, j) in entries.items():
        f = np.array([d['C'][i - 1][j - 1] for d in levels])
        g = gci(h, f, Fs)
        o = np.argsort(h)                                          # fine -> coarse
        g_prev = gci(h[o][1:4], f[o][1:4], Fs) if len(h) >= 4 else None
        fit = fit_all(h, f) if len(h) >= 2 else None
        ref = g['f_ext'] if g else (fit['f0'] if fit else f[-1])
        out['entries'][name] = dict(values=f.tolist(), gci=g, gci_prev=g_prev, fit=fit, f_ref=float(ref),
                                    rel_err=((f - ref) / ref).tolist() if ref else None)
    return out
