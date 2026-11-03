"""Density laws for the homogenized stiffness and mesh-convergence extrapolation.

Mixture law (the form of the HD repository's polynomial_law with a void 'matrix'):
    C_ij(rho) = C_ij^s h(rho),  h(rho) = rho + rho (1 - rho) sum_{i<d} c_i rho^i,  h(0) = 0, h(1) = 1
    Doruk's g:  g = (P_RVE - P_f) / (P_m - P_f) with P_m = 0 (void), P_f = P_s (solid)  ->  g = 1 - h
The number of coefficients d is chosen by leave-one-out cross-validation over the densities: each
density is left out once, the law is fitted to the others and predicts it. The fit minimises
relative errors, so the small bending-dominated values at low density count as much as the large ones.

Power-mixture law (for bending-dominated entries, which fall like rho^2 ... rho^3 and are not
captured by the mixture polynomial over 0.1 ... 0.7):
    h(rho) = rho^n [1 + (1 - rho) sum_{i<d} b_i rho^i],  h(0) = 0, h(1) = 1, n fitted
select_law() keeps whichever of the two forms predicts the left-out densities better.

Power law (Gibson-Ashby form, for the scaling exponent): C_ij / C_ij^s = a rho^n.
"""
import numpy as np


def _design(rho, d):
    rho = np.asarray(rho, dtype=float)
    return np.stack([rho * (1 - rho) * rho ** i for i in range(d)], 1) if d else np.zeros((len(rho), 0))


def fit_mixture(rho, y, d):
    rho, y = np.asarray(rho, float), np.asarray(y, float)
    if d == 0:
        return np.zeros(0)
    A = _design(rho, d) / y[:, None]
    b = (y - rho) / y
    return np.linalg.lstsq(A, b, rcond=None)[0]


def mixture(rho, c):
    rho = np.asarray(rho, dtype=float)
    return rho + (_design(rho, len(c)) @ c if len(c) else 0.0)


def fit_power(rho, y):
    n, loga = np.polyfit(np.log(rho), np.log(y), 1)
    return np.array([np.exp(loga), n])


def power(rho, p):
    return p[0] * np.asarray(rho, float) ** p[1]


def loo(rho, y, fit, predict):
    rho, y = np.asarray(rho, float), np.asarray(y, float)
    err = []
    for k in range(len(rho)):
        m = np.arange(len(rho)) != k
        p = fit(rho[m], y[m])
        err.append((predict(rho[k:k + 1], p)[0] - y[k]) / y[k])
    return np.array(err)


def select_mixture(rho, y, d_max=5):
    """Number of coefficients with the smallest leave-one-out RMS relative error."""
    rows = []
    for d in range(0, d_max + 1):
        if d > len(rho) - 2:
            break
        e = loo(rho, y, lambda r, v, d=d: fit_mixture(r, v, d), mixture)
        rows.append(dict(d=d, loo_rms=float(np.sqrt(np.mean(e ** 2))), loo_max=float(np.abs(e).max())))
    best = min(rows, key=lambda r: r['loo_rms'])
    c = fit_mixture(rho, y, best['d'])
    fit_err = (mixture(rho, c) - y) / y
    return dict(d=best['d'], coeffs=c.tolist(), loo=rows, loo_rms=best['loo_rms'], loo_max=best['loo_max'],
                fit_max=float(np.abs(fit_err).max()))


def fit_powmix(rho, y, d):
    rho, y = np.asarray(rho, float), np.asarray(y, float)
    best = None
    for n in np.arange(0.8, 4.501, 0.01):
        base = rho ** n
        A = (base * (1 - rho))[:, None] * np.stack([rho ** i for i in range(d)], 1) / y[:, None] if d else np.zeros((len(rho), 0))
        b = np.linalg.lstsq(A, (y - base) / y, rcond=None)[0] if d else np.zeros(0)
        r = np.sum((A @ b - (y - base) / y) ** 2) if d else np.sum(((base - y) / y) ** 2)
        if best is None or r < best[0]:
            best = (r, n, b)
    return np.concatenate([[best[1]], best[2]])


def powmix(rho, p):
    rho = np.asarray(rho, float)
    n, b = p[0], p[1:]
    q = sum(bi * rho ** i for i, bi in enumerate(b)) if len(b) else 0.0
    return rho ** n * (1 + (1 - rho) * q)


def select_law(rho, y, d_max=5):
    """Best of the mixture law and the power-mixture law by leave-one-out RMS relative error."""
    mix = select_mixture(rho, y, d_max)
    rows = []
    for d in range(0, 4):
        if d + 1 > len(rho) - 2:
            break
        e = loo(rho, y, lambda r, v, d=d: fit_powmix(r, v, d), powmix)
        rows.append(dict(d=d, loo_rms=float(np.sqrt(np.mean(e ** 2))), loo_max=float(np.abs(e).max())))
    bp = min(rows, key=lambda r: r['loo_rms'])
    p = fit_powmix(rho, y, bp['d'])
    pm = dict(d=bp['d'], n=float(p[0]), coeffs=p[1:].tolist(), loo=rows, loo_rms=bp['loo_rms'], loo_max=bp['loo_max'],
              fit_max=float(np.abs((powmix(rho, p) - y) / y).max()))
    chosen = 'mixture' if mix['loo_rms'] <= pm['loo_rms'] else 'power_mixture'
    return dict(chosen=chosen, mixture=mix, power_mixture=pm)


def evaluate(law, rho):
    """h(rho) of a select_law() result."""
    if law['chosen'] == 'mixture':
        return mixture(rho, np.array(law['mixture']['coeffs']))
    pm = law['power_mixture']
    return powmix(rho, np.concatenate([[pm['n']], pm['coeffs']]))


def power_law(rho, y):
    p = fit_power(rho, y)
    e = loo(rho, y, fit_power, power)
    return dict(a=float(p[0]), n=float(p[1]), loo_rms=float(np.sqrt(np.mean(e ** 2))),
                fit_max=float(np.abs((power(rho, p) - y) / y).max()))


def extrapolate(h, f):
    """f(h) = f0 + a h^p fitted to >= 3 mesh levels (p searched in [0.5, 8]); returns f0, p and the
    relative error of every level against f0. With 2 levels p = 2 is assumed."""
    h, f = np.asarray(h, float), np.asarray(f, float)
    if len(h) < 2:
        return dict(f0=float(f[0]), p=None, rel_err=[0.0])
    if len(h) == 2:
        p = 2.0
        a = (f[0] - f[1]) / (h[0] ** p - h[1] ** p)
        f0 = f[1] - a * h[1] ** p
        return dict(f0=float(f0), p=p, rel_err=((f - f0) / f0).tolist(), assumed_p=True)
    best = None
    for p in np.linspace(0.5, 8, 301):
        A = np.column_stack([np.ones_like(h), h ** p])
        sol, res, *_ = np.linalg.lstsq(A, f, rcond=None)
        r = np.sum((A @ sol - f) ** 2)
        if best is None or r < best[0]:
            best = (r, p, sol)
    _, p, (f0, a) = best
    return dict(f0=float(f0), p=float(p), rel_err=((f - f0) / f0).tolist(), assumed_p=False)
