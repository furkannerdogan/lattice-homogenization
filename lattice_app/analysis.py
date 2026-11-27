"""Everything the explorer shows, computed from the registry with the lattice package.

Continuous rho. The FE matrices of the selected source (production mesh, one per density) are symmetrized,
rotated to the print frame (x, y in the print plane, z = build direction) and reduced to the independent
engineering constants of the cell's symmetry class:
    cubic, gyroid         E, G, nu
    grid (tetragonal, z)  E_x = E_y, E_z, G_xy, G_xz = G_yz, nu_xy, nu_zx = nu_zy
    triangles, honeycomb  E_x = E_y, E_z, G_xz = G_yz, nu_xy, nu_zx;  G_xy = E_x / (2 (1 + nu_xy))
Moduli follow density laws E/E_s = h(rho), G/G_s = h(rho) (laws.select_law: mixture or power-mixture law,
chosen by leave-one-out error; h(1) = 1). Poisson ratios are interpolated monotonically (PCHIP) through
the FE points and nu_s at rho = 1. C(rho) is assembled from these constants.

Why constants and not the C_ij entries: the in-plane moduli of bending-dominated cells are small
differences of large C_ij (honeycomb at rho = 0.1: E_x / E_s = 0.002 from entries of order 0.02-0.05), so
separate fits of the entries amplify their fit error. Interior leave-one-out on the Abaqus data: honeycomb
E_x 9.9 % (entry laws) vs 1.8 % (modulus laws), triangles 3.9 % vs 0.5 %, grid nu_xy 56 % vs 0.9 %.
The C_ij entry laws (HD convention) are still reported in the laws tab.
Linear elasticity: C scales with E_s; records are used only when their nu equals the material's nu.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

from lattice import convergence, laws, post
from lattice.geometry import PATTERNS, smallest_feature

PRODUCTION_NT = {'grid': 8, 'triangles': 3, 'honeycomb': 3, 'cubic': 3, 'gyroid': 2}
RHO_RANGE = (0.1, 0.7)                          # FE data range; outside it the laws extrapolate
RHO_MIN = 0.05                                  # below this the extrapolation is not trusted at all


def iso_C(E, nu):
    return post.iso_C(E, nu)


def solid_KG(E, nu):
    return E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu))


def hs_upper(rho, E, nu):
    """Hashin-Shtrikman upper bound, isotropic solid + voids (Hashin & Shtrikman 1963)."""
    K, G = solid_KG(E, nu)
    rho = np.asarray(rho, float)
    v = 1 - rho
    Ku = K + v / (-1 / K + 3 * rho / (3 * K + 4 * G))
    Gu = G + v / (-1 / G + 6 * rho * (K + 2 * G) / (5 * G * (3 * K + 4 * G)))
    Eu = 9 * Ku * Gu / (3 * Ku + Gu)
    return dict(K=Ku, G=Gu, E=Eu)


def symmetry_class(pattern):
    return 'cubic' if pattern in ('cubic', 'gyroid') else ('tetragonal' if pattern == 'grid' else 'hexagonal')


def reduced_constants(pattern, Cp):
    """Independent engineering constants (print frame), symmetry-averaged."""
    c = post.engineering_constants(Cp)
    cls = symmetry_class(pattern)
    if cls == 'cubic':
        return dict(E=np.mean([c['E1'], c['E2'], c['E3']]), G=np.mean([c['G23'], c['G13'], c['G12']]),
                    nu=np.mean([c['nu12'], c['nu13'], c['nu23']]))
    d = dict(E_x=0.5 * (c['E1'] + c['E2']), E_z=c['E3'], G_xz=0.5 * (c['G13'] + c['G23']), nu_xy=c['nu12'],
             nu_zx=0.5 * (c['nu31'] + c['nu32']))
    if cls == 'tetragonal':
        d['G_xy'] = c['G12']
    return d


def C_from_constants(pattern, d):
    """Print-frame C from reduced_constants(). Raises ValueError if not positive definite."""
    S = np.zeros((6, 6))
    if symmetry_class(pattern) == 'cubic':
        S[:3, :3] = -d['nu'] / d['E']
        S[[0, 1, 2], [0, 1, 2]] = 1 / d['E']
        S[[3, 4, 5], [3, 4, 5]] = 1 / d['G']
    else:
        Gxy = d['G_xy'] if 'G_xy' in d else d['E_x'] / (2 * (1 + d['nu_xy']))
        S[0, 0] = S[1, 1] = 1 / d['E_x']
        S[2, 2] = 1 / d['E_z']
        S[0, 1] = S[1, 0] = -d['nu_xy'] / d['E_x']
        S[0, 2] = S[2, 0] = S[1, 2] = S[2, 1] = -d['nu_zx'] / d['E_z']
        S[3, 3] = S[4, 4] = 1 / d['G_xz']
        S[5, 5] = 1 / Gxy
    if not np.all(np.linalg.eigvalsh(S) > 0):
        raise ValueError('the density laws give no positive-definite stiffness here (extrapolation failed)')
    C = np.linalg.inv(S)
    return 0.5 * (C + C.T)


class Analysis:
    def __init__(self, registry):
        self.reg = registry
        self._cache = {}

    # ------------------------------------------------------------------ data selection
    def points(self, pattern, source, nu, level='production'):
        """One record per rho: the production mesh if present, else the finest."""
        recs = self.reg.all(source=source, nu=nu, pattern=pattern)
        by = {}
        for r in recs:
            by.setdefault(round(r['rho'], 4), []).append(r)
        out = []
        for rho, g in sorted(by.items()):
            prod = [r for r in g if abs(r['n_t'] - PRODUCTION_NT.get(pattern, -1)) < 1e-9]
            if level == 'production' and prod:
                out.append(prod[0])
            else:
                out.append(max(g, key=lambda r: r['n_t']))
        return out

    def fitted(self, pattern, source, nu):
        key = ('fit', pattern, source, round(nu, 6), self.reg.version)
        if key in self._cache:
            return self._cache[key]
        pts = self.points(pattern, source, nu)
        res = None
        if len(pts) >= 4:
            rho = np.array([r['rho'] for r in pts])
            res = dict(rho=rho.tolist(), records=[r['key'] for r in pts], n_t=[r['n_t'] for r in pts],
                       cls=symmetry_class(pattern), moduli={}, poisson={}, entries={})
            cons = [reduced_constants(pattern, post.to_print_frame(post.symmetrize(np.array(r['C']), pattern),
                                                                    pattern)) for r in pts]
            Es = np.array([r['E'] for r in pts])
            Gs = Es / (2 * (1 + nu))
            for name in cons[0]:
                v = np.array([c[name] for c in cons])
                if name.startswith('nu'):
                    res['poisson'][name] = dict(y=v.tolist(), pchip=PchipInterpolator(np.r_[rho, 1.0], np.r_[v, nu]))
                else:
                    y = v / (Es if name.startswith('E') else Gs)
                    res['moduli'][name] = dict(y=y.tolist(), law=laws.select_law(rho, y), power=laws.power_law(rho, y))
            for name, (i, j) in post.independent_entries(pattern).items():     # C_ij laws, reported only
                y = np.array([post.symmetrize(np.array(r['C']), pattern)[i - 1, j - 1] / iso_C(r['E'], nu)[i - 1, j - 1]
                              for r in pts])
                res['entries'][name] = dict(ij=(i, j), y=y.tolist(), law=laws.select_law(rho, y),
                                            power=laws.power_law(rho, y))
        self._cache[key] = res
        return res

    def constants_at(self, pattern, rho, source, E, nu):
        f = self.fitted(pattern, source, nu)
        if f is None:
            return None
        G = E / (2 * (1 + nu))
        d = {n: float(laws.evaluate(m['law'], np.array([rho]))[0]) * (E if n.startswith('E') else G)
             for n, m in f['moduli'].items()}
        d.update({n: float(p['pchip'](rho)) for n, p in f['poisson'].items()})
        return d

    def C_at(self, pattern, rho, source, E, nu):
        """C in the print frame, or None (no data). Raises ValueError where the laws give no valid C."""
        d = self.constants_at(pattern, rho, source, E, nu)
        if d is None:
            return None
        return C_from_constants(pattern, d)

    def fe_C(self, pattern, rho, source, E, nu):
        """FE stiffness at a data point (scaled to E), or None."""
        for r in self.points(pattern, source, nu):
            if abs(r['rho'] - rho) < 1e-6:
                return np.array(r['C']) * E / r['E'], r
        return None, None

    # ------------------------------------------------------------------ derived quantities
    @staticmethod
    def describe(C_print, E_s):
        ec = post.engineering_constants(C_print)
        au, _ = post.universal_anisotropy(C_print)
        out = dict(constants={k: float(v) for k, v in ec.items()}, A_U=au)
        out['zener'] = post.zener(C_print)
        return out

    @staticmethod
    def surface(C, n_theta=46, n_phi=91):
        th = np.linspace(0, np.pi, n_theta)
        ph = np.linspace(0, 2 * np.pi, n_phi)
        T, P = np.meshgrid(th, ph, indexing='ij')
        n = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
        E = post.directional_E(C, n)
        return dict(x=(E * n[..., 0]).tolist(), y=(E * n[..., 1]).tolist(), z=(E * n[..., 2]).tolist(),
                    E=E.tolist(), Emin=float(E.min()), Emax=float(E.max()))

    @staticmethod
    def polar(C, plane, n=361):
        a = np.linspace(0, 2 * np.pi, n)
        c, s = np.cos(a), np.sin(a)
        z = np.zeros_like(a)
        if plane == 'xy':
            nv, mv = np.stack([c, s, z], -1), np.stack([-s, c, z], -1)
        elif plane == 'xz':
            nv, mv = np.stack([c, z, s], -1), np.stack([-s, z, c], -1)
        else:                                                        # yz
            nv, mv = np.stack([z, c, s], -1), np.stack([z, -s, c], -1)
        E = post.directional_E(C, nv)
        G = post.directional_G(C, nv, mv)
        return dict(angle=np.degrees(a).tolist(), E=E.tolist(), G=G.tolist())

    @staticmethod
    def virtual_tests(C_print):
        """Responses of stress-controlled tests (print frame) predicted from C. Engineering shear."""
        S = np.linalg.inv(C_print)
        out = []
        ax = 'xyz'
        for a in range(3):
            eps = S[:, a]
            out.append(dict(test=f'UT_{ax[a]}', description=f'uniaxial tension along {ax[a]}',
                            values={'E': 1 / eps[a], **{f'nu_{ax[a]}{ax[b]}': -eps[b] / eps[a]
                                                         for b in range(3) if b != a}}))
        for a, b in ((0, 1), (1, 2), (0, 2)):
            sig = np.zeros(6)
            sig[a] = sig[b] = 1.0
            eps = S @ sig
            out.append(dict(test=f'ET_{ax[a]}{ax[b]}', description=f'equibiaxial tension in the {ax[a]}-{ax[b]} plane',
                            values={'E_eb': 1 / (0.5 * (eps[a] + eps[b])), f'eps_{ax[a]}/eps_{ax[b]}': eps[a] / eps[b]}))
        for k, pl in ((5, 'xy'), (3, 'yz'), (4, 'xz')):
            out.append(dict(test=f'SS_{pl}', description=f'simple shear in the {pl[0]}-{pl[1]} plane',
                            values={'G': 1 / S[k, k]}))
        return out

    def explore(self, pattern, rho, source, material):
        E, nu = material['E'], material['nu']
        try:
            Cp = self.C_at(pattern, rho, source, E, nu)
        except ValueError as e:
            return dict(ok=False, message=f'{pattern}, rho = {rho}: {e}')
        if Cp is None:
            return dict(ok=False, message=f'No {source} data for {pattern} with nu = {nu} (need >= 4 densities). '
                                          'Run jobs or choose another source/material.')
        fe, rec = self.fe_C(pattern, rho, source, E, nu)
        out = dict(ok=True, pattern=pattern, rho=rho, source=source, material=material,
                   extrapolated=not (RHO_RANGE[0] - 1e-9 <= rho <= RHO_RANGE[1] + 1e-9),
                   C_print=Cp.tolist(), C_fe=post.to_fe_frame(Cp, pattern).tolist(), C_solid=iso_C(E, nu).tolist(),
                   reduced=self.constants_at(pattern, rho, source, E, nu), symmetry=symmetry_class(pattern),
                   surface=self.surface(Cp), polar={p: self.polar(Cp, p) for p in ('xy', 'xz', 'yz')},
                   tests=self.virtual_tests(Cp), hs=dict((k, float(v)) for k, v in hs_upper(rho, E, nu).items()))
        out.update(self.describe(Cp, E))
        if fe is not None:
            fep = post.to_print_frame(post.symmetrize(fe, pattern), pattern)
            out['fe'] = dict(C_print=fep.tolist(), n_t=rec['n_t'], nodes=rec.get('nodes'), key=rec['key'],
                             law_vs_fe=float(np.max(np.abs(Cp - fep)) / np.max(np.abs(fep))))
        return out

    def trends(self, source, material, rhos=None):
        """Engineering constants (print frame) vs rho for every pattern: law curves and FE points."""
        E, nu = material['E'], material['nu']
        rr = np.round(np.linspace(RHO_MIN, 0.95, 86), 4) if rhos is None else np.asarray(rhos)
        out = dict(rho=rr.tolist(), patterns={}, hs=dict((k, v.tolist()) for k, v in hs_upper(rr, E, nu).items()),
                   E_s=E)
        keys = ['E1', 'E2', 'E3', 'G23', 'G13', 'G12', 'nu12', 'nu13', 'nu23', 'nu21', 'nu31', 'nu32']
        for p in PATTERNS:
            if self.fitted(p, source, nu) is None:
                continue
            curves = {k: [] for k in keys + ['A_U', 'Emax_Emin']}
            for r in rr:
                try:
                    Cp = self.C_at(p, r, source, E, nu)
                except ValueError:
                    for k in curves:
                        curves[k].append(None)
                    continue
                d = self.describe(Cp, E)
                for k in keys:
                    curves[k].append(d['constants'][k])
                curves['A_U'].append(d['A_U'])
                s = self.surface(Cp, 19, 37)
                curves['Emax_Emin'].append(s['Emax'] / s['Emin'])
            pts = {k: [] for k in keys + ['A_U']}
            prho = []
            for rec in self.points(p, source, nu):
                Cp = post.to_print_frame(np.array(rec['C']) * E / rec['E'], p)
                d = self.describe(Cp, E)
                prho.append(rec['rho'])
                for k in keys:
                    pts[k].append(d['constants'][k])
                pts['A_U'].append(d['A_U'])
            out['patterns'][p] = dict(curves=curves, points=dict(rho=prho, **pts))
        return out

    @staticmethod
    def _law_summary(e, rr):
        law = e['law']
        ch = law[law['chosen']]
        other = 'power_mixture' if law['chosen'] == 'mixture' else 'mixture'
        return dict(y=e['y'], curve=laws.evaluate(law, rr).tolist(), power_curve=(e['power']['a'] * rr ** e['power']['n']).tolist(),
                    chosen=law['chosen'], params=dict(d=ch['d'], coeffs=ch['coeffs'], n=ch.get('n')),
                    loo_rms=ch['loo_rms'], loo_max=ch['loo_max'], fit_max=ch['fit_max'], loo_table=ch['loo'],
                    other=dict(name=other, loo_rms=law[other]['loo_rms']), power=e['power'])

    def law_detail(self, pattern, source, material):
        f = self.fitted(pattern, source, material['nu'])
        if f is None:
            return dict(ok=False)
        rr = np.linspace(0.02, 1.0, 120)
        rp = np.linspace(RHO_MIN, 1.0, 120)
        return dict(ok=True, rho=f['rho'], n_t=f['n_t'], curve_rho=rr.tolist(), symmetry=f['cls'],
                    moduli={n: self._law_summary(e, rr) for n, e in f['moduli'].items()},
                    poisson={n: dict(y=p['y'], rho=rp.tolist(), curve=p['pchip'](rp).tolist())
                             for n, p in f['poisson'].items()},
                    entries={n: self._law_summary(e, rr) for n, e in f['entries'].items()})

    # ------------------------------------------------------------------ mesh convergence
    def convergence_sets(self, source, nu):
        groups = {}
        for r in self.reg.all(source=source, nu=nu):
            groups.setdefault((r['pattern'], round(r['rho'], 4)), set()).add(r['n_t'])
        return sorted([dict(pattern=p, rho=rho, levels=sorted(v)) for (p, rho), v in groups.items() if len(v) >= 2],
                      key=lambda d: (PATTERNS.index(d['pattern']) if d['pattern'] in PATTERNS else 99, d['rho']))

    def convergence(self, pattern, rho, source, nu):
        recs = [r for r in self.reg.all(source=source, nu=nu, pattern=pattern) if abs(r['rho'] - rho) < 1e-6]
        lv = {}
        for r in recs:
            lv[r['n_t']] = r
        if len(lv) < 2:
            return dict(ok=False, message='need at least two mesh levels')
        tmin = smallest_feature(pattern, rho)
        levels = [dict(h=tmin / nt, n_t=nt, nodes=r.get('nodes'), C=post.symmetrize(np.array(r['C']), pattern).tolist())
                  for nt, r in lv.items()]
        st = convergence.study(levels, post.independent_entries(pattern))
        st.update(ok=True, pattern=pattern, rho=rho, t_min=tmin, production_nt=PRODUCTION_NT.get(pattern))
        return st
