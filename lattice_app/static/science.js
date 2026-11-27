// Browser port of the derived quantities in lattice/post.py and lattice_app/analysis.py, used by the
// read-only web version. Input: C (6x6, Voigt, engineering shear, print frame). Checked against Python in
// tests/test_static.py.
const W = [1, 1, 1, Math.SQRT2, Math.SQRT2, Math.SQRT2];
const PAIRS = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]];
const PRISMATIC = ['grid', 'triangles', 'honeycomb'];
const FE_TO_PRINT = [1, 2, 0, 4, 5, 3];

export function inv(A) {
  const n = A.length, M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c];
    for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; if (f) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j]; }
  }
  return M.map((r) => r.slice(n));
}
export function fromUpper(u) {
  const C = Array.from({ length: 6 }, () => Array(6).fill(0));
  let k = 0;
  for (let i = 0; i < 6; i++) for (let j = i; j < 6; j++) { C[i][j] = C[j][i] = u[k++]; }
  return C;
}
export function toFe(C, pattern) {
  if (!PRISMATIC.includes(pattern)) return C.map((r) => r.slice());
  const invp = [2, 0, 1, 5, 3, 4];                       // argsort(FE_TO_PRINT)
  return invp.map((i) => invp.map((j) => C[i][j]));
}
export function isoC(E, nu) {
  const l = E * nu / ((1 + nu) * (1 - 2 * nu)), mu = E / (2 * (1 + nu));
  const C = Array.from({ length: 6 }, () => Array(6).fill(0));
  for (let i = 0; i < 3; i++) { for (let j = 0; j < 3; j++) C[i][j] = l; C[i][i] = l + 2 * mu; C[i + 3][i + 3] = mu; }
  return C;
}
function mandelS(C) { return inv(C.map((r, i) => r.map((v, j) => v * W[i] * W[j]))); }
function quad(S, N) { let s = 0; for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) s += N[i] * S[i][j] * N[j]; return s; }
const dirE = (Sm, n) => 1 / quad(Sm, PAIRS.map(([i, j], k) => n[i] * n[j] * W[k]));
const dirG = (Sm, n, m) => 1 / (4 * quad(Sm, PAIRS.map(([i, j], k) => 0.5 * (n[i] * m[j] + n[j] * m[i]) * W[k])));

export function constants(C) {
  const S = inv(C), E = [0, 1, 2].map((i) => 1 / S[i][i]), G = [3, 4, 5].map((i) => 1 / S[i][i]);
  return { E1: E[0], E2: E[1], E3: E[2], G23: G[0], G13: G[1], G12: G[2], nu12: -S[0][1] * E[0], nu13: -S[0][2] * E[0],
    nu23: -S[1][2] * E[1], nu21: -S[0][1] * E[1], nu31: -S[0][2] * E[2], nu32: -S[1][2] * E[2] };
}
export function anisotropyU(C) {
  const S = inv(C);
  const KV = (C[0][0] + C[1][1] + C[2][2] + 2 * (C[0][1] + C[1][2] + C[0][2])) / 9;
  const GV = (C[0][0] + C[1][1] + C[2][2] - (C[0][1] + C[1][2] + C[0][2]) + 3 * (C[3][3] + C[4][4] + C[5][5])) / 15;
  const KR = 1 / (S[0][0] + S[1][1] + S[2][2] + 2 * (S[0][1] + S[1][2] + S[0][2]));
  const GR = 15 / (4 * (S[0][0] + S[1][1] + S[2][2]) - 4 * (S[0][1] + S[1][2] + S[0][2]) + 3 * (S[3][3] + S[4][4] + S[5][5]));
  return 5 * GV / GR + KV / KR - 6;
}
export const zener = (C) => 2 * C[3][3] / (C[0][0] - C[0][1]);

export function surface(C, nth = 46, nph = 91) {
  const Sm = mandelS(C), x = [], y = [], z = [], E = [];
  let Emin = Infinity, Emax = -Infinity;
  for (let a = 0; a < nth; a++) {
    const th = Math.PI * a / (nth - 1), rx = [], ry = [], rz = [], re = [];
    for (let b = 0; b < nph; b++) {
      const ph = 2 * Math.PI * b / (nph - 1), n = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
      const e = dirE(Sm, n);
      rx.push(e * n[0]); ry.push(e * n[1]); rz.push(e * n[2]); re.push(e);
      if (e < Emin) Emin = e; if (e > Emax) Emax = e;
    }
    x.push(rx); y.push(ry); z.push(rz); E.push(re);
  }
  return { x, y, z, E, Emin, Emax };
}
export function polar(C, plane, n = 361) {
  const Sm = mandelS(C), angle = [], E = [], G = [];
  for (let k = 0; k < n; k++) {
    const a = 2 * Math.PI * k / (n - 1), c = Math.cos(a), s = Math.sin(a);
    let nv, mv;
    if (plane === 'xy') { nv = [c, s, 0]; mv = [-s, c, 0]; } else if (plane === 'xz') { nv = [c, 0, s]; mv = [-s, 0, c]; } else { nv = [0, c, s]; mv = [0, -s, c]; }
    angle.push(a * 180 / Math.PI); E.push(dirE(Sm, nv)); G.push(dirG(Sm, nv, mv));
  }
  return { angle, E, G };
}
export function virtualTests(C) {
  const S = inv(C), ax = 'xyz', out = [];
  for (let a = 0; a < 3; a++) {
    const v = { E: 1 / S[a][a] };
    for (let b = 0; b < 3; b++) if (b !== a) v[`nu_${ax[a]}${ax[b]}`] = -S[b][a] / S[a][a];
    out.push({ test: `UT_${ax[a]}`, description: `uniaxial tension along ${ax[a]}`, values: v });
  }
  for (const [a, b] of [[0, 1], [1, 2], [0, 2]]) {
    const ea = S[a][a] + S[a][b], eb = S[b][a] + S[b][b];
    out.push({ test: `ET_${ax[a]}${ax[b]}`, description: `equibiaxial tension in the ${ax[a]}-${ax[b]} plane`,
      values: { E_eb: 1 / (0.5 * (ea + eb)), [`eps_${ax[a]}/eps_${ax[b]}`]: ea / eb } });
  }
  for (const [k, pl] of [[5, 'xy'], [3, 'yz'], [4, 'xz']]) out.push({ test: `SS_${pl}`, description: `simple shear in the ${pl[0]}-${pl[1]} plane`, values: { G: 1 / S[k][k] } });
  return out;
}
export function hsUpper(rho, E, nu) {
  const K = E / (3 * (1 - 2 * nu)), G = E / (2 * (1 + nu)), v = 1 - rho;
  const Ku = K + v / (-1 / K + 3 * rho / (3 * K + 4 * G));
  const Gu = G + v / (-1 / G + 6 * rho * (K + 2 * G) / (5 * G * (3 * K + 4 * G)));
  return { K: Ku, G: Gu, E: 9 * Ku * Gu / (3 * Ku + Gu) };
}

// explore() of lattice_app/analysis.py from a C(rho) grid: grid = {C: [upper | null], fe: [...]}, start/step of rho
export function explore(pattern, rho, grid, g0, step, rhoRange, material) {
  // rho is snapped to the grid by the caller; interpolate linearly otherwise (C entries, small step)
  const t = (rho - g0) / step, i = Math.max(0, Math.min(grid.C.length - 2, Math.floor(t + 1e-9))), f = Math.min(1, Math.max(0, t - i));
  const a = grid.C[i], b = f > 1e-9 ? grid.C[i + 1] : grid.C[i];
  if (!a || !b) return { ok: false, message: `${pattern}, rho = ${rho}: the density laws give no positive-definite stiffness here (extrapolation failed)` };
  const sc = material.E / 2100;
  const Cp = fromUpper(a.map((v, k) => ((1 - f) * v + f * b[k]) * sc));
  const fe = grid.fe.find((p) => Math.abs(p.rho - rho) < 1e-6);
  const c = constants(Cp), s = surface(Cp);
  const out = { ok: true, pattern, rho, material, extrapolated: !(rho >= rhoRange[0] - 1e-9 && rho <= rhoRange[1] + 1e-9),
    C_print: Cp, C_fe: toFe(Cp, pattern), constants: c, A_U: anisotropyU(Cp), zener: zener(Cp), surface: s,
    polar: { xy: polar(Cp, 'xy'), xz: polar(Cp, 'xz'), yz: polar(Cp, 'yz') }, tests: virtualTests(Cp), hs: hsUpper(rho, material.E, material.nu) };
  if (fe) out.fe = { n_t: fe.n_t, nodes: fe.nodes, law_vs_fe: fe.law_vs_fe, C_print: fromUpper(fe.C_print.map((v) => v * sc)) };
  return out;
}
export { FE_TO_PRINT };
