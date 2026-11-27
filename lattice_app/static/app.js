// Lattice Explorer front end. Plain ES module; Plotly (global) for charts, three.js for 3D meshes.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import * as SCI from './science.js';

// Read-only web version (scripts/build_static.py): data from ./data, derived quantities computed here.
const STATIC = document.body.dataset.static === '1';

const $ = (id) => document.getElementById(id);
const NAMES = { grid: 'Grid', triangles: 'Triangles', honeycomb: 'Honeycomb', cubic: 'Cubic', gyroid: 'Gyroid' };
const S = { state: null, pattern: 'honeycomb', rho: 0.3, material: 'PLA', source: 'abaqus', tab: 'explore',
  polarq: 'E', cframe: 'print', cunit: 'MPa', trscale: 'log', last: null, trends: null, fieldsel: null };

// ------------------------------------------------------------------ helpers
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const color = (p) => css('--s' + (S.state.patterns.indexOf(p) + 1));
function toast(msg, ms = 3500) { const t = $('toast'); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => (t.hidden = true), ms); }
const _json = {};
async function getJSON(url) {
  if (!_json[url]) _json[url] = fetch(url).then((r) => { if (!r.ok) throw new Error(`not available in the web version (${url})`); return r.json(); });
  return _json[url];
}
async function staticApi(path) {
  const u = new URL(path, 'http://x'), p = Object.fromEntries(u.searchParams), route = u.pathname;
  const src = p.source || 'abaqus';
  if (route === '/api/state') return getJSON('data/state.json');
  if (route === '/api/explore') {
    const st = await getJSON('data/state.json'), g = await getJSON(`data/${src}/cgrid.json`), grid = g.patterns[p.pattern];
    if (!grid) return { ok: false, message: `No ${src} data for ${p.pattern}.` };
    return SCI.explore(p.pattern, +p.rho, grid, st.rho_grid.start, st.rho_grid.step, st.rho_range, st.materials[p.material || 'PLA']);
  }
  if (route === '/api/trends') return getJSON(`data/${src}/trends.json`);
  if (route === '/api/laws') return getJSON(`data/${src}/laws_${p.pattern}.json`).catch(() => ({ ok: false }));
  if (route === '/api/convergence/sets') return getJSON(`data/${src}/conv_sets.json`);
  if (route === '/api/convergence') return getJSON(`data/${src}/conv_${p.pattern}_${+p.rho}.json`);
  if (route === '/api/mesh') return getJSON(`data/mesh/${p.pattern}_${+p.rho}_${+p.n_t}.json`);
  if (route === '/api/jobs') return [];
  throw new Error('not available in the read-only web version');
}
async function api(path, opts = {}) {
  if (STATIC) return staticApi(path);
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) { /* not json */ } throw new Error(m); }
  return r.json();
}
function q(o) { return new URLSearchParams(o).toString(); }
function b64(s, T) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new T(u.buffer); }
function fmt(v, d = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '-';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return Number(v.toPrecision(d)).toString();
}
function debounce(f, ms) { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => f(...a), ms); }; }
function seg(id, key, cb) {
  $(id).addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    [...$(id).children].forEach((x) => x.classList.toggle('on', x === b)); S[key] = b.dataset.v; cb();
  });
}
function layout(extra = {}) {
  const ink = css('--text'), mute = css('--muted'), grid = css('--grid');
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { family: css('--sans'), size: 12, color: ink },
    margin: { l: 56, r: 14, t: 26, b: 44 }, hovermode: 'closest',
    xaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: mute, tickcolor: mute },
    yaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: mute, tickcolor: mute },
    legend: { orientation: 'h', y: -0.18, font: { size: 11 } },
  }, extra);
}
const CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };
const plot = (el, data, lay) => Plotly.react(el, data, lay, CFG);

// ------------------------------------------------------------------ init
async function init() {
  S.state = await api('/api/state');
  const st = S.state;
  $('ver').textContent = 'v' + st.version;
  S.source = st.settings.source || 'abaqus';
  S.material = st.materials[st.settings.material] ? st.settings.material : 'PLA';
  const pb = $('patterns');
  st.patterns.forEach((p) => {
    const b = document.createElement('button');
    b.innerHTML = `<span class="sw" style="background:${color(p)}"></span>${NAMES[p] || p}`;
    b.dataset.p = p; b.onclick = () => setPattern(p); pb.appendChild(b);
  });
  const ticks = $('rhoticks');
  st.rhos.forEach((r) => { const s = document.createElement('span'); s.style.left = ((r - 0.05) / 0.9 * 100) + '%'; s.textContent = r; ticks.appendChild(s); });
  fillMaterials(); fillSources(); renderSolver(st.solver, st.cgx);
  $('jpattern').innerHTML = st.patterns.filter((p) => !st.no_solve.includes(p)).map((p) => `<option value="${p}">${NAMES[p]}</option>`).join('');
  $('rho').oninput = () => setRho(+$('rho').value, true);
  $('rhonum').onchange = () => setRho(+$('rhonum').value, false);
  $('material').onchange = () => { S.material = $('material').value; refresh(); };
  $('source').onchange = () => { S.source = $('source').value; refresh(); };
  document.querySelectorAll('#tabs button').forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
  seg('polarq', 'polarq', () => drawPolar()); seg('cframe', 'cframe', () => drawMatrix()); seg('cunit', 'cunit', () => drawMatrix());
  seg('trscale', 'trscale', () => drawTrends());
  $('polarall').onchange = () => drawPolar();
  initMesh(); initFields(); initJobs(); initSettings(); initData();
  if (STATIC) await initStatic();
  setPattern(S.pattern);
  if (!STATIC) { setInterval(pollJobs, 2500); pollJobs(); }
}
function fillMaterials() {
  const m = S.state.materials;
  const opts = Object.values(m).map((x) => `<option value="${x.name}">${x.name} (E = ${x.E} MPa, &nu; = ${x.nu})</option>`).join('');
  $('material').innerHTML = opts; $('jmat').innerHTML = opts; $('material').value = S.material;
  const cur = m[S.material]; $('matnote').textContent = cur ? cur.source : '';
}
function fillSources() {
  const o = S.state.sources.map((s) => `<option value="${s.id}">${s.label} (${s.count})</option>`).join('');
  $('source').innerHTML = o; $('datasource').innerHTML = '<option value="">all sources</option>' + o; $('source').value = S.source;
}
function renderSolver(sol, cgx) {
  if (STATIC) { $('solverstatus').innerHTML = '<span class="muted">read-only web version · FE results bundled</span>'; return; }
  const ok = sol && sol.ok;
  $('solverstatus').innerHTML = `<span class="dot" style="background:${ok ? css('--good') : css('--bad')}"></span>` +
    (ok ? `CalculiX ${sol.version}` : 'CalculiX not found');
  $('solverstatus').title = ok ? sol.path : sol.message;
}
function setPattern(p) {
  S.pattern = p;
  document.querySelectorAll('#patterns button').forEach((b) => b.classList.toggle('on', b.dataset.p === p));
  if (!S.state.no_solve.includes(p)) $('jpattern').value = p;
  $('convrun').disabled = S.state.no_solve.includes(p);
  $('convrun').title = $('convrun').disabled ? `${NAMES[p]}: solving switched off; bundled results only` : '';
  $('convlevels').value = p === 'gyroid' ? '1.5, 2, 2.5, 3' : '2, 3, 4, 6';
  $('meshnt').value = S.state.production_nt[p] || 3;
  refresh();
}
const refreshDeb = debounce(() => refresh(), 90);
function setRho(r, fromSlider) {
  r = Math.min(0.95, Math.max(0.05, r)); S.rho = Math.round(r * 1000) / 1000;
  if (STATIC) S.rho = Math.round(Math.round(r / 0.005) * 0.005 * 1000) / 1000;   // web version: C(rho) stored on this grid
  if (!fromSlider) $('rho').value = S.rho; $('rhonum').value = S.rho.toFixed(3);
  $('jrho').value = S.rho;
  refreshDeb();
}
function showTab(t) {
  S.tab = t;
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === t));
  document.querySelectorAll('.tab').forEach((s) => (s.hidden = s.id !== 'tab-' + t));
  refresh();
  setTimeout(() => document.querySelectorAll(`#tab-${t} .js-plotly-plot`).forEach((el) => Plotly.Plots.resize(el)), 30);
  viewers.forEach((v) => v.resize());
}
async function refresh() {
  try {
    if (S.tab === 'explore') await loadExplore();
    else if (S.tab === 'trends') await loadTrends();
    else if (S.tab === 'laws') await loadLaws();
    else if (S.tab === 'mesh') await loadConvSets();
    else if (S.tab === 'fields') await loadFieldRecords();
    else if (S.tab === 'data') await loadData();
    else if (S.tab === 'settings') loadSettings();
  } catch (e) { toast(e.message); }
}

// ------------------------------------------------------------------ explorer
let exploreSeq = 0;
async function loadExplore() {
  const seq = ++exploreSeq;
  const d = await api('/api/explore?' + q({ pattern: S.pattern, rho: S.rho, source: S.source, material: S.material }));
  if (seq !== exploreSeq) return;
  S.last = d;
  const st = $('exstatus');
  if (!d.ok) { st.className = 'status warn'; st.textContent = d.message; return; }
  let msg = `<b>${NAMES[S.pattern]}</b>, &rho; = ${S.rho.toFixed(3)}, ${d.material.name}: `;
  if (d.fe) msg += `FE data point (n<sub>t</sub> = ${d.fe.n_t}${d.fe.nodes ? ', ' + d.fe.nodes.toLocaleString() + ' nodes' : ''}); density law differs from the FE matrix by ${(100 * d.fe.law_vs_fe).toFixed(2)} % of max |C|.`;
  else if (d.extrapolated) msg += `<b>outside the FE range ${S.state.rho_range[0]}-${S.state.rho_range[1]}</b>: the density law extrapolates; treat as an estimate.`;
  else msg += 'interpolated with the density laws between FE points.';
  st.className = 'status' + (d.extrapolated ? ' warn' : ''); st.innerHTML = msg;
  $('rhonote').textContent = d.extrapolated ? 'extrapolated' : '';
  drawSurface(); drawPolar(); drawMatrix(); drawConstants(); drawCellRho();
}
function drawSurface() {
  const d = S.last, s = d.surface;
  const sc = [[0, '#cce3fd'], [0.5, '#338ced'], [1, '#0e3d7e']];
  const r = s.Emax * 1.08;
  plot('surf', [{ type: 'surface', x: s.x, y: s.y, z: s.z, surfacecolor: s.E, colorscale: sc, cmin: s.Emin, cmax: s.Emax,
    colorbar: { title: { text: 'E (MPa)', side: 'right' }, thickness: 12, len: 0.7 },
    hovertemplate: 'E = %{surfacecolor:.4g} MPa<extra></extra>', lighting: { ambient: 0.75, diffuse: 0.5, specular: 0.1 } }],
  layout({ margin: { l: 0, r: 0, t: 0, b: 0 }, scene: {
    aspectmode: 'cube', camera: { eye: { x: 1.5, y: -1.5, z: 1.0 } },
    xaxis: { title: 'x', range: [-r, r], gridcolor: css('--grid') }, yaxis: { title: 'y', range: [-r, r], gridcolor: css('--grid') },
    zaxis: { title: 'z (build)', range: [-r, r], gridcolor: css('--grid') } } }));
  $('surfinfo').textContent = `E_max / E_min = ${fmt(s.Emax / s.Emin, 3)}`;
}
async function drawPolar() {
  const d = S.last; if (!d || !d.ok) return;
  const k = S.polarq;
  const planes = [['xy', 'print plane x-y'], ['xz', 'vertical x-z'], ['yz', 'vertical y-z']];
  let sets = [[S.pattern, d]];
  if ($('polarall').checked) {
    const others = S.state.patterns.filter((p) => p !== S.pattern);
    const res = await Promise.all(others.map((p) => api('/api/explore?' + q({ pattern: p, rho: S.rho, source: S.source, material: S.material })).catch(() => null)));
    others.forEach((p, i) => { if (res[i] && res[i].ok) sets.push([p, res[i]]); });
    sets.sort((a, b) => S.state.patterns.indexOf(a[0]) - S.state.patterns.indexOf(b[0]));
  }
  const data = [];
  planes.forEach(([pl, title], i) => {
    sets.forEach(([p, dd]) => {
      data.push({ type: 'scatterpolar', mode: 'lines', r: dd.polar[pl][k], theta: dd.polar[pl].angle, subplot: i ? 'polar' + (i + 1) : 'polar',
        name: NAMES[p], legendgroup: p, showlegend: i === 0 && sets.length > 1, line: { color: color(p), width: 2 },
        hovertemplate: `${NAMES[p]}<br>θ = %{theta:.0f}°<br>${k} = %{r:.4g} MPa<extra>${title}</extra>` });
    });
  });
  const pol = (i, title) => ({ domain: { x: [i / 3 + 0.02, (i + 1) / 3 - 0.02], y: [0.08, 0.92] },
    radialaxis: { gridcolor: css('--grid'), tickfont: { size: 9 }, angle: 90, tickangle: 90 },
    angularaxis: { gridcolor: css('--grid'), tickfont: { size: 9 }, dtick: 45, direction: 'counterclockwise', rotation: 0 },
    bgcolor: 'rgba(0,0,0,0)' });
  const lay = layout({ margin: { l: 20, r: 20, t: 30, b: 30 }, polar: pol(0), polar2: pol(1), polar3: pol(2), showlegend: sets.length > 1,
    annotations: planes.map(([pl, t], i) => ({ text: `${t}<br><span style="font-size:10px">θ from ${pl[0]} toward ${pl[1]}</span>`, x: (i + 0.5) / 3, y: 1.04, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 12 } })) });
  lay.legend = { orientation: 'h', y: -0.02, x: 0.5, xanchor: 'center' };
  plot('polar', data, lay);
}
function drawMatrix() {
  const d = S.last; if (!d || !d.ok) return;
  const C = S.cframe === 'print' ? d.C_print : d.C_fe;
  const Es = d.material.E;
  const lab = ['11', '22', '33', '23', '13', '12'];
  const mx = Math.max(...C.flat().map(Math.abs));
  let h = '<tr><th></th>' + lab.map((l) => `<th>${l}</th>`).join('') + '</tr>';
  C.forEach((row, i) => {
    h += `<tr><th>${lab[i]}</th>` + row.map((v, j) => {
      const z = Math.abs(v) < 1e-9 * mx;
      const val = S.cunit === 'norm' ? v / Es : v;
      return `<td class="${z ? 'z' : ''} ${i === j ? 'dg' : ''}">${z ? '0' : fmt(val, S.cunit === 'norm' ? 4 : 5)}</td>`;
    }).join('') + '</tr>';
  });
  $('cmat').innerHTML = h;
  $('cnote').innerHTML = `Voigt order 11, 22, 33, 23, 13, 12 with engineering shear strains. ${S.cframe === 'fe' ? 'FE frame: for prismatic cells x is the extrusion axis (= build direction z).' : ''} ` +
    `Solid: E<sub>s</sub> = ${Es} MPa, &nu;<sub>s</sub> = ${d.material.nu}.`;
}
function drawConstants() {
  const d = S.last, c = d.constants, Es = d.material.E;
  const rows = [['', 'Young\'s moduli (MPa)'], ['E_x', c.E1], ['E_y', c.E2], ['E_z (build)', c.E3],
    ['', 'Shear moduli (MPa)'], ['G_yz', c.G23], ['G_xz', c.G13], ['G_xy', c.G12],
    ['', 'Poisson ratios'], ['ν_xy', c.nu12], ['ν_xz', c.nu13], ['ν_yz', c.nu23], ['ν_zx', c.nu31],
    ['', 'Anisotropy'], ['A^U (0 = isotropic)', d.A_U], ['E_max/E_min', d.surface.Emax / d.surface.Emin]];
  if (['cubic', 'gyroid'].includes(S.pattern)) rows.push(['Zener 2C44/(C11-C12)', d.zener]);
  rows.push(['', 'Relative to solid'], ['E_z / E_s', c.E3 / Es], ['E_x / E_s', c.E1 / Es], ['HS upper E / E_s', d.hs.E / Es]);
  $('consts').innerHTML = rows.map(([k, v]) => (k === '' ? `<tr class="sec"><td colspan="2">${v}</td></tr>` : `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`)).join('');
  $('vtests').innerHTML = d.tests.map((t) => `<tr class="sec"><td colspan="2" title="${t.description}">${t.test}</td></tr>` +
    Object.entries(t.values).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v)}${/^(E|G)/.test(k) ? '' : ''}</td></tr>`).join('')).join('');
}
async function drawCellRho() {
  const tr = await getTrends(); if (!tr) return;
  const p = tr.patterns[S.pattern]; if (!p) return;
  const groups = [[['E1', 'E_x'], ['E2', 'E_y'], ['E3', 'E_z']], [['G23', 'G_yz'], ['G13', 'G_xz'], ['G12', 'G_xy']], [['nu12', 'ν_xy'], ['nu13', 'ν_xz'], ['nu31', 'ν_zx']]];
  const dash = ['solid', 'dash', 'dot'];
  const data = [];
  groups.forEach((g, gi) => g.forEach(([k, nm], j) => {
    const ax = gi ? String(gi + 1) : '';
    data.push({ x: tr.rho, y: p.curves[k], mode: 'lines', name: nm, xaxis: 'x' + ax, yaxis: 'y' + ax, line: { color: color(S.pattern), dash: dash[j], width: 2 },
      hovertemplate: `${nm} = %{y:.4g}<br>ρ = %{x:.3f}<extra></extra>` });
    data.push({ x: p.points.rho, y: p.points[k], mode: 'markers', showlegend: false, xaxis: 'x' + ax, yaxis: 'y' + ax,
      marker: { color: color(S.pattern), size: 8, symbol: ['circle', 'square', 'diamond'][j], line: { color: css('--surface'), width: 1.5 } }, hovertemplate: `FE ${nm} = %{y:.4g}<extra></extra>` });
  }));
  const ax = (i, t) => ({ domain: [i / 3 + 0.04, (i + 1) / 3 - 0.03], title: { text: 'ρ' }, gridcolor: css('--grid'), anchor: 'y' + (i ? i + 1 : '') });
  const ay = (i, t) => ({ title: { text: t }, gridcolor: css('--grid'), anchor: 'x' + (i ? i + 1 : '') });
  const lay = layout({ xaxis: ax(0), xaxis2: ax(1), xaxis3: ax(2), yaxis: ay(0, 'E (MPa)'), yaxis2: ay(1, 'G (MPa)'), yaxis3: ay(2, 'ν'),
    shapes: [0, 1, 2].map((i) => ({ type: 'line', x0: S.rho, x1: S.rho, yref: `y${i ? i + 1 : ''} domain`, y0: 0, y1: 1, xref: 'x' + (i ? i + 1 : ''), line: { color: css('--muted'), width: 1, dash: 'dot' } })),
    margin: { l: 56, r: 10, t: 10, b: 40 } });
  lay.legend = { orientation: 'h', y: -0.25 };
  plot('cellrho', data, lay);
}

// ------------------------------------------------------------------ trends
async function getTrends() {
  const k = S.source + '|' + S.material + '|' + (S.state.version_data || '');
  if (S.trends && S.trends.k === k) return S.trends.d;
  const d = await api('/api/trends?' + q({ source: S.source, material: S.material }));
  S.trends = { k, d }; return d;
}
async function loadTrends() { await getTrends(); drawTrends(); }
function drawTrends() {
  const tr = S.trends && S.trends.d; if (!tr) return;
  const Es = tr.E_s;
  const charts = [['E1', 'E_x / E_s', true], ['E2', 'E_y / E_s', true], ['E3', 'E_z / E_s (build)', true],
    ['G12', 'G_xy / E_s', true], ['G23', 'G_yz / E_s', true], ['nu12', 'ν_xy', false], ['nu31', 'ν_zx', false],
    ['A_U', 'Universal anisotropy A^U', false], ['Emax_Emin', 'E_max / E_min', false]];
  const grid = $('trendgrid');
  $('trlegend').innerHTML = Object.keys(tr.patterns).map((p) => `<span><span class="sw" style="background:${color(p)}"></span>${NAMES[p]}</span>`).join('') +
    '<span><span class="dot"></span>Voigt ρE<sub>s</sub></span><span><span class="dash"></span>HS upper (isotropic)</span>';
  if (!grid.children.length) charts.forEach(([k]) => { const c = document.createElement('div'); c.className = 'card span4'; c.innerHTML = `<div id="tr_${k}" class="plot mid"></div>`; grid.appendChild(c); });
  charts.forEach(([k, title, norm]) => {
    const data = [];
    Object.entries(tr.patterns).forEach(([p, d]) => {
      const sc = norm ? 1 / Es : 1;
      data.push({ x: tr.rho, y: d.curves[k].map((v) => v * sc), mode: 'lines', name: NAMES[p], legendgroup: p, line: { color: color(p), width: 2 },
        hovertemplate: `${NAMES[p]}: %{y:.4g}<br>ρ = %{x:.3f}<extra></extra>` });
      if (d.points[k]) data.push({ x: d.points.rho, y: d.points[k].map((v) => v * sc), mode: 'markers', legendgroup: p, showlegend: false,
        marker: { color: color(p), size: 8, line: { color: css('--surface'), width: 1.5 } }, hovertemplate: `${NAMES[p]} FE: %{y:.4g}<extra></extra>` });
    });
    if (k.startsWith('E') && k !== 'Emax_Emin') {
      data.push({ x: tr.rho, y: tr.rho, mode: 'lines', name: 'Voigt ρE_s', line: { color: css('--muted'), width: 1.2, dash: 'dot' }, hovertemplate: 'Voigt: %{y:.4g}<extra></extra>' });
      data.push({ x: tr.rho, y: tr.hs.E.map((v) => v / Es), mode: 'lines', name: 'HS upper (isotropic)', line: { color: css('--muted'), width: 1.2, dash: 'dash' }, hovertemplate: 'HS: %{y:.4g}<extra></extra>' });
    }
    const logy = S.trscale === 'log' && !k.startsWith('nu');
    const lay = layout({ title: { text: title, x: 0.02, font: { size: 13 } }, margin: { l: 52, r: 10, t: 32, b: 40 }, showlegend: false,
      xaxis: { title: { text: 'ρ' }, type: S.trscale === 'log' ? 'log' : 'linear', gridcolor: css('--grid'), tickvals: [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1] },
      yaxis: { type: logy ? 'log' : 'linear', gridcolor: css('--grid') },
      shapes: [{ type: 'rect', xref: 'x', yref: 'paper', x0: S.state.rho_range[0], x1: S.state.rho_range[1], y0: 0, y1: 1, fillcolor: css('--grid'), opacity: 0.35, line: { width: 0 }, layer: 'below' }] });
    plot('tr_' + k, data, lay);
  });
}

// ------------------------------------------------------------------ laws
const CONST_LABEL = { E: 'E / E_s', G: 'G / G_s', E_x: 'E_x = E_y', E_z: 'E_z (build)', G_xz: 'G_xz = G_yz', G_xy: 'G_xy', nu: 'ν', nu_xy: 'ν_xy', nu_zx: 'ν_zx' };
const SYM_TEXT = { cubic: 'cubic: E, G, ν', tetragonal: 'tetragonal about z: E_x = E_y, E_z, G_xy, G_xz = G_yz, ν_xy, ν_zx', hexagonal: 'transversely isotropic about z: E_x = E_y, E_z, G_xz = G_yz, ν_xy, ν_zx, with G_xy = E_x / 2(1 + ν_xy)' };
function lawRows(entries, withHead) {
  let h = withHead ? '<tr><th>quantity</th><th>law</th><th>d</th><th>n</th><th>coefficients</th><th>LOO rms</th><th>LOO max</th><th>other family LOO rms</th><th>power law a, n (LOO rms)</th></tr>' : '';
  Object.entries(entries).forEach(([n, e]) => {
    h += `<tr><td>${CONST_LABEL[n] || n}</td><td>${e.chosen.replace('_', '-')}</td><td class="n">${e.params.d}</td><td class="n">${e.params.n ? fmt(e.params.n, 3) : '-'}</td>` +
      `<td class="n" style="font-size:11px;white-space:normal">${e.params.coeffs.map((c) => fmt(c, 4)).join(', ')}</td><td class="n">${(100 * e.loo_rms).toFixed(2)} %</td>` +
      `<td class="n">${(100 * e.loo_max).toFixed(2)} %</td><td class="n">${(100 * e.other.loo_rms).toFixed(2)} %</td>` +
      `<td class="n">${fmt(e.power.a, 3)}, ${fmt(e.power.n, 3)} (${(100 * e.power.loo_rms).toFixed(1)} %)</td></tr>`;
  });
  return h;
}
async function loadLaws() {
  const d = await api('/api/laws?' + q({ pattern: S.pattern, source: S.source, material: S.material }));
  $('lawtitle').textContent = `Modulus laws: ${NAMES[S.pattern]} (print frame)`;
  if (!d.ok) { $('lawtable').innerHTML = '<tr><td>no data for this source and material</td></tr>'; return; }
  const cols = ['--s1', '--s2', '--s3', '--s4', '--s5'];
  const data = [];
  Object.entries(d.moduli).forEach(([n, e], i) => {
    const c = css(cols[i % 5]); const nm = (CONST_LABEL[n] || n) + (n.startsWith('G') ? ' / G_s' : n.startsWith('E') && n !== 'E' ? ' / E_s' : '');
    data.push({ x: d.curve_rho, y: e.curve, mode: 'lines', name: nm, legendgroup: n, line: { color: c, width: 2 }, hovertemplate: `${nm}: %{y:.4g}<br>ρ = %{x:.3f}<extra></extra>` });
    data.push({ x: d.curve_rho, y: e.power_curve, mode: 'lines', legendgroup: n, showlegend: false, line: { color: c, width: 1, dash: 'dash' }, hovertemplate: `power law: %{y:.4g}<extra></extra>` });
    data.push({ x: d.rho, y: e.y, mode: 'markers', legendgroup: n, showlegend: false, marker: { color: c, size: 8, line: { color: css('--surface'), width: 1.5 } }, hovertemplate: `FE: %{y:.4g}<extra></extra>` });
  });
  data.push({ x: [0.02, 1], y: [0.02, 1], mode: 'lines', name: 'h = ρ', line: { color: css('--muted'), width: 1, dash: 'dot' } });
  const vline = [{ type: 'line', x0: S.rho, x1: S.rho, yref: 'paper', y0: 0, y1: 1, line: { color: css('--muted'), width: 1, dash: 'dot' } }];
  plot('lawplot', data, layout({ xaxis: { type: 'log', title: { text: 'ρ' }, gridcolor: css('--grid'), tickvals: [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1] },
    yaxis: { type: 'log', title: { text: 'h(ρ)' }, gridcolor: css('--grid') }, shapes: vline }));
  const nd = [];
  Object.entries(d.poisson).forEach(([n, e], i) => {
    const c = css(cols[i % 5]);
    nd.push({ x: e.rho, y: e.curve, mode: 'lines', name: CONST_LABEL[n] || n, legendgroup: n, line: { color: c, width: 2 }, hovertemplate: `${n}: %{y:.4g}<br>ρ = %{x:.3f}<extra></extra>` });
    nd.push({ x: d.rho, y: e.y, mode: 'markers', legendgroup: n, showlegend: false, marker: { color: c, size: 8, line: { color: css('--surface'), width: 1.5 } }, hovertemplate: `FE: %{y:.4g}<extra></extra>` });
  });
  plot('nuplot', nd, layout({ xaxis: { title: { text: 'ρ' }, gridcolor: css('--grid'), range: [0, 1.02] }, yaxis: { title: { text: 'ν' }, gridcolor: css('--grid') }, shapes: vline }));
  $('lawsym').textContent = SYM_TEXT[d.symmetry];
  $('lawtable').innerHTML = lawRows(d.moduli, true) + `<tr><td colspan="9" class="muted">FE points at n_t = ${[...new Set(d.n_t)].join(', ')}, ρ = ${d.rho.join(', ')}. LOO: each point predicted by a fit to the others; relative errors.</td></tr>`;
  $('entrytable').innerHTML = lawRows(d.entries, true) + '<tr><td colspan="9" class="muted">Entries normalized by the solid C_ij; FE frame (prismatic cells: 1 = extrusion = build direction).</td></tr>';
}

// ------------------------------------------------------------------ 3D viewer (three.js)
const viewers = [];
class Viewer {
  constructor(el) {
    this.el = el; el.innerHTML = '';
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 5000); this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.addEventListener('change', () => this.render());
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 1.7));
    this.light = new THREE.DirectionalLight(0xffffff, 1.6); this.camera.add(this.light); this.light.position.set(1, 2, 3);
    this.scene.add(this.camera);
    this.group = new THREE.Group(); this.scene.add(this.group);
    this.hint = document.createElement('div'); this.hint.className = 'viewhint'; this.hint.textContent = 'drag: rotate · right-drag: pan · wheel: zoom · z = build direction'; el.appendChild(this.hint);
    new ResizeObserver(() => this.resize()).observe(el); viewers.push(this);
  }
  resize() { const w = this.el.clientWidth, h = this.el.clientHeight; if (!w || !h) return; this.renderer.setSize(w, h); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this.render(); }
  clear() { this.group.children.slice().forEach((o) => { this.group.remove(o); o.geometry && o.geometry.dispose(); }); }
  add(o) { this.group.add(o); }
  fit(keep) {
    if (!keep) this.fitted = false;
    const box = new THREE.Box3().setFromObject(this.group); const c = box.getCenter(new THREE.Vector3()); const r = box.getSize(new THREE.Vector3()).length() / 2;
    if (!this.fitted) {
      this.controls.target.copy(c);
      this.camera.position.copy(c).add(new THREE.Vector3(1.6, -2.2, 1.3).normalize().multiplyScalar(r / Math.tan(Math.PI * 15 / 180) * 1.05));
      this.camera.near = r / 50; this.camera.far = r * 50; this.camera.updateProjectionMatrix(); this.fitted = true;
    }
    this.controls.update(); this.render();
  }
  render() { this.renderer.setClearColor(new THREE.Color(css('--surface-2') || '#f8f8f6')); this.renderer.render(this.scene, this.camera); }
}

// colour maps: sequential (viridis-like) or banded rainbow as in Abaqus/CalculiX
const VIRIDIS = [[0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530], [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551], [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318], [0.741, 0.873, 0.150], [0.993, 0.906, 0.144]];
function cmap(t, kind) {
  t = Math.min(1, Math.max(0, t));
  if (kind === 'rainbow') {
    const b = Math.min(11, Math.floor(t * 12)) / 11;                       // 12 bands, blue -> red
    const h = (1 - b) * 240 / 360; const c = new THREE.Color().setHSL(h, 0.95, 0.5); return [c.r, c.g, c.b];
  }
  const x = t * (VIRIDIS.length - 1), i = Math.min(VIRIDIS.length - 2, Math.floor(x)), f = x - i;
  return VIRIDIS[i].map((v, k) => v + f * (VIRIDIS[i + 1][k] - v));
}
function cssGradient(kind) {
  if (kind === 'rainbow') { const s = []; for (let i = 0; i < 12; i++) { const c = cmap((i + 0.5) / 12, kind).map((v) => Math.round(v * 255)).join(','); s.push(`rgb(${c}) ${(i / 12 * 100).toFixed(2)}%`, `rgb(${c}) ${((i + 1) / 12 * 100).toFixed(2)}%`); } return `linear-gradient(90deg, ${s.join(',')})`; } const s = []; for (let i = 0; i <= 24; i++) { const c = cmap(i / 24, kind); s.push(`rgb(${c.map((v) => Math.round(v * 255)).join(',')}) ${(i / 24 * 100).toFixed(1)}%`); } return `linear-gradient(90deg, ${s.join(',')})`; }

// ------------------------------------------------------------------ mesh tab
let meshViewer = null;
function initMesh() {
  $('meshbtn').onclick = buildMesh;
  $('convset').onchange = () => loadConvergence();
  $('convrun').onclick = async () => {
    const [p, r] = ($('convset').value || `${S.pattern}|${S.rho}`).split('|');
    const lv = $('convlevels').value.split(/[ ,;]+/).filter(Boolean).map(Number);
    try { await api('/api/jobs', { method: 'POST', body: JSON.stringify({ pattern: S.pattern, rho: S.rho, n_t: lv, material: S.material }) }); toast(`queued ${lv.length} jobs for ${NAMES[S.pattern]} ρ = ${S.rho}; see Jobs`); pollJobs(); } catch (e) { toast(e.message); }
  };
}
let meshIndex = null;
function meshRho() {                       // static: previews exist only at the FE densities
  if (!STATIC) return S.rho;
  const rs = Object.keys(meshIndex[S.pattern] || {}).map(Number);
  return rs.reduce((a, b) => (Math.abs(b - S.rho) < Math.abs(a - S.rho) ? b : a), rs[0]);
}
function fillMeshLevels() {
  if (!STATIC) return;
  const r = meshRho(), lv = (meshIndex[S.pattern] || {})[String(r)] || [];
  const sel = $('meshnt'), prev = +sel.value;
  sel.innerHTML = lv.map((n) => `<option value="${n}">${n}</option>`).join('');
  const prod = S.state.production_nt[S.pattern];
  sel.value = lv.includes(prev) ? prev : (lv.includes(prod) ? prod : lv[0]);
  $('meshrhonote').textContent = Math.abs(r - S.rho) > 1e-9 ? `previews exist at the FE densities: showing ρ = ${r}` : '';
}
async function initStatic() {
  ['fields', 'jobs', 'data', 'settings'].forEach((t) => (document.querySelector(`#tabs button[data-tab="${t}"]`).hidden = true));
  document.querySelectorAll('.server-only').forEach((e) => (e.hidden = true));
  const inp = $('meshnt'), sel = document.createElement('select');
  sel.id = 'meshnt'; inp.replaceWith(sel);
  const note = document.createElement('span'); note.id = 'meshrhonote'; note.className = 'muted'; sel.parentElement.after(note);
  sel.onchange = () => buildMesh();
  meshIndex = await getJSON('data/mesh_index.json');
}
async function buildMesh() {
  if (!meshViewer) meshViewer = new Viewer($('meshview'));
  fillMeshLevels();
  const nt = +$('meshnt').value, mrho = meshRho();
  $('meshbtn').disabled = true; $('meshbtn').textContent = 'building...';
  try {
    const d = await api('/api/mesh?' + q({ pattern: S.pattern, rho: mrho, n_t: nt }));
    const X = b64(d.X, Float32Array), T = b64(d.T, Uint32Array), E = b64(d.Ed, Uint32Array);
    meshViewer.clear();
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(X, 3)); g.setIndex(new THREE.BufferAttribute(T, 1)); g.computeVertexNormals();
    const solid = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0xc9ccd2, roughness: 0.8, metalness: 0, flatShading: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
    const eg = new THREE.BufferGeometry(); eg.setAttribute('position', new THREE.BufferAttribute(X, 3)); eg.setIndex(new THREE.BufferAttribute(E, 1));
    const lines = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x1d2733, transparent: true, opacity: 0.55 }));
    meshViewer.add(solid); meshViewer.add(lines); meshViewer.fit(meshViewer.pat === S.pattern); meshViewer.pat = S.pattern;
    const qq = d.quality;
    const rows = [['', 'Mesh'], ['element type', d.etype + (d.etype === 'C3D10' ? ' (quadratic tet)' : ' (quadratic hex, reduced integration)')], ['nodes / DOF', `${d.nodes.toLocaleString()} / ${d.dof_estimate.toLocaleString()}`], ['elements', d.elements.toLocaleString()],
      ['t_min (mm)', fmt(d.t_min)], ['h = t_min / n_t (mm)', fmt(d.h)], ['element size min / median / max (mm, ∛V)', `${fmt(d.size.min, 3)} / ${fmt(d.size.median, 3)} / ${fmt(d.size.max, 3)}`],
      ['volume fraction (target ρ)', `${fmt(d.volume_fraction, 5)} (${S.rho})`], ['build time (s)', d.build_s],
      ['', `Quality (${qq.metric}; 1 = ideal)`], ['min', fmt(qq.min, 3)], ['1st percentile', fmt(qq.p1, 3)], ['median', fmt(qq.median, 3)],
      [qq.metric === 'radius ratio' ? 'elements below 0.1' : 'elements below 0.2', qq.below_0_1 ?? qq.below_0_2]];
    $('meshstats').innerHTML = rows.map(([k, v]) => (k === '' ? `<tr class="sec"><td colspan="2">${v}</td></tr>` : `<tr><td>${k}</td><td>${v}</td></tr>`)).join('');
  } catch (e) { toast(e.message); }
  $('meshbtn').disabled = false; $('meshbtn').textContent = 'Build mesh';
}
async function loadConvSets() {
  const sets = await api('/api/convergence/sets?' + q({ source: S.source, material: S.material }));
  const sel = $('convset'), prev = sel.value;
  sel.innerHTML = sets.map((s) => `<option value="${s.pattern}|${s.rho}">${NAMES[s.pattern]} ρ = ${s.rho} (n_t ${s.levels.join(', ')})</option>`).join('') || '<option value="">no study for this source</option>';
  const pref = sets.find((s) => s.pattern === S.pattern && Math.abs(s.rho - S.rho) < 1e-6) || sets.find((s) => s.pattern === S.pattern);
  if (pref) sel.value = `${pref.pattern}|${pref.rho}`; else if (prev) sel.value = prev;
  if (!meshViewer || STATIC || meshViewer.pat !== S.pattern) buildMesh();
  loadConvergence();
}
async function loadConvergence() {
  const v = $('convset').value; if (!v) { Plotly.purge('convplot'); Plotly.purge('converr'); $('convtable').innerHTML = ''; return; }
  const [p, r] = v.split('|');
  const d = await api('/api/convergence?' + q({ pattern: p, rho: r, source: S.source, material: S.material }));
  if (!d.ok) return;
  const cols = ['--s1', '--s2', '--s3', '--s4', '--s5'];
  const a = [], b = [];
  Object.entries(d.entries).forEach(([n, e], i) => {
    const c = css(cols[i % 5]);
    const f0 = e.f_ref;
    a.push({ x: d.h, y: e.values.map((v) => v / f0), mode: 'lines+markers', name: n, legendgroup: n, line: { color: c, width: 2 }, marker: { size: 8 },
      text: d.n_t.map((t, k) => `n_t = ${t}${d.nodes[k] ? ', ' + d.nodes[k].toLocaleString() + ' nodes' : ''}`), hovertemplate: `${n}: %{y:.5f} × f_ext<br>h = %{x:.3f} mm<br>%{text}<extra></extra>` });
    const err = e.values.map((v) => Math.max(Math.abs(v - f0) / Math.abs(f0), 1e-7));
    b.push({ x: d.h, y: err.map((x) => 100 * x), mode: 'lines+markers', name: n, legendgroup: n, showlegend: false, line: { color: c, width: 2 }, marker: { size: 8 }, hovertemplate: `${n}: %{y:.3g} %<br>h = %{x:.3f} mm<extra></extra>` });
  });
  const hx = Math.min(...d.h), hX = Math.max(...d.h);
  const y0 = Math.min(...b.map((t) => t.y[t.x.indexOf(hx)]).filter((v) => v > 0)) || 0.01;
  ;[1, 2].forEach((pp) => b.push({ x: [hx, hX], y: [y0, y0 * Math.pow(hX / hx, pp)], mode: 'lines', name: `slope ${pp}`, line: { color: css('--muted'), width: 1, dash: pp === 1 ? 'dot' : 'dash' }, hoverinfo: 'skip' }));
  const prodH = d.production_nt ? d.t_min / d.production_nt : null;
  const shapes = prodH ? [{ type: 'line', x0: prodH, x1: prodH, yref: 'paper', y0: 0, y1: 1, line: { color: css('--muted'), width: 1, dash: 'dot' } }] : [];
  const convLay = layout({ title: { text: `C_ij / f_ext vs element size (${NAMES[p]}, ρ = ${r}, FE frame)`, x: 0.02, font: { size: 13 } }, xaxis: { title: { text: 'h (mm), finer to the left' }, gridcolor: css('--grid') },
    yaxis: { title: { text: 'C_ij / C_ij,ext' }, gridcolor: css('--grid') }, shapes, margin: { l: 60, r: 10, t: 54, b: 40 },
    annotations: prodH ? [{ x: prodH, y: 1, yref: 'paper', text: 'production', showarrow: false, font: { size: 10, color: css('--muted') }, yanchor: 'bottom' }] : [] });
  convLay.legend = { orientation: 'h', y: 1.0, yanchor: 'bottom', x: 1, xanchor: 'right', font: { size: 11 } };
  plot('convplot', a, convLay);
  const errLay = layout({ title: { text: 'relative error |f - f_ext| / f_ext (dotted: slope 1, dashed: slope 2)', x: 0.02, font: { size: 13 } }, xaxis: { type: 'log', title: { text: 'h (mm)' }, gridcolor: css('--grid') },
    yaxis: { type: 'log', title: { text: 'error (%)' }, gridcolor: css('--grid') }, shapes, margin: { l: 60, r: 10, t: 32, b: 40 }, showlegend: false });
  plot('converr', b, errLay);
  let h = '<tr><th>entry</th><th>finest value (MPa)</th><th>p (3 finest)</th><th>p (next triple)</th><th>LSQ p (all)</th><th>f_ext (MPa)</th><th>e_a21</th><th>GCI fine</th><th></th></tr>';
  Object.entries(d.entries).forEach(([n, e]) => {
    const g = e.gci;
    const f1 = e.values[e.values.length - 1];
    if (!g) { h += `<tr><td>${n}</td><td class="n">${fmt(f1, 5)}</td><td colspan="7" class="muted">needs three levels</td></tr>`; return; }
    const gp = e.gci_prev;
    const notAsym = gp && gp.p && g.p && Math.abs(gp.p - g.p) / g.p > 0.25;
    const warn = g.oscillatory ? 'oscillatory' : (notAsym ? 'p not settled' : (gp ? 'asymptotic' : ''));
    h += `<tr><td>${n}</td><td class="n">${fmt(f1, 5)}</td><td class="n">${fmt(g.p, 3)}</td><td class="n">${gp && gp.p ? fmt(gp.p, 3) : '-'}</td><td class="n">${e.fit && e.fit.p ? fmt(e.fit.p, 3) : '-'}</td>` +
      `<td class="n">${fmt(g.f_ext, 5)}</td><td class="n">${(100 * g.e_a21).toFixed(3)} %</td><td class="n"><b>${(100 * g.gci21).toFixed(3)} %</b></td><td class="muted">${warn}</td></tr>`;
  });
  $('convtable').innerHTML = h;
  const g0 = Object.values(d.entries)[0].gci;
  $('convnote').innerHTML = g0 ? `Three finest meshes: n_t = ${d.n_t.slice(-3).join(', ')}; r<sub>21</sub> = ${fmt(g0.r21, 3)}, r<sub>32</sub> = ${fmt(g0.r32, 3)}. ` +
    `GCI fine is the estimated relative error band of the finest mesh. Production mesh for ${NAMES[p]}: n_t = ${d.production_nt}.` : '';
}

// ------------------------------------------------------------------ fields tab
let fieldViewer = null, fieldData = null;
function initFields() {
  ['fieldstep'].forEach((id) => ($(id).onchange = () => loadField()));
  $('fieldrec').onchange = () => { S.fieldsel = $('fieldrec').value; loadField(); };
  ['fcmap', 'flimit'].forEach((id) => ($(id).onchange = () => drawField(true)));
  $('ftiles').onchange = () => drawField(false);
  $('fdeform').oninput = () => { $('fdeformv').textContent = $('fdeform').value; drawField(true); };
  $('fvtu').onclick = () => { if (S.fieldsel) window.location = `/api/fields/${S.fieldsel}/vtu`; };
  $('ffolder').onclick = () => S.fieldsel && api(`/api/fields/${S.fieldsel}/open?tool=folder`, { method: 'POST' }).catch((e) => toast(e.message));
  $('fcgx').onclick = () => S.fieldsel && api(`/api/fields/${S.fieldsel}/open?tool=cgx`, { method: 'POST' }).catch((e) => toast(e.message));
}
async function loadFieldRecords() {
  const recs = (await api('/api/records?source=calculix')).filter((r) => r.has_fields);
  const sel = $('fieldrec');
  if (!recs.length) {
    sel.innerHTML = '<option value="">no runs with field output yet</option>';
    $('fieldview').innerHTML = '<div class="placeholder">Stress fields exist for CalculiX runs made in this app with field output on.<br>Queue one in the Jobs tab (e.g. the selected cell at n_t = production level), then come back here.</div>';
    fieldViewer = null; return;
  }
  sel.innerHTML = recs.map((r) => `<option value="${r.key}">${NAMES[r.pattern]} ρ = ${r.rho}, n_t = ${r.n_t} (${r.material || ''}, ${(r.nodes || 0).toLocaleString()} nodes)</option>`).join('');
  if (S.fieldsel && recs.find((r) => r.key === S.fieldsel)) sel.value = S.fieldsel; else S.fieldsel = sel.value;
  if (!fieldData || fieldData.key !== S.fieldsel) loadField();
}
const STEP_TEXT = { E11: 'average strain ε_11 (FE frame)', E22: 'average strain ε_22', E33: 'average strain ε_33', G23: 'shear γ_23', G13: 'shear γ_13', G12: 'shear γ_12' };
async function loadField() {
  if (!S.fieldsel) return;
  const step = $('fieldstep').value || 'E11';
  try {
    $('fieldnote').textContent = 'loading (first view parses the .frd, can take a few seconds)...';
    const d = await api(`/api/fields/${S.fieldsel}?` + q({ step }));
    const keep = $('fieldstep').options.length > 0 && fieldData && fieldData.key === S.fieldsel;
    fieldData = { key: S.fieldsel, d, X: b64(d.X, Float32Array), U: b64(d.U, Float32Array), T: b64(d.T, Uint32Array), M: b64(d.mises, Float32Array) };
    $('fieldstep').innerHTML = d.steps.map((s) => `<option value="${s}">${s}${STEP_TEXT[s] ? ' - ' + STEP_TEXT[s] : ''}</option>`).join(''); $('fieldstep').value = d.step;
    if (!keep) { const umax = d.umax || 1e-9; const auto = Math.round(0.08 * Math.max(...d.box) / umax / 10) * 10; $('fdeform').max = Math.max(50, auto * 3); }
    drawField(keep);
  } catch (e) { $('fieldnote').textContent = e.message; }
}
function drawField(keepCam) {
  if (!fieldData) return;
  if (!fieldViewer || !document.body.contains(fieldViewer.el.firstChild)) fieldViewer = new Viewer($('fieldview'));
  const { d, X, U, T, M } = fieldData;
  const n = +$('ftiles').value, sc = +$('fdeform').value, kind = $('fcmap').value;
  const vmax = $('flimit').value === 'max' ? d.vmax : d.p995;
  const nv = X.length / 3, H = d.H, box = d.box;
  const P = new Float32Array(nv * 3 * n * n * n), UV = new Float32Array(nv * 2 * n * n * n), I = new Uint32Array(T.length * n * n * n);
  const NT = kind === 'rainbow' ? 12 : 256;                   // texels of the colour map; one extra grey texel for "above"
  let c = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
    const t = [i * box[0], j * box[1], k * box[2]];
    const Ht = [0, 1, 2].map((a) => H[a][0] * t[0] + H[a][1] * t[1] + H[a][2] * t[2]);
    const o = c * nv;
    for (let v = 0; v < nv; v++) {
      for (let a = 0; a < 3; a++) P[(o + v) * 3 + a] = X[v * 3 + a] + t[a] + sc * (U[v * 3 + a] + Ht[a]);
      const s = M[v] / vmax;
      UV[(o + v) * 2] = s > 1.0001 ? (NT + 0.5) / (NT + 1) : Math.min(s, 1) * NT / (NT + 1); UV[(o + v) * 2 + 1] = 0.5;
    }
    for (let e = 0; e < T.length; e++) I[c * T.length + e] = T[e] + o;
    c++;
  }
  const tex = new Uint8Array((NT + 1) * 4);
  for (let i = 0; i < NT; i++) { const col = cmap((i + 0.5) / NT, kind); tex.set([...col.map((x) => Math.round(x * 255)), 255], i * 4); }
  tex.set([158, 158, 158, 255], NT * 4);
  const map = new THREE.DataTexture(tex, NT + 1, 1, THREE.RGBAFormat); map.magFilter = THREE.NearestFilter; map.minFilter = THREE.NearestFilter; map.colorSpace = THREE.SRGBColorSpace; map.needsUpdate = true;
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(P, 3)); g.setAttribute('uv', new THREE.BufferAttribute(UV, 2)); g.setIndex(new THREE.BufferAttribute(I, 1)); g.computeVertexNormals();
  const smooth = $('fieldrec').selectedOptions[0] && /Gyroid/.test($('fieldrec').selectedOptions[0].text);
  fieldViewer.clear();
  fieldViewer.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({ map, roughness: 0.9, metalness: 0, flatShading: !smooth, side: THREE.DoubleSide })));
  fieldViewer.fit(keepCam);
  let cb = fieldViewer.el.querySelector('.colorbar');
  if (!cb) { cb = document.createElement('div'); cb.className = 'colorbar'; fieldViewer.el.appendChild(cb); }
  cb.innerHTML = `<div>von Mises stress (MPa), step ${d.step}</div><div class="bar" style="background:${cssGradient(kind)}"></div>` +
    `<div class="lbl"><span>0</span><span>${fmt(vmax / 2, 3)}</span><span>${fmt(vmax, 3)}</span></div>` +
    (vmax < d.vmax ? `<div style="margin-top:4px"><span style="display:inline-block;width:10px;height:10px;background:#9e9e9e;border-radius:2px"></span> above ${fmt(vmax, 3)} (max ${fmt(d.vmax, 3)})</div>` : '');
  $('fieldnote').innerHTML = `${d.n_vertices.toLocaleString()} surface nodes, ${d.n_triangles.toLocaleString()} triangles per cell. Mean von Mises ${fmt(d.vmean, 3)} MPa, 99.5th percentile ${fmt(d.p995, 3)} MPa, max ${fmt(d.vmax, 3)} MPa. ` +
    'The maximum sits at re-entrant wall junctions where the elastic solution is singular; it grows with mesh refinement, so the colour limit defaults to the 99.5th percentile (grey above). ' +
    'Strain steps apply 0.2 % average strain; virtual tests (UT, ET, SS) apply 1 MPa average stress. Displacements are scaled by the factor above; tiles repeat the periodic cell with u(x + t) = u(x) + H t.';
}

// ------------------------------------------------------------------ jobs
let jobSel = null;
function initJobs() {
  $('jsubmit').onclick = async () => {
    const nt = $('jnt').value.split(/[ ,;]+/).filter(Boolean).map(Number);
    try {
      const r = await api('/api/jobs', { method: 'POST', body: JSON.stringify({ pattern: $('jpattern').value, rho: +$('jrho').value, n_t: nt, material: $('jmat').value, field: $('jfield').checked, force: $('jforce').checked }) });
      $('jnote').textContent = r.map((j) => `${j.pattern} n_t=${j.n_t}: ${j.status}`).join('; ');
      pollJobs();
    } catch (e) { toast(e.message); }
  };
}
let lastDone = 0;
async function pollJobs() {
  let jobs;
  try { jobs = await api('/api/jobs'); } catch (e) { return; }
  const active = jobs.filter((j) => ['queued', 'running'].includes(j.status)).length;
  $('jobbadge').hidden = !active; $('jobbadge').textContent = active;
  const done = jobs.filter((j) => j.status === 'done').length;
  if (done !== lastDone) {
    if (lastDone) { S.trends = null; S.state = await api('/api/state'); fillSources(); if (['explore', 'mesh', 'fields', 'data', 'trends'].includes(S.tab)) refresh(); }
    lastDone = done;
  }
  if (S.tab !== 'jobs') return;
  $('jobsummary').textContent = `${jobs.length} jobs, ${active} active`;
  let h = '<tr><th>created</th><th>cell</th><th>ρ</th><th>n_t</th><th>material</th><th>status</th><th>last line</th><th></th></tr>';
  jobs.slice(0, 80).forEach((j) => {
    h += `<tr class="click ${j.id === jobSel ? 'sel' : ''}" data-id="${j.id}"><td>${j.created}</td><td>${NAMES[j.pattern]}</td><td class="n">${j.rho}</td><td class="n">${j.n_t}</td><td>${j.material.name}</td>` +
      `<td><span class="st ${j.status}">${j.status}</span></td><td class="muted" style="max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${(j.error || j.last || '').replace(/</g, '&lt;')}</td>` +
      `<td>${['queued', 'running'].includes(j.status) ? `<button data-cancel="${j.id}">cancel</button>` : ''}</td></tr>`;
  });
  $('jobtable').innerHTML = h;
  $('jobtable').querySelectorAll('tr.click').forEach((tr) => (tr.onclick = (e) => {
    if (e.target.dataset.cancel) { api(`/api/jobs/${e.target.dataset.cancel}/cancel`, { method: 'POST' }).then(pollJobs); return; }
    jobSel = tr.dataset.id; pollJobs();
  }));
  if (!jobSel && jobs.length) jobSel = jobs[0].id;
  if (jobSel) {
    const j = await api('/api/jobs/' + jobSel);
    $('logtitle').textContent = `${NAMES[j.pattern]} ρ = ${j.rho} n_t = ${j.n_t} (${j.id})`;
    const pre = $('joblog'); const atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
    pre.textContent = (j.log || []).join('\n') + (j.error ? '\n\nERROR: ' + j.error : '');
    if (atEnd) pre.scrollTop = pre.scrollHeight;
  }
}

// ------------------------------------------------------------------ data
function initData() { $('datasource').onchange = () => loadData(); }
async function loadData() {
  const src = $('datasource').value;
  const recs = await api('/api/records' + (src ? '?source=' + src : ''));
  $('csvlink').href = '/api/export/stiffness.csv?source=' + (src || S.source);
  $('datacount').textContent = `${recs.length} records`;
  let h = '<tr><th>source</th><th>cell</th><th>ρ</th><th>n_t</th><th>h (mm)</th><th>nodes</th><th>E_s</th><th>ν_s</th><th>created</th><th>solve (s)</th><th></th></tr>';
  recs.forEach((r) => {
    h += `<tr><td>${r.source}</td><td>${NAMES[r.pattern] || r.pattern}</td><td class="n">${r.rho}</td><td class="n">${r.n_t}</td><td class="n">${fmt(r.h, 3)}</td><td class="n">${r.nodes ? r.nodes.toLocaleString() : '-'}</td>` +
      `<td class="n">${r.E}</td><td class="n">${r.nu}</td><td>${r.created || ''}</td><td class="n">${r.time_solve ? r.time_solve.toFixed(1) : ''}</td>` +
      `<td>${r.origin === 'local' ? `<button data-del="${r.key}">delete</button>` : ''}</td></tr>`;
  });
  $('datatable').innerHTML = h;
  $('datatable').querySelectorAll('button[data-del]').forEach((b) => (b.onclick = async () => {
    if (!confirm('Delete this record from the registry? The run folder stays on disk.')) return;
    await api('/api/records/' + b.dataset.del, { method: 'DELETE' }); S.trends = null; S.state = await api('/api/state'); fillSources(); loadData();
  }));
}

// ------------------------------------------------------------------ settings
function initSettings() {
  $('sdetect').onclick = async () => {
    const c = await api('/api/settings/detect');
    const list = (c.ccx || []).map((p) => `<a href="#" data-p="${p}">${p}</a>`).join('<br>');
    $('sdetected').innerHTML = list ? 'Found (click to use):<br>' + list : 'No ccx found in PATH or the usual folders. Paste the full path of ccx_i4_static.exe.';
    $('sdetected').querySelectorAll('a').forEach((a) => (a.onclick = (e) => { e.preventDefault(); $('sccx').value = a.dataset.p; }));
    if (!$('scgx').value && c.cgx && c.cgx.length) $('scgx').value = c.cgx[0];
  };
  $('ssave').onclick = async () => {
    try {
      const r = await api('/api/settings', { method: 'POST', body: JSON.stringify({ ccx_path: $('sccx').value.trim(), cgx_path: $('scgx').value.trim(), threads: +$('sthreads').value, field_output: $('sfield').checked }) });
      S.state.settings = r.settings; S.state.solver = r.solver; renderSolver(r.solver, r.cgx); loadSettings(); toast(r.solver.ok ? `CalculiX ${r.solver.version} ready` : r.solver.message);
    } catch (e) { toast(e.message); }
  };
  $('madd').onclick = async () => {
    try {
      S.state.materials = await api('/api/materials', { method: 'POST', body: JSON.stringify({ name: $('mname').value.trim(), E: +$('mE').value, nu: +$('mnu').value, source: $('msrc').value.trim() }) });
      fillMaterials(); loadSettings();
    } catch (e) { toast(e.message); }
  };
}
function loadSettings() {
  const st = S.state, s = st.settings;
  $('sccx').value = s.ccx_path || ''; $('scgx').value = s.cgx_path || ''; $('sthreads').value = s.threads; $('sfield').checked = !!s.field_output;
  const sol = st.solver;
  $('sccxstatus').innerHTML = sol.ok ? `In use: <code>${sol.path}</code> (CalculiX ${sol.version})${s.ccx_path ? '' : ' - auto-detected'}` : `<span style="color:var(--bad)">${sol.message}</span>`;
  $('sws').textContent = `Workspace (results registry, job folders): ${st.workspace}`;
  $('mattable').innerHTML = '<tr><th>name</th><th>E (MPa)</th><th>ν</th><th>source</th><th></th></tr>' + Object.values(st.materials).map((m) =>
    `<tr><td>${m.name}</td><td class="n">${m.E}</td><td class="n">${m.nu}</td><td class="muted">${m.source || ''}</td><td>${m.user ? `<button data-mdel="${m.name}">remove</button>` : ''}</td></tr>`).join('');
  $('mattable').querySelectorAll('button[data-mdel]').forEach((b) => (b.onclick = async () => { S.state.materials = await api('/api/materials/' + encodeURIComponent(b.dataset.mdel), { method: 'DELETE' }); fillMaterials(); loadSettings(); }));
}

init().catch((e) => toast('start failed: ' + e.message, 10000));
