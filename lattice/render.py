"""Shaded 3D pictures of triangle meshes with three.js in headless Chromium (Playwright).

render(mesh, png, scalars=None, ...) draws one mesh, optionally coloured by a nodal scalar
(e.g. von Mises stress) with a jet-like colour map. Needs a local copy of three.min.js
(THREE_JS environment variable or lattice/vendor/three.min.js).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THREE_JS = os.environ.get('THREE_JS', os.path.join(HERE, 'vendor', 'three.min.js'))

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<style>html,body{margin:0;background:#fff}canvas{display:block}</style>
<script src="http://render.local/three.min.js"></script></head><body><script>
(async function(){
  const cfg = await (await fetch('http://render.local/cfg.json')).json();
  const buf = async n => await (await fetch('http://render.local/' + n)).arrayBuffer();
  const pos = new Float32Array(await buf('pos.bin')), idx = new Uint32Array(await buf('idx.bin'));
  const W = cfg.width, H = cfg.height;
  const r = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
  r.setPixelRatio(1); r.setSize(W, H); r.setClearColor(0xffffff, 1);
  document.body.appendChild(r.domElement);
  const scene = new THREE.Scene();
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  let mat;
  if (cfg.colors) {
    const col = new Float32Array(await buf('col.bin'));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    mat = new THREE.MeshLambertMaterial({vertexColors: true, side: THREE.DoubleSide});
  } else {
    mat = new THREE.MeshStandardMaterial({color: cfg.color, roughness: 0.55, metalness: 0.05, side: THREE.DoubleSide});
  }
  const gm = cfg.flat ? g.toNonIndexed() : g;
  gm.computeVertexNormals();
  scene.add(new THREE.Mesh(gm, mat));
  if (cfg.edges) {
    const e = new THREE.EdgesGeometry(g, cfg.edge_angle);
    scene.add(new THREE.LineSegments(e, new THREE.LineBasicMaterial({color: 0x3a3a3a})));
  }
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a8a, cfg.colors ? 1.6 : 1.1));
  const d1 = new THREE.DirectionalLight(0xffffff, cfg.colors ? 1.0 : 1.9); d1.position.set(...cfg.light1); scene.add(d1);
  const d2 = new THREE.DirectionalLight(0xffffff, 0.5); d2.position.set(...cfg.light2); scene.add(d2);
  const c = cfg.center, R = cfg.radius;
  const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01 * R, 100 * R);
  const az = cfg.azimuth * Math.PI / 180, el = cfg.elevation * Math.PI / 180;
  cam.up.set(0, 0, 1);
  cam.position.set(c[0] + 10 * R * Math.cos(el) * Math.cos(az), c[1] + 10 * R * Math.cos(el) * Math.sin(az),
                   c[2] + 10 * R * Math.sin(el));
  cam.lookAt(c[0], c[1], c[2]);
  cam.updateMatrixWorld();
  // fit: project the bounding-box corners
  const v = new THREE.Vector3(); let xmin=1e9,xmax=-1e9,ymin=1e9,ymax=-1e9;
  const inv = cam.matrixWorldInverse;
  for (const p of cfg.corners) { v.set(...p).applyMatrix4(inv); xmin=Math.min(xmin,v.x); xmax=Math.max(xmax,v.x); ymin=Math.min(ymin,v.y); ymax=Math.max(ymax,v.y); }
  const m = cfg.margin, cx = (xmin+xmax)/2, cy = (ymin+ymax)/2;
  let hw = (xmax-xmin)/2*(1+m), hh = (ymax-ymin)/2*(1+m);
  if (hw / hh > W / H) hh = hw * H / W; else hw = hh * W / H;
  cam.left = cx-hw; cam.right = cx+hw; cam.top = cy+hh; cam.bottom = cy-hh; cam.updateProjectionMatrix();
  r.render(scene, cam);
  window.DONE = true;
})();
</script></body></html>"""


def jet(s):
    """Colour map close to the Abaqus default rainbow: blue -> cyan -> green -> yellow -> red."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    stops = np.array([[0.0, 0.0, 1.0], [0.0, 0.73, 1.0], [0.0, 1.0, 0.27], [0.67, 1.0, 0.0],
                      [1.0, 0.73, 0.0], [1.0, 0.0, 0.0]])
    x = s * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    f = (x - i)[:, None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def jet_banded(s, bands=12):
    """Banded contours as in Abaqus/Viewer (12 bands)."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0 - 1e-9)
    return jet((np.floor(s * bands) + 0.5) / bands)


def render(vertices, faces, png, scalars=None, vrange=None, banded=True, color='#c9ccd2', azimuth=-58,
           elevation=24, width=900, height=900, margin=0.04, edges=False, edge_angle=30, flat=False):
    from playwright.sync_api import sync_playwright
    v = np.ascontiguousarray(vertices, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    lo, hi = v.min(0), v.max(0)
    corners = [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    cfg = dict(width=width, height=height, color=color, azimuth=azimuth, elevation=elevation, margin=margin,
               center=((lo + hi) / 2).tolist(), radius=float(np.linalg.norm(hi - lo) / 2),
               corners=[[float(c) for c in p] for p in corners], light1=[0.6, -1.0, 1.4], light2=[-1.0, 0.8, 0.3],
               colors=scalars is not None, edges=edges, edge_angle=edge_angle, flat=flat)
    files = {'cfg.json': json.dumps(cfg).encode(), 'pos.bin': v.tobytes(), 'idx.bin': f.tobytes(),
             'three.min.js': open(THREE_JS, 'rb').read()}
    if scalars is not None:
        a, b = vrange if vrange is not None else (float(np.min(scalars)), float(np.max(scalars)))
        s = (np.asarray(scalars, dtype=float) - a) / max(b - a, 1e-30)
        files['col.bin'] = np.ascontiguousarray((jet_banded(s) if banded else jet(s)), dtype=np.float32).tobytes()
    with sync_playwright() as p:
        br = p.chromium.launch(args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
        pg = br.new_page(viewport={'width': width, 'height': height}, device_scale_factor=1)

        def handle(route):
            name = route.request.url.split('render.local/')[-1]
            ctype = 'application/javascript' if name.endswith('.js') else (
                'application/json' if name.endswith('.json') else 'application/octet-stream')
            route.fulfill(status=200, body=files[name], headers={'Content-Type': ctype,
                                                                 'Access-Control-Allow-Origin': '*'})
        pg.route('http://render.local/**', handle)
        pg.route('http://page.local/', lambda rt: rt.fulfill(status=200, body=PAGE, headers={'Content-Type': 'text/html'}))
        pg.goto('http://page.local/')
        pg.wait_for_function('window.DONE === true', timeout=600000)
        pg.locator('canvas').screenshot(path=png)
        br.close()
    return png


def tile(vertices, faces, box, n, tol=1e-6):
    """Copies of a periodic cell, n = (nx, ny, nz). Faces lying on cell faces that end up inside
    the block (the closing caps of the STL) are removed, so no seams show."""
    V, F = [], []
    k = 0
    for i in range(n[0]):
        for j in range(n[1]):
            for l in range(n[2]):
                V.append(vertices + np.array([i * box[0], j * box[1], l * box[2]]))
                F.append(faces + k * len(vertices))
                k += 1
    V, F = np.concatenate(V), np.concatenate(F)
    keep = np.ones(len(F), dtype=bool)
    for a in range(3):
        c = V[F][:, :, a]                                   # (m, 3)
        q = np.round(c / box[a])
        on = (np.abs(c - q * box[a]) < tol * box[a]).all(1) & (q == q[:, :1]).all(1)
        inner = on & (q[:, 0] > 0) & (q[:, 0] < n[a])
        keep &= ~inner
    return V, F[keep]
