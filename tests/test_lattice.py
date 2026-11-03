"""python -m pytest -q   (the solver tests need ccx on the PATH and are skipped otherwise)"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice import inp, laws, mesh, post, results                       # noqa: E402
from lattice.geometry import PATTERNS, density, wall_thickness             # noqa: E402
from lattice.pbc import BoxCell, periodic_classes                         # noqa: E402

HAS_CCX = shutil.which('ccx') is not None


@pytest.mark.parametrize('name', ['grid', 'triangles', 'honeycomb', 'cubic'])
def test_thickness_inverts(name):
    for rho in (0.1, 0.4, 0.7):
        assert abs(density(name, wall_thickness(name, rho)) - rho) < 1e-12


@pytest.mark.parametrize('name', PATTERNS)
def test_mesh_periodic_and_exact_density(name):
    m = mesh.build_mesh(name, 0.4, n_t=2 if name != 'gyroid' else 1.5)
    periodic_classes(m.nodes, BoxCell(*m.box))                            # raises if not periodic
    assert abs(m.volume_fraction - 0.4) < (5e-4 if name == 'gyroid' else 1e-9)


def test_mixture_law_recovers_coefficients():
    rho = np.linspace(0.1, 0.7, 7)
    c = np.array([-0.5, 0.3])
    y = laws.mixture(rho, c)
    assert np.allclose(laws.fit_mixture(rho, y, 2), c)


def test_extrapolation():
    h = np.array([1 / 2, 1 / 3, 1 / 4, 1 / 6])
    e = laws.extrapolate(h, 10 + 3 * h ** 2)
    assert abs(e['f0'] - 10) < 1e-6 and abs(e['p'] - 2) < 0.05


@pytest.mark.skipif(not HAS_CCX, reason='ccx not installed')
def test_solid_cell_returns_bulk_stiffness(tmp_path):
    xs = np.linspace(0, 20, 3)
    n, h = mesh.tensor_hex(xs, np.linspace(0, 30, 3), np.linspace(0, 10, 3), lambda I, J, K: np.ones_like(I, bool))
    qn, h20 = mesh.make_quadratic(n, mesh.fix_hex_orientation(n, h), mesh.HEX_EDGES)
    m = mesh.Mesh(qn, h20, 'C3D20R', (20, 30, 10), dict(pattern='grid', rho=1.0, n_t=0))
    p = str(tmp_path / 'solid.inp')
    meta = inp.write_deck(m, p, 'ccx', field_output=False)
    subprocess.run(['ccx', '-i', p[:-4]], capture_output=True, check=True)
    st = results.parse_ccx_dat(p[:-4] + '.dat', meta, m.nodes)
    C, _ = results.stiffness(meta, st)
    assert np.allclose(C, post.iso_C(meta['E'], meta['nu']), rtol=1e-6, atol=1e-6)
    vt = results.virtual_test_results(meta, st)
    assert abs(post.test_moduli(vt['UT_x']['sigma_voigt'], vt['UT_x']['eps_voigt'], 'UT_x')['E'] - meta['E']) < 1e-6 * meta['E']
    assert abs(post.test_moduli(vt['SS_xy']['sigma_voigt'], vt['SS_xy']['eps_voigt'], 'SS_xy')['G']
               - meta['E'] / (2 * (1 + meta['nu']))) < 1e-6 * meta['E']


def test_directional_moduli():
    C = post.iso_C(2100.0, 0.35)
    rng = np.random.default_rng(0)
    n = rng.normal(size=(50, 3))
    n /= np.linalg.norm(n, axis=1)[:, None]
    assert np.allclose(post.directional_E(C, n), 2100.0)
    m = np.cross(n, [0.3, 0.5, 0.8])
    m /= np.linalg.norm(m, axis=1)[:, None]
    assert np.allclose(post.directional_G(C, n, m), 2100.0 / 2.7)
    # transversely isotropic about x: E constant in the y-z plane, E(x) = E1
    Cti = np.zeros((6, 6))
    Cti[0, 0], Cti[1, 1], Cti[2, 2] = 700, 300, 300
    Cti[0, 1] = Cti[1, 0] = Cti[0, 2] = Cti[2, 0] = 150
    Cti[1, 2] = Cti[2, 1] = 120
    Cti[3, 3] = 90
    Cti[4, 4] = Cti[5, 5] = 130
    a = np.linspace(0, np.pi, 13)
    E = post.directional_E(Cti, np.stack([0 * a, np.cos(a), np.sin(a)], 1))
    assert np.allclose(E, E[0])
    assert abs(post.directional_E(Cti, np.array([1.0, 0, 0])) - post.engineering_constants(Cti)['E1']) < 1e-9
