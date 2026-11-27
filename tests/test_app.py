"""App-level checks: solver detection, frames, convergence, constant laws, VTU, API."""
import os
import stat
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from lattice import convergence, post, solver, vtu


def test_ccx_detection_explicit_env_and_name(tmp_path, monkeypatch):
    exe = tmp_path / ('ccx_i4_static.exe' if sys.platform.startswith('win') else 'ccx_i4_static')
    exe.write_text('#!/bin/sh\necho "Version 2.22"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    assert solver.find_ccx(str(exe)) == str(exe)
    monkeypatch.setenv('LATTICE_CCX', str(exe))
    assert solver.find_ccx(None) == str(exe)
    monkeypatch.delenv('LATTICE_CCX')
    monkeypatch.setenv('PATH', str(tmp_path))
    assert os.path.basename(solver.find_ccx(None)).startswith('ccx_i4_static')


def test_print_frame_roundtrip_and_rotation():
    rng = np.random.default_rng(1)
    A = rng.random((6, 6))
    C = A @ A.T + 6 * np.eye(6)
    R = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.]])           # x_print = R x_fe
    assert np.allclose(post.rotate(C, R), post.to_print_frame(C, 'honeycomb'))
    assert np.allclose(post.to_fe_frame(post.to_print_frame(C, 'grid'), 'grid'), C)
    n = np.array([0.2, -0.5, 0.84])
    n /= np.linalg.norm(n)
    assert np.isclose(post.directional_E(C, n), post.directional_E(post.to_print_frame(C, 'grid'), R @ n))


def test_gci_recovers_order_unequal_ratios():
    h = np.array([1.0, 0.7, 0.45, 0.3])
    f = 10.0 + 3.0 * h ** 1.7
    g = convergence.gci(h, f)
    assert abs(g['p'] - 1.7) < 1e-6 and abs(g['f_ext'] - 10.0) < 1e-6
    assert not g['oscillatory']


def test_constants_roundtrip_all_classes():
    from lattice_app.analysis import C_from_constants, reduced_constants
    for p, C in (('cubic', post.iso_C(100, 0.3) + np.diag([0, 0, 0, -5, -5, -5])),
                 ('honeycomb', post.to_print_frame(post.symmetrize(_tiso(), 'honeycomb'), 'honeycomb')),
                 ('grid', post.to_print_frame(post.symmetrize(_tiso() + np.diag([0, 0, 0, 3, 0, 0]), 'grid'), 'grid'))):
        assert np.allclose(C_from_constants(p, reduced_constants(p, C)), C, rtol=1e-9, atol=1e-9)


def _tiso():
    C = np.zeros((6, 6))                                            # transversely isotropic about FE x
    C[0, 0], C[1, 1], C[2, 2] = 700, 250, 250
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = 150
    C[1, 2] = C[2, 1] = 190
    C[3, 3] = 30
    C[4, 4] = C[5, 5] = 130
    return C


def test_vtu_is_valid_xml(tmp_path):
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5],
                      [.5, 0, .5], [0, .5, .5]], float)
    p = tmp_path / 't.vtu'
    vtu.write_run(str(p), nodes, np.arange(10)[None], 'C3D10', [(np.zeros((10, 3)), np.ones((10, 6)))], ['E11'])
    root = ET.parse(p).getroot()
    names = [a.get('Name') for a in root.iter('DataArray')]
    assert {'U_E11', 'S_E11', 'Mises_E11', 'connectivity', 'types'} <= set(names)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('LATTICE_WORKSPACE', str(tmp_path))
    from fastapi.testclient import TestClient

    from lattice_app.server import create_app
    return TestClient(create_app())


def test_api_explore_matches_fe_at_data_points(client):
    st = client.get('/api/state').json()
    assert {s['id'] for s in st['sources']} >= {'abaqus', 'calculix-ref'}
    for p in st['patterns']:
        d = client.get('/api/explore', params=dict(pattern=p, rho=0.4)).json()
        assert d['ok'] and d['fe']['law_vs_fe'] < 5e-3, (p, d.get('fe'))
        d = client.get('/api/explore', params=dict(pattern=p, rho=0.35)).json()
        assert d['ok'] and not d['extrapolated']
    c = client.get('/api/convergence', params=dict(pattern='honeycomb', rho=0.4)).json()
    assert c['ok'] and c['entries']['C11']['gci']['gci21'] < 0.01
    m = client.get('/api/mesh', params=dict(pattern='cubic', rho=0.3, n_t=2)).json()
    assert abs(m['volume_fraction'] - 0.3) < 1e-6


def test_api_settings_rejects_missing_file(client):
    r = client.post('/api/settings', json=dict(ccx_path='/no/such/ccx'))
    assert r.status_code == 400
