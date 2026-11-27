"""ParaView export: one .vtu (VTK XML unstructured grid, binary, base64 inline) per run with the
quadratic elements and, per step, displacement U_<step>, stress S_<step> (xx yy zz xy yz zx) and von Mises.

Node orders of C3D20 and C3D10 match VTK_QUADRATIC_HEXAHEDRON (25) and VTK_QUADRATIC_TETRA (24).
No external package needed.
"""
import base64
import struct

import numpy as np

from .fields import mises

VTK_TYPE = {'C3D20R': 25, 'C3D20': 25, 'C3D10': 24}


def _b64(a):
    a = np.ascontiguousarray(a)
    raw = a.tobytes()
    return base64.b64encode(struct.pack('<Q', len(raw)) + raw).decode('ascii')


def _array(name, a, ncomp):
    t = {np.dtype('float32'): 'Float32', np.dtype('float64'): 'Float64', np.dtype('int64'): 'Int64',
         np.dtype('uint8'): 'UInt8'}[a.dtype]
    return (f'<DataArray type="{t}" Name="{name}" NumberOfComponents="{ncomp}" format="binary">'
            f'{_b64(a)}</DataArray>')


def write(path, nodes, elements, etype, point_data=None):
    """point_data: {name: (n,) or (n, k) array}."""
    nodes = np.asarray(nodes, np.float64)
    el = np.asarray(elements, np.int64)
    n, ne, npe = len(nodes), len(el), el.shape[1]
    pd = []
    for k, v in (point_data or {}).items():
        v = np.asarray(v, np.float32)
        pd.append(_array(k, v, 1 if v.ndim == 1 else v.shape[1]))
    xml = ['<?xml version="1.0"?>',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           '<UnstructuredGrid>', f'<Piece NumberOfPoints="{n}" NumberOfCells="{ne}">',
           '<PointData>', *pd, '</PointData>',
           '<Points>', _array('Points', nodes, 3), '</Points>',
           '<Cells>', _array('connectivity', el.ravel(), 1),
           _array('offsets', np.arange(1, ne + 1, dtype=np.int64) * npe, 1),
           _array('types', np.full(ne, VTK_TYPE[str(etype)], np.uint8), 1), '</Cells>',
           '</Piece>', '</UnstructuredGrid>', '</VTKFile>']
    with open(path, 'w') as f:
        f.write('\n'.join(xml))


def write_run(path, nodes, elements, etype, steps, names):
    """steps: [(U, S), ...] from fields.parse_frd; names: step names in the same order."""
    pd = {}
    for name, (U, S) in zip(names, steps):
        pd[f'U_{name}'] = U
        pd[f'S_{name}'] = S
        pd[f'Mises_{name}'] = mises(S)
    write(path, nodes, elements, etype, pd)
