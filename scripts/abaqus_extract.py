"""Run with Abaqus Python (Abaqus 2024 or newer, numpy included):

    abaqus python abaqus_extract.py JOB.odb [--fields]

Reads, for every step, the last frame of the ODB written from one of our decks and saves
    JOB_rp.json       U and RF of the reference points RP_X, RP_Y, RP_Z per step (-> lattice.results)
    JOB_fields.npz    with --fields: node coordinates, connectivity, and per step the nodal displacement
                      and the nodal-averaged stress (xx yy zz xy yz zx) for the contour pictures
The reference-point node labels come from JOB.json, written next to the deck.

Written against the Abaqus Scripting Reference; not run here (no Abaqus licence in this environment).
Check it on the first job and report any error.
"""
import json
import os
import sys

import numpy as np
from abaqusConstants import ELEMENT_NODAL
from odbAccess import openOdb


def main(odb_path, fields):
    base = odb_path[:-4]
    meta = json.load(open(base + '.json'))
    rp = meta['rp_labels']
    odb = openOdb(odb_path, readOnly=True)
    inst = list(odb.rootAssembly.instances.values())[0]          # flat input: PART-1-1
    rp_nodes = [inst.getNodeFromLabel(l) for l in rp]
    out = {'odb': os.path.basename(odb_path), 'instance': inst.name, 'steps': []}
    U_all, S_all, names = [], [], []
    n = meta['nodes']
    for name, step in odb.steps.items():
        fr = step.frames[-1]
        U, RF = fr.fieldOutputs['U'], fr.fieldOutputs['RF']
        u = [list(map(float, U.getSubset(region=nd).values[0].data)) for nd in rp_nodes]
        f = [list(map(float, RF.getSubset(region=nd).values[0].data)) for nd in rp_nodes]
        out['steps'].append({'name': name, 'frame_time': float(fr.frameValue), 'U': u, 'RF': f})
        if fields:
            Un = np.zeros((n, 3))
            for b in U.bulkDataBlocks:
                lab = np.asarray(b.nodeLabels)
                keep = lab <= n
                Un[lab[keep] - 1] = np.asarray(b.data)[keep]
            Ssum, cnt = np.zeros((n, 6)), np.zeros(n)
            for b in fr.fieldOutputs['S'].getSubset(position=ELEMENT_NODAL).bulkDataBlocks:
                lab = np.asarray(b.nodeLabels)
                d = np.asarray(b.data)[:, [0, 1, 2, 3, 5, 4]]          # S11 S22 S33 S12 S13 S23 -> xx yy zz xy yz zx
                np.add.at(Ssum, lab - 1, d)
                np.add.at(cnt, lab - 1, 1)
            U_all.append(Un)
            S_all.append(Ssum / np.maximum(cnt, 1)[:, None])
            names.append(name)
    json.dump(out, open(base + '_rp.json', 'w'), indent=1)
    if fields:
        xyz = np.zeros((n, 3))
        for nd in inst.nodes:
            if nd.label <= n:
                xyz[nd.label - 1] = nd.coordinates
        conn = np.array([e.connectivity for e in inst.elements], dtype=np.int64) - 1
        np.savez_compressed(base + '_fields.npz', nodes=xyz, elements=conn, etype=meta['etype'],
                            box=np.array(meta['box']), steps=np.array(names), U=np.array(U_all), S=np.array(S_all))
    odb.close()
    print('wrote', base + '_rp.json', '(and _fields.npz)' if fields else '')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    main(args[0], '--fields' in sys.argv)
