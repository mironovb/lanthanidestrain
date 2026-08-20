#!/usr/bin/env python3
"""VR edge asset for the collaborator's expanded population (Aug-20 update).

Starts from our has3d asset (1,235 complexes = shipped 956 + 279 borderline
rebuilds) and appends every structure in ``collaborator_update/geometries/``
whose build id the asset does not already carry -- these are the
collaborator's re-optimised/repaired builds behind his new ``geometry_ok``
(4,746 -> 5,479 rows).  Same conventions as build_vr_has3d: 4.0 A cutoff
edges, zero triangles, NaN partial charges for appended complexes (the
encoder's charge_missing flag handles them), coreCN per build taken from the
collaborator's parquet for donor detection.

Writes automl/artifacts/vr_cutoff/collab/vietoris_rips_inputs.npz (+ meta).

Usage:  PYTHONPATH=$PWD python3 -m automl.topo.build_vr_collab
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "automl/artifacts/vr_cutoff/has3d/vietoris_rips_inputs.npz"
GEOMS = REPO / "collaborator_update/geometries"
CDATA = REPO / "collaborator_update/dataset.parquet"
OUT = REPO / "automl/artifacts/vr_cutoff/collab"

CUTOFF = 4.0


def main() -> int:
    import ase.data as ad
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.geometry_features import read_extxyz
    from automl.topo.build_vr_conformers import donor_indices
    from automl.topo.build_neighbor_graph import _edges_cutoff
    from automl.topo.build_vr_gxtb import LANTH

    z = np.load(BASE)
    have = {str(b) for b in z["build_ids"]}

    c = pd.read_parquet(CDATA, columns=["build_id",
                                        "geometry_feature_build_id",
                                        "coreCN"])
    cn_map = {}
    for col in ("build_id", "geometry_feature_build_id"):
        for bid, cn in zip(c[col].astype(str), c["coreCN"]):
            if pd.notna(cn):
                cn_map.setdefault(bid, int(cn))

    files = {}
    for p in glob.glob(str(GEOMS / "*.xyz")):
        bid = os.path.basename(p).split("_")[1].split(".")[0]
        files[bid] = p
    todo = sorted(set(files) - have)
    print(f"asset has {len(have)}; his files {len(files)}; appending {len(todo)}")

    co = [z["coordinates"]]; zs = [z["atomic_numbers"]]
    pc = [z["partial_charges"]]; im = [z["is_metal"]]
    idn = [z["is_coord_donor"]]
    nptr = list(z["node_ptr"])
    e_idx = [z["edge_index"].T]; e_filt = [z["edge_filtration"]]
    e_ptr = list(z["edge_ptr"])
    bids = [str(b) for b in z["build_ids"]]

    skipped = []
    for bid in todo:
        geom = read_extxyz(Path(files[bid]))
        sym = list(geom.symbols)
        xyz = np.asarray(geom.coordinates, dtype=float)
        metal = next((s for s in sym if s in LANTH), None)
        if metal is None:
            skipped.append((bid, "no metal")); continue
        mi, don = donor_indices(sym, xyz, metal, cn_map.get(bid, 9))
        if mi is None:
            skipped.append((bid, "no donor shell")); continue
        n = len(sym)
        co.append(xyz.astype(np.float32))
        zs.append(np.array([ad.atomic_numbers.get(s, 0) for s in sym],
                           np.int16))
        pc.append(np.full(n, np.nan, np.float32))
        mm = np.zeros(n, np.int8); mm[mi] = 1
        dd = np.zeros(n, np.int8); dd[list(don)] = 1
        im.append(mm); idn.append(dd)
        pairs, filt = _edges_cutoff(xyz, CUTOFF)
        e_idx.append(pairs + nptr[-1])
        e_filt.append(filt.astype(np.float32))
        e_ptr.append(e_ptr[-1] + len(pairs))
        nptr.append(nptr[-1] + n)
        bids.append(bid)

    n_c = len(bids)
    payload = dict(
        coordinates=np.concatenate(co).astype(np.float32),
        atomic_numbers=np.concatenate(zs),
        partial_charges=np.concatenate(pc).astype(np.float32),
        is_metal=np.concatenate(im), is_coord_donor=np.concatenate(idn),
        node_ptr=np.asarray(nptr, np.int64),
        edge_index=np.concatenate(e_idx).T.astype(np.int64),
        edge_filtration=np.concatenate(e_filt).astype(np.float32),
        edge_ptr=np.asarray(e_ptr, np.int64),
        triangle_index=np.zeros((3, 0), np.int64),
        triangle_filtration=np.zeros(0, np.float32),
        triangle_ptr=np.zeros(n_c + 1, np.int64),
        build_ids=np.array(bids, dtype="<U32"))
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "vietoris_rips_inputs.tmp.npz"
    np.savez_compressed(tmp, **payload)
    tmp.replace(OUT / "vietoris_rips_inputs.npz")
    (OUT / "meta.json").write_text(json.dumps(
        {"name": "collab", "cutoff": CUTOFF, "n_complexes": n_c,
         "base": "has3d(1235)", "appended": n_c - len(have),
         "skipped": skipped, "triangles": 0}, indent=2) + "\n")
    print(f"[vr_collab] {n_c} complexes ({n_c - len(have)} appended, "
          f"{len(skipped)} skipped) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
