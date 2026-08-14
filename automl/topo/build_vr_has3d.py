#!/usr/bin/env python3
"""Extend the VR edge asset to the expanded (has3d) population.

The shipped asset covers the 956 geometry_ok complexes; 279 further builds
back the borderline-QC rows that the August campaign's fresh-444 confirmation
pairs live on.  All 279 have completed structures on local disk
(geometry_index_merged.csv, status existing_ok).  This module writes
``automl/artifacts/vr_cutoff/has3d/vietoris_rips_inputs.npz``:

  * the 956 shipped complexes copied VERBATIM from the shipped npz (byte-equal
    node/edge arrays, so the ok rows see exactly the geometry they always saw);
  * the 279 extra complexes appended with the same 4.0 A cutoff edge build
    (``build_neighbor_graph._edges_cutoff``) and donor detection
    (``build_vr_conformers.donor_indices``);
  * partial charges NaN for the extras -- the shipped charges came from the
    xTB feature pipeline which never ran on these builds; the encoder's
    ``charge_missing`` input flag exists for exactly this case.

Zero triangles (the vr_gxtb convention): usable by --arch dist, or snn with
--no-triangles.

Usage:  PYTHONPATH=$PWD python3 -m automl.topo.build_vr_has3d
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "data/processed/feature_blocks/vietoris_rips_inputs.npz"
INDEX = REPO / "data/processed/geometry_index_merged.csv"
MATRIX = REPO / "automl/artifacts/matrix/matrix.parquet"
OUT = REPO / "automl/artifacts/vr_cutoff/has3d"

CUTOFF = 4.0


def local_xyz(orig: str) -> str | None:
    base = os.path.basename(str(orig))
    el = base.split("_")[0]
    for cand in (REPO / f"data/geometries/{el}/{base}",
                 REPO / f"data/processed/geometries_family_regenerated/{base}",
                 REPO / "data/processed/geometries_regenerated_fail_long_bond"
                        f"/accepted/{el}/{base}"):
        if cand.exists():
            return str(cand)
    return None


def main() -> int:
    import ase.data as ad
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.geometry_features import read_extxyz
    from automl.topo.build_vr_conformers import donor_indices
    from automl.topo.build_neighbor_graph import _edges_cutoff
    from automl.topo.build_vr_gxtb import LANTH

    z = np.load(SHIPPED)
    shipped_ids = set(str(b) for b in z["build_ids"])

    m = pd.read_parquet(MATRIX, columns=["geometry_feature_build_id",
                                         "has_3d"])
    need = sorted(set(m[m["has_3d"].astype(bool)]
                      ["geometry_feature_build_id"].astype(str)) - shipped_ids)
    gi = pd.read_csv(INDEX)
    gi = gi[gi["build_id"].astype(str).isin(need)]
    print(f"extra builds: {len(need)}, indexed: {len(gi)}")

    co = [z["coordinates"]]
    zs = [z["atomic_numbers"]]
    pc = [z["partial_charges"]]
    im = [z["is_metal"]]
    idn = [z["is_coord_donor"]]
    nptr = list(z["node_ptr"])
    e_idx = [z["edge_index"].T]
    e_filt = [z["edge_filtration"]]
    e_ptr = list(z["edge_ptr"])
    bids = [str(b) for b in z["build_ids"]]

    skipped = []
    for _, row in gi.iterrows():
        bid = str(row["build_id"])
        p = local_xyz(row["xyz_path"])
        if p is None:
            skipped.append((bid, "no local file")); continue
        geom = read_extxyz(Path(p))
        sym = list(geom.symbols)
        xyz = np.asarray(geom.coordinates, dtype=float)
        metal = next((s for s in sym if s in LANTH), None)
        if metal is None:
            skipped.append((bid, "no metal")); continue
        cn = int(row.get("coreCN", 9) or 9)
        mi, don = donor_indices(sym, xyz, metal, cn)
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
        {"name": "has3d", "cutoff": CUTOFF, "n_complexes": n_c,
         "n_shipped": len(shipped_ids), "n_extra": n_c - len(shipped_ids),
         "extra_charges": "NaN (charge_missing flag carries it)",
         "skipped": skipped, "triangles": 0}, indent=2) + "\n")
    print(f"[vr_has3d] {n_c} complexes "
          f"({n_c - len(shipped_ids)} extra, {len(skipped)} skipped) -> {OUT}")
    for s in skipped:
        print("  skipped:", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
