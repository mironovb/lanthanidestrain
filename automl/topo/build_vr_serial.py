#!/usr/bin/env python3
"""Trainable neighbour-graph assets from the serial (in-correspondence) structures.

Why this exists
---------------
`serial_metals.py` rebuilt the lanthanide series in correspondence: adjacent-pair
heavy-atom RMSD fell from **5.46 A to 0.0120 A**, the contraction signal-to-noise
rose from **0.14 to 0.799**, and the response correlation with the Shannon radius
step went from r = 0.197 to **r = 0.574**.  That is a much cleaner geometric
signal than anything the models have ever been given.

Whether it is a *more useful* signal is a separate, empirical question.  The
interpolation test showed the serial deformation is essentially rank-1 in the
metal coordinate, and it is tempting to conclude from that alone that no model
can gain -- but "one degree of freedom" and "no usable signal" are different
claims.  A single CLEAN degree of freedom may well be worth more than a single
one buried in conformer noise.  This module makes that testable.

Two arms, identical in every respect but the coordinates
--------------------------------------------------------
    serial   the metal-substituted, in-correspondence geometries
    orig     the SAME 786 complexes' shipped coordinates

Same complexes, same build ids, same order, same row set, same graph
construction, same node features.  The only difference is where the atoms are.
Anything else would confound geometry with sample.

Assets carry NO triangles (they are for ``--arch dist``), matching the format
``build_neighbor_graph.py`` writes and the gate ``train.py`` enforces.

    python3 -m automl.topo.build_vr_serial --arm serial --cutoff 4.0
    python3 -m automl.topo.build_vr_serial --arm orig   --cutoff 4.0
    python3 -m automl.topo.build_vr_serial --verify
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.topo.build_neighbor_graph import _edges_cutoff              # noqa: E402
from automl.topo.build_vr_conformers import donor_indices               # noqa: E402
from src.geometry_features import read_extxyz                           # noqa: E402

SHIPPED = _REPO / "data/processed/feature_blocks/vietoris_rips_inputs.npz"
SERIAL = _REPO / "automl/artifacts/serial_metals"
OUT_ROOT = _REPO / "automl/artifacts/vr_serial"
LANTH = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
         "Er", "Tm", "Yb", "Lu"}


def serial_records() -> pd.DataFrame:
    rows = []
    for p in (SERIAL / "records").glob("serial__*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:                                               # noqa: BLE001
            continue
        if r.get("ok") and r.get("path"):
            rows.append(r)
    return pd.DataFrame(rows)


def _core_cn(build_ids: list[str]) -> dict[str, int]:
    """coreCN per build id, from the dataset -- a COUNT rule input, not geometry."""
    df = pd.read_parquet(_REPO / "data/processed/final_ml_dataset_3d.parquet",
                         columns=["geometry_feature_build_id", "coreCN"])
    df = df.dropna().drop_duplicates("geometry_feature_build_id")
    return {str(r.geometry_feature_build_id): int(r.coreCN)
            for r in df.itertuples()}


def build(arm: str, cutoff: float = 4.0, verbose: bool = True) -> Path:
    recs = serial_records()
    if recs.empty:
        raise SystemExit("no serial records; run automl.qc.serial_metals first")
    cn = _core_cn(list(recs.build_id))
    z = np.load(SHIPPED)
    ship_ids = [str(b) for b in z["build_ids"].tolist()]
    ship_pos = {b: i for i, b in enumerate(ship_ids)}
    ship_ptr = z["node_ptr"]

    keep = recs[recs.build_id.isin(ship_pos)].drop_duplicates("build_id")
    # SHIPPED ORDER, not record order: build_row_table resolves rows through
    # index_of(build_id), and a reordered list silently repoints every row.
    keep = keep.assign(_o=keep.build_id.map(ship_pos)).sort_values("_o")

    co, zs, pc, im, idn, nptr, bids = [], [], [], [], [], [0], []
    e_idx, e_filt, e_ptr = [], [], [0]
    skipped = 0
    for r in keep.itertuples():
        bid = str(r.build_id)
        if arm == "serial":
            g = read_extxyz(_REPO / r.path)
            sym = [str(s) for s in g.symbols]
            xyz = np.asarray(g.coordinates, dtype=float)
            q = np.asarray(g.partial_charges, dtype=float)
        else:                                    # the matched control
            i = ship_pos[bid]
            a, b = int(ship_ptr[i]), int(ship_ptr[i + 1])
            xyz = z["coordinates"][a:b].astype(float)
            znum = z["atomic_numbers"][a:b]
            from src.geometry_features import LANTHANIDE_SYMBOLS  # noqa
            import ase.data as _ad                                # noqa
            sym = [_ad.chemical_symbols[int(v)] for v in znum]
            q = z["partial_charges"][a:b].astype(float)
        metal = next((s for s in sym if s in LANTH), None)
        if metal is None:
            skipped += 1
            continue
        mi, don = donor_indices(sym, xyz, metal, cn.get(bid, 9))
        if mi is None:
            skipped += 1
            continue
        import ase.data as _ad
        n = len(sym)
        co.append(xyz)
        zs.append(np.array([_ad.atomic_numbers.get(s, 0) for s in sym], dtype=np.int16))
        pc.append(q if len(q) == n else np.full(n, np.nan))
        m = np.zeros(n, dtype=np.int8); m[mi] = 1
        d = np.zeros(n, dtype=np.int8); d[list(don)] = 1
        im.append(m); idn.append(d)
        p, dd = _edges_cutoff(xyz, cutoff)
        e_idx.append(p + nptr[-1]); e_filt.append(dd)
        e_ptr.append(e_ptr[-1] + len(p))
        nptr.append(nptr[-1] + n)
        bids.append(bid)

    n_c = len(bids)
    payload = dict(
        coordinates=np.concatenate(co).astype(np.float32),
        atomic_numbers=np.concatenate(zs),
        partial_charges=np.concatenate(pc).astype(np.float32),
        is_metal=np.concatenate(im), is_coord_donor=np.concatenate(idn),
        node_ptr=np.asarray(nptr, dtype=np.int64),
        edge_index=np.concatenate(e_idx).T.astype(np.int64),
        edge_filtration=np.concatenate(e_filt).astype(np.float32),
        edge_ptr=np.asarray(e_ptr, dtype=np.int64),
        triangle_index=np.zeros((3, 0), np.int64),
        triangle_filtration=np.zeros(0, np.float32),
        triangle_ptr=np.zeros(n_c + 1, np.int64),
        build_ids=np.array(bids, dtype="<U32"))
    out = OUT_ROOT / arm
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "vietoris_rips_inputs.tmp.npz"
    np.savez_compressed(tmp, **payload)
    tmp.replace(out / "vietoris_rips_inputs.npz")
    meta = {"arm": arm, "cutoff": cutoff, "n_complexes": n_c,
            "n_nodes": int(payload["node_ptr"][-1]),
            "n_edges": int(payload["edge_index"].shape[1]),
            "skipped": skipped, "triangles": 0}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if verbose:
        print(f"[vr_serial] {arm}: {n_c} complexes, {meta['n_edges']:,} edges, "
              f"{skipped} skipped -> {out}")
    return out


def verify() -> int:
    """The two arms must be identical in everything except coordinates.

    If they differ in complex set, order, atom counts or element composition,
    then a serial-vs-orig contrast is comparing datasets and not geometries --
    which is the single way this experiment could produce a confident wrong
    answer.
    """
    a = np.load(OUT_ROOT / "serial/vietoris_rips_inputs.npz")
    b = np.load(OUT_ROOT / "orig/vietoris_rips_inputs.npz")
    ok = True
    for k in ("build_ids", "node_ptr", "atomic_numbers", "is_metal"):
        same = np.array_equal(a[k], b[k])
        print(f"  {k:18s} identical: {same}")
        ok &= bool(same)
    d = np.abs(a["coordinates"] - b["coordinates"]).max()
    print(f"  coordinates differ by max {d:.4f} A  (they MUST differ)")
    print(f"  edges: serial {a['edge_index'].shape[1]:,}  orig {b['edge_index'].shape[1]:,}")
    print("  VERDICT:", "matched" if ok and d > 0 else "NOT MATCHED")
    return 0 if (ok and d > 0) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("serial", "orig"))
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return verify()
    if args.both:
        build("serial", args.cutoff)
        build("orig", args.cutoff)
        return verify()
    if not args.arm:
        raise SystemExit("give --arm, --both or --verify")
    build(args.arm, args.cutoff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
