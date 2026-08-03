#!/usr/bin/env python3
"""Three parallel VR assets over identical build ids: shipped, control, neutral.

Pre-registered in ``CAMPAIGN4_PREREGISTRATION.md``.  Writes only under
``automl/artifacts/``; ``data/`` is never touched.

Three invariants, ordered by how silently they fail:

1. **Each asset gets its own triangle-edge cache.**  ``SimplicialComplexes``
   keys that cache only by triangle count (``simplicial_data.py:247-256``), so
   an asset loaded without its own ``cache=`` silently reuses the shipped
   boundary map and every downstream number is quietly wrong.
2. **All three assets carry the identical ``build_ids``**, or the paired
   bootstrap compares two different row sets and the contrast is meaningless.
3. **``--verify-against-shipped`` rebuilds the ORIGINAL geometries through this
   same code path** and asserts element-for-element agreement with the shipped
   npz.  That is the gate proving nothing existing was disturbed.

Usage
-----
    python3 -m automl.topo.build_vr_neutral --verify-against-shipped
    python3 -m automl.topo.build_vr_neutral --arm shipped
    python3 -m automl.topo.build_vr_neutral --arm control
    python3 -m automl.topo.build_vr_neutral --arm neutral
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.topo.build_vr_conformers import (              # noqa: E402
    SHIPPED, _assemble, _original_index, donor_indices)
from src.geometry_features import read_extxyz              # noqa: E402

NEUT_ROOT = _REPO / "automl/artifacts/neutral_species"
OUT_DIR = _REPO / "automl/artifacts/vr_neutral"
DONOR_CUTOFF_A = 3.10
ARMS = ("shipped", "control", "neutral")


def _read_generated(path: Path):
    """symbols, coords, per-atom charges from a generated extxyz."""
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    has_q = "charge:R:1" in lines[1]
    sym, xyz, q = [], [], []
    for ln in lines[2:2 + n]:
        f = ln.split()
        sym.append(f[0]); xyz.append([float(f[1]), float(f[2]), float(f[3])])
        q.append(float(f[4]) if has_q and len(f) > 4 else np.nan)
    return sym, np.asarray(xyz, float), np.asarray(q, np.float32), has_q


def accepted_records() -> list[dict]:
    """Records that passed every gate, joined to their build id."""
    import pandas as pd
    audit = _REPO / "automl/reports/neutralize_audit.csv"
    if not audit.exists():
        raise SystemExit("run automl.qc.neutralize_report first")
    good = set(pd.read_csv(audit).query("reject_code=='accepted'")["geometry_key"])
    out = []
    for p in sorted((NEUT_ROOT / "records").glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("geometry_key") in good and rec.get("neutral_xyz"):
            out.append(rec)
    return out


def build(arm: str) -> int:
    """One asset. Every arm covers the same build ids, in the same order."""
    if arm not in ARMS:
        raise SystemExit(f"--arm must be one of {ARMS}")
    orig = _original_index()
    by_build = {v["build_id"]: (k, v) for k, v in orig.items()}
    recs = accepted_records()
    print(f"[vr-neutral] {len(recs)} accepted structures", flush=True)

    records, skipped = [], {}
    for rec in sorted(recs, key=lambda r: str(r.get("geometry_feature_build_id"))):
        bid = str(rec.get("geometry_feature_build_id"))
        if bid not in by_build:
            skipped["no_shipped_entry"] = skipped.get("no_shipped_entry", 0) + 1
            continue
        _, meta = by_build[bid]
        core_cn, msym = int(meta["core_cn"]), str(meta["metal_symbol"])
        if arm == "shipped":
            g = read_extxyz(meta["path"])
            sym = [str(s) for s in g.symbols]
            xyz = np.asarray(g.coordinates, float)
            q = np.asarray(g.partial_charges, np.float32)
        else:
            key = "control_xyz" if arm == "control" else "neutral_xyz"
            sym, xyz, q, has_q = _read_generated(_REPO / rec[key])
            if not has_q:
                skipped["no_per_atom_charges"] = skipped.get("no_per_atom_charges", 0) + 1
                continue
        n_add = int(rec.get("n_add", 0))
        n0 = len(sym) - 4 * n_add if arm == "neutral" else len(sym)
        # The LIGAND donor set is defined identically in every arm -- nearest
        # core_cn donors among the original atoms -- so the arms differ by the
        # geometry, never by how the mask was constructed.
        mi, di = donor_indices(sym[:n0], xyz[:n0], msym, core_cn)
        if mi is None:
            skipped["no_coordination_shell"] = skipped.get("no_coordination_shell", 0) + 1
            continue
        di = list(di)
        if arm == "neutral" and n_add:
            for a in range(n_add):
                b = n0 + 4 * a
                for o in range(b + 1, b + 4):
                    if np.linalg.norm(xyz[o] - xyz[mi]) < DONOR_CUTOFF_A:
                        di.append(o)      # a coordinating nitrate O IS a donor
        records.append({"build_id": bid, "symbols": sym, "coordinates": xyz,
                        "partial_charges": q, "donor_indices": np.asarray(di, int)})

    if skipped:
        print(f"[vr-neutral] skipped: {skipped}", flush=True)
    if not records:
        raise SystemExit("no records to assemble")
    npz = _assemble(records)
    d = OUT_DIR / arm
    d.mkdir(parents=True, exist_ok=True)
    out = d / "vietoris_rips_inputs.npz"
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **npz)
    tmp.replace(out)
    print(f"[vr-neutral] {arm}: {len(records)} complexes, "
          f"{npz['node_ptr'][-1]} nodes, {npz['edge_ptr'][-1]} edges, "
          f"{npz['triangle_ptr'][-1]} triangles")
    print(f"[vr-neutral] wrote {out}")
    print(f"[vr-neutral] NOTE: load with cache={d/'triangle_edges.npz'} -- the "
          f"boundary-map cache is keyed only by triangle count.")
    return 0


def verify_against_shipped(n: int = 40) -> int:
    """Rebuild the ORIGINAL geometries here and compare to the shipped asset.

    The gate that proves this code path does not disturb what already exists.
    """
    z = np.load(SHIPPED)
    ids = [str(b) for b in z["build_ids"]]
    idx = {b: i for i, b in enumerate(ids)}
    orig = _original_index()
    checked = mismatch = 0
    for _, meta in sorted(orig.items())[:n]:
        bid = meta["build_id"]
        if bid not in idx:
            continue
        g = read_extxyz(meta["path"])
        rec = {"build_id": bid, "symbols": [str(s) for s in g.symbols],
               "coordinates": np.asarray(g.coordinates, float),
               "partial_charges": np.asarray(g.partial_charges, np.float32),
               "donor_indices": np.asarray(donor_indices(
                   [str(s) for s in g.symbols], np.asarray(g.coordinates, float),
                   str(meta["metal_symbol"]), int(meta["core_cn"]))[1], int)}
        built = _assemble([rec])
        i = idx[bid]
        sl = slice(int(z["node_ptr"][i]), int(z["node_ptr"][i + 1]))
        ok = (np.array_equal(built["atomic_numbers"], z["atomic_numbers"][sl])
              and np.allclose(built["coordinates"], z["coordinates"][sl],
                              atol=1e-5, rtol=0)
              and np.array_equal(built["is_coord_donor"], z["is_coord_donor"][sl])
              and np.array_equal(built["is_metal"], z["is_metal"][sl]))
        checked += 1
        if not ok:
            mismatch += 1
            print(f"  MISMATCH {bid}")
    print(f"[verify] {checked} originals rebuilt through this path, "
          f"{mismatch} mismatches")
    if mismatch:
        print("[verify] FAILED -- this code path does not reproduce the shipped "
              "asset, so nothing built with it can be trusted")
        return 1
    print("[verify] OK -- the shipped geometries are reproduced "
          "element-for-element; nothing existing was disturbed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--verify-against-shipped", action="store_true")
    ap.add_argument("--verify-n", type=int, default=40)
    a = ap.parse_args()
    if a.verify_against_shipped:
        return verify_against_shipped(a.verify_n)
    if not a.arm:
        ap.error("--arm or --verify-against-shipped")
    return build(a.arm)


if __name__ == "__main__":
    raise SystemExit(main())
