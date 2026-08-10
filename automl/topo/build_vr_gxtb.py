#!/usr/bin/env python3
"""Trainable assets from the g-xTB-relaxed geometries, plus their matched control.

Two arms over the SAME complexes, identical in every respect but where the
atoms are:

    gxtb    the g-xTB minimum, reached from the shipped coordinates
    ship    the shipped coordinates themselves (what models get today)

This mirrors ``build_vr_serial`` exactly, and for the same reason: if the arms
differ in complex set, order, atom count or composition, the contrast compares
datasets rather than geometries, which is the one way this experiment could
produce a confident wrong answer.

The gate that decides whether the contrast means anything
---------------------------------------------------------
The g-xTB minima sit a **median 1.24 A** from the shipped ones.  That is large.
It is the difference between two readings:

  * the metal-ligand physics is genuinely different -- bond lengths and donor
    geometry corrected by a Hamiltonian that gets the lanthanide contraction
    right (slope 1.14 vs GFN2's 0.39 against Shannon radii); or
  * the ligand fell into a different conformer basin, in which case the arms
    differ by conformer noise and we would be re-running the confound that made
    the original set uninformative in the first place -- adjacent-lanthanide
    pairs there differ by 5.46 A median RMSD, ~99 % of it conformer sampling.

``basin_report`` separates them.  A structure keeps its identity if the donor
SET is unchanged (same atom indices coordinating the metal) and the coordination
number is unchanged; then the M-donor distances can move as much as the physics
requires without the comparison being contaminated.  Complexes that hop are
counted, reported, and -- with ``--drop-hops`` -- excluded from BOTH arms
together, never from one.

    python3 -m automl.topo.build_vr_gxtb --both --cutoff 4.0
    python3 -m automl.topo.build_vr_gxtb --basin-report
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

from automl.topo.build_neighbor_graph import _edges_cutoff        # noqa: E402
from automl.topo.build_vr_conformers import donor_indices         # noqa: E402

SHIPPED = _REPO / "data/processed/feature_blocks/vietoris_rips_inputs.npz"
REOPT = _REPO / "automl/artifacts/gxtb_reopt/records"
OUT_ROOT = _REPO / "automl/artifacts/vr_gxtb"
LANTH = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
         "Er", "Tm", "Yb", "Lu"}


def load_reopt() -> dict[str, dict]:
    out = {}
    for p in REOPT.glob("gxtb__*.json"):
        try:
            r = json.loads(p.read_text())
        except Exception:                                          # noqa: BLE001
            continue
        if r.get("ok") and r.get("coords"):
            out[str(r["build_id"])] = r
    return out


def _core_cn() -> dict[str, int]:
    import pandas as pd
    df = pd.read_parquet(_REPO / "data/processed/final_ml_dataset_3d.parquet",
                         columns=["geometry_feature_build_id", "coreCN"])
    df = df.dropna().drop_duplicates("geometry_feature_build_id")
    return {str(r.geometry_feature_build_id): int(r.coreCN)
            for r in df.itertuples()}


def _shipped_index():
    import ase.data as ad
    z = np.load(SHIPPED)
    ids = [str(b) for b in z["build_ids"].tolist()]
    pos = {b: i for i, b in enumerate(ids)}
    ptr = z["node_ptr"]

    def get(bid):
        i = pos[bid]
        a, b = int(ptr[i]), int(ptr[i + 1])
        sym = [ad.chemical_symbols[int(v)] for v in z["atomic_numbers"][a:b]]
        return (sym, z["coordinates"][a:b].astype(float),
                z["partial_charges"][a:b].astype(float))
    return z, ids, pos, get


def basin_report(verbose: bool = True) -> dict:
    """Did the ligand stay in its basin, or fold into a different one?"""
    ro = load_reopt()
    cn_map = _core_cn()
    _, _, pos, get = _shipped_index()
    kept, hops, rows = [], [], []
    for bid, r in sorted(ro.items()):
        if bid not in pos:
            continue
        sym_s, xyz_s, _ = get(bid)
        xyz_g = np.asarray(r["coords"], dtype=float)
        metal = next((s for s in sym_s if s in LANTH), None)
        if metal is None or xyz_g.shape != xyz_s.shape:
            continue
        cn = cn_map.get(bid, 9)
        mi_s, don_s = donor_indices(sym_s, xyz_s, metal, cn)
        mi_g, don_g = donor_indices(sym_s, xyz_g, metal, cn)
        if mi_s is None or mi_g is None:
            continue
        same = set(don_s) == set(don_g)
        rmsd = float(np.sqrt(((xyz_g - xyz_s) ** 2).sum(axis=1).mean()))
        d_s = float(np.mean(np.linalg.norm(xyz_s[list(don_s)] - xyz_s[mi_s], axis=1)))
        d_g = float(np.mean(np.linalg.norm(xyz_g[list(don_g)] - xyz_g[mi_g], axis=1)))
        rows.append({"build_id": bid, "same_donor_set": same, "rmsd": rmsd,
                     "n_shared": len(set(don_s) & set(don_g)), "cn": cn,
                     "mean_md_shipped": d_s, "mean_md_gxtb": d_g,
                     "d_mean_md": d_g - d_s})
        (kept if same else hops).append(bid)
    n = len(rows)
    rep = {"n": n, "kept": len(kept), "hopped": len(hops),
           "hop_rate": len(hops) / n if n else float("nan"),
           "hops": hops}
    if verbose and n:
        r = np.array([x["rmsd"] for x in rows])
        dk = np.array([x["d_mean_md"] for x in rows if x["same_donor_set"]])
        share = np.array([x["n_shared"] / x["cn"] for x in rows])
        print(f"[basin] {n} complexes compared")
        print(f"  donor set preserved : {len(kept)}/{n} = {len(kept)/n:.1%}")
        print(f"  donor overlap frac  : median {np.median(share):.3f}")
        print(f"  heavy-atom RMSD     : median {np.median(r):.3f} A "
              f"p90 {np.percentile(r,90):.3f}")
        if len(dk):
            print(f"  mean M-donor shift (basin-preserving only): "
                  f"{np.median(dk)*1000:+.1f} mA  (p10 {np.percentile(dk,10)*1000:+.0f}, "
                  f"p90 {np.percentile(dk,90)*1000:+.0f})")
        print("\n  READING: a large RMSD with the donor set preserved is the")
        print("  Hamiltonian moving the structure. A large RMSD with the donor")
        print("  set CHANGED is a conformer hop and is not evidence of anything.")
    rep["rows"] = rows
    return rep


def eligible(drop_hops: bool = True, verbose: bool = True) -> list[str]:
    """The complex set BOTH arms will use, resolved once.

    Two ways the arms drift apart if each build decides for itself, and the
    first build attempt hit both:

    * the re-optimisation is still writing records, so a second ``build()``
      call sees complexes the first did not -- the arms then differ by whatever
      landed in between;
    * ``donor_indices`` runs on each arm's own coordinates and can succeed for
      one arm and fail for the other, silently dropping a complex from one side.

    Either produces a contrast between two different datasets rather than two
    geometries, which is the single way this experiment could return a
    confident wrong answer.  So the set is fixed here, from a single snapshot,
    and requires the donor shell to resolve under BOTH coordinate sets.
    """
    ro = load_reopt()
    cn_map = _core_cn()
    _, ship_ids, pos, get = _shipped_index()
    keep, no_donor = [], 0
    for bid in ship_ids:                            # SHIPPED order, always
        if bid not in ro:
            continue
        sym, xyz_s, _ = get(bid)
        xyz_g = np.asarray(ro[bid]["coords"], dtype=float)
        if xyz_g.shape != xyz_s.shape:
            continue
        metal = next((s for s in sym if s in LANTH), None)
        if metal is None:
            continue
        cn = cn_map.get(bid, 9)
        mi_s, _ = donor_indices(sym, xyz_s, metal, cn)
        mi_g, _ = donor_indices(sym, xyz_g, metal, cn)
        if mi_s is None or mi_g is None:
            no_donor += 1
            continue
        keep.append(bid)
    if drop_hops:
        hops = set(basin_report(verbose=False)["hops"])
        before = len(keep)
        keep = [b for b in keep if b not in hops]
        if verbose:
            print(f"[eligible] {before} resolvable in both arms, "
                  f"{before - len(keep)} basin hops dropped from BOTH, "
                  f"{no_donor} donor-shell failures -> {len(keep)}")
    elif verbose:
        print(f"[eligible] {len(keep)} complexes ({no_donor} donor failures)")
    return keep


def build(arm: str, cutoff: float = 4.0, drop_hops: bool = True,
          verbose: bool = True, keep: list[str] | None = None) -> Path:
    import ase.data as ad
    ro = load_reopt()
    cn_map = _core_cn()
    z, ship_ids, pos, get = _shipped_index()
    if keep is None:
        keep = eligible(drop_hops, verbose=verbose)
    keep = [b for b in keep if b in ro]

    co, zs, pc, im, idn, nptr, bids = [], [], [], [], [], [0], []
    e_idx, e_filt, e_ptr = [], [], [0]
    skipped = 0
    for bid in keep:
        sym, xyz_s, q = get(bid)
        xyz = (np.asarray(ro[bid]["coords"], dtype=float) if arm == "gxtb"
               else xyz_s)
        if xyz.shape != xyz_s.shape:
            skipped += 1
            continue
        metal = next((s for s in sym if s in LANTH), None)
        if metal is None:
            skipped += 1
            continue
        mi, don = donor_indices(sym, xyz, metal, cn_map.get(bid, 9))
        if mi is None:
            skipped += 1
            continue
        n = len(sym)
        co.append(xyz)
        zs.append(np.array([ad.atomic_numbers.get(s, 0) for s in sym], np.int16))
        pc.append(q if len(q) == n else np.full(n, np.nan))
        m = np.zeros(n, np.int8); m[mi] = 1
        dd = np.zeros(n, np.int8); dd[list(don)] = 1
        im.append(m); idn.append(dd)
        p, f = _edges_cutoff(xyz, cutoff)
        e_idx.append(p + nptr[-1]); e_filt.append(f)
        e_ptr.append(e_ptr[-1] + len(p))
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
    out = OUT_ROOT / arm
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "vietoris_rips_inputs.tmp.npz"
    np.savez_compressed(tmp, **payload)
    tmp.replace(out / "vietoris_rips_inputs.npz")
    (out / "meta.json").write_text(json.dumps(
        {"arm": arm, "cutoff": cutoff, "n_complexes": n_c,
         "n_nodes": int(payload["node_ptr"][-1]),
         "n_edges": int(payload["edge_index"].shape[1]),
         "skipped": skipped, "drop_hops": drop_hops, "triangles": 0},
        indent=2) + "\n")
    if verbose:
        print(f"[vr_gxtb] {arm}: {n_c} complexes, "
              f"{payload['edge_index'].shape[1]:,} edges -> {out}")
    return out


def verify() -> int:
    a = np.load(OUT_ROOT / "gxtb/vietoris_rips_inputs.npz")
    b = np.load(OUT_ROOT / "ship/vietoris_rips_inputs.npz")
    ok = True
    for k in ("build_ids", "node_ptr", "atomic_numbers", "is_metal"):
        same = np.array_equal(a[k], b[k])
        print(f"  {k:18s} identical: {same}")
        ok &= bool(same)
    d = np.abs(a["coordinates"] - b["coordinates"]).max()
    print(f"  coordinates differ by max {d:.4f} A  (they MUST differ)")
    print(f"  edges: gxtb {a['edge_index'].shape[1]:,}  ship {b['edge_index'].shape[1]:,}")
    print("  VERDICT:", "matched" if ok and d > 0 else "NOT MATCHED")
    return 0 if (ok and d > 0) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("gxtb", "ship"))
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--basin-report", action="store_true")
    ap.add_argument("--keep-hops", action="store_true")
    args = ap.parse_args()
    if args.basin_report:
        basin_report()
        return 0
    if args.verify:
        return verify()
    drop = not args.keep_hops
    if args.both:
        shared = eligible(drop)                     # ONE snapshot, both arms
        build("gxtb", args.cutoff, drop, keep=list(shared))
        build("ship", args.cutoff, drop, keep=list(shared))
        return verify()
    if not args.arm:
        raise SystemExit("give --arm, --both, --verify or --basin-report")
    build(args.arm, args.cutoff, drop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
