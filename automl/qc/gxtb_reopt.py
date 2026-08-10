#!/usr/bin/env python3
"""Re-optimise the real dataset complexes with g-xTB, so the modelling question
is answered directly instead of by proxy.

Everything so far has been measured on substituted *anchors*: 6 ligands, then
71.  Those establish that g-xTB reproduces the lanthanide contraction (slope
1.142 against Shannon radii, versus GFN2's 0.386 -- a 2.5x underestimate that
holds in gas and in solvent) and that GFN2's per-ligand response is mostly
noise (23 % shared across ligands, against g-xTB's 96 %).

None of that is a score.  This module produces the one asset that can give a
score: the **same 956 complexes the models already train on**, relaxed under
g-xTB instead of GFN2, so a paired comparison changes the Hamiltonian and
nothing else.

Design, following the serial-vs-orig contrast that just ran
-----------------------------------------------------------
The starting coordinates are the **shipped** ones -- the coordinates the models
are currently given.  So the contrast is exactly "the geometry we have" versus
"the geometry a better Hamiltonian relaxes it into", on an identical complex
set, in identical order, with identical build ids.  ``build_vr_serial.verify``
already enforces that discipline and the same check applies here.

Charge is taken from the shipped per-atom partial charges (they sum to the
molecular charge by construction, which is what ``xtb_backend.infer_charge``
relies on) and cross-checked against the integer it rounds to; anything that
does not round cleanly is rejected rather than guessed, because a wrong charge
on a +3 lanthanide silently produces a different chemistry rather than an error.

``uhf`` is the Hund high-spin count for Ln(III).  Under GFN2 f is in the core
and uhf 0 is right; under g-xTB f is in the valence and Gd(III) is f7 with
seven unpaired electrons.  Getting this wrong would not error -- it would
converge to an unphysical state and quietly weaken the very effect being tested.

Gas phase.  g-xTB has no ALPB parameters, and its ddCOSMO arm failed 14-23 % of
optimisations in the pilot with the failures **concentrated on particular
metals** (Pm 10, Pr 5, Dy 4, Yb 4; La/Ce/Sm/Tm/Lu clean).  Non-random
missingness correlated with the metal is fatal to a selectivity study, because
the adjacent pairs involving the hard metals drop out systematically.  The
gas-phase arm had zero failures in 270.

    python3 -m automl.qc.gxtb_reopt --shard 0 --num-shards 4 --workers 48
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.qc.gxtb_probe import (  # noqa: E402
    Z_OF, high_spin_uhf, optimize_with_retry,
)

SHIPPED = _REPO / "data/processed/feature_blocks/vietoris_rips_inputs.npz"
OUT = _REPO / "automl/artifacts/gxtb_reopt"


def load_complexes() -> list[dict[str, Any]]:
    import ase.data as ad
    z = np.load(SHIPPED)
    ids = [str(b) for b in z["build_ids"].tolist()]
    ptr = z["node_ptr"]
    out = []
    for i, bid in enumerate(ids):
        a, b = int(ptr[i]), int(ptr[i + 1])
        sym = [ad.chemical_symbols[int(v)] for v in z["atomic_numbers"][a:b]]
        q = z["partial_charges"][a:b].astype(float)
        # Some complexes carry no Mulliken charges at all (the shipped asset's
        # documented "charge missing" marker).  Their charge cannot be inferred
        # from the sum, and every lanthanide in this dataset is Ln(III) with a
        # +3 complex charge -- but assuming that here would be guessing on the
        # exact rows most likely to be irregular, so they are flagged and the
        # caller decides.
        finite = bool(np.all(np.isfinite(q)))
        tot = float(np.sum(q)) if finite else float("nan")
        chg = int(round(tot)) if finite else None
        metal = next((s for s in sym if s in Z_OF), None)
        out.append({
            "build_id": bid, "index": i, "symbols": sym,
            "coords": z["coordinates"][a:b].astype(float).tolist(),
            "charge": chg,
            "charge_residual": abs(tot - chg) if finite else float("nan"),
            "charges_present": finite,
            "metal": metal, "n_atoms": len(sym),
        })
    return out


def _one(t: dict[str, Any]) -> dict[str, Any]:
    base = {k: t[k] for k in ("build_id", "index", "metal", "charge",
                              "n_atoms", "charge_residual", "charges_present")}
    if t["metal"] is None:
        return {**base, "ok": False, "reason": "NO_LANTHANIDE"}
    if not t["charges_present"]:
        return {**base, "ok": False, "reason": "CHARGE_MISSING"}
    if t["charge_residual"] > 0.05:
        # Do not guess.  A +3 lanthanide run at the wrong charge converges
        # happily to different chemistry instead of failing.
        return {**base, "ok": False, "reason": "CHARGE_NOT_INTEGRAL"}
    uhf = high_spin_uhf(t["metal"])
    r = optimize_with_retry(t["symbols"], np.asarray(t["coords"]),
                            charge=t["charge"], uhf=uhf, method="gxtb",
                            solvent=None, threads=1,
                            timeout=t.get("timeout", 14400))
    rec = {**base, "uhf": uhf, "ok": bool(r.get("ok")),
           "reason": r.get("reason"), "seconds": r.get("seconds"),
           "needed_etemp": r.get("needed_etemp"),
           "energy_eh": r.get("energy_eh"),
           "homo_lumo_gap_ev": r.get("homo_lumo_gap_ev"),
           "converged": r.get("converged"),
           "rmsd_from_shipped_ang": r.get("rmsd_from_input_ang")}
    if r.get("ok"):
        rec["coords"] = r["coords"]
        rec["symbols"] = r["symbols"]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--max-atoms", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cx = load_complexes()
    if args.max_atoms:
        cx = [c for c in cx if c["n_atoms"] <= args.max_atoms]
    if args.limit:
        cx = cx[:args.limit]
    # Shard round-robin on the SHIPPED index so shards are size-balanced and
    # every shard's output can be reassembled by build_id without ordering
    # assumptions.
    mine = [c for c in cx if c["index"] % args.num_shards == args.shard]
    mine.sort(key=lambda c: c["n_atoms"])          # cheapest first
    for c in mine:
        c["timeout"] = args.timeout
    print(f"[gxtb_reopt] shard {args.shard}/{args.num_shards}: {len(mine)} of "
          f"{len(cx)} complexes, {sum(c['n_atoms'] for c in mine):,} atoms",
          flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    recdir = OUT / "records"
    recdir.mkdir(exist_ok=True)
    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, t): t for t in mine}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:                                # noqa: BLE001
                t = futs[fu]
                r = {"build_id": t["build_id"], "ok": False,
                     "reason": f"exception:{e}"}
            n_ok += bool(r.get("ok"))
            # One atomic file per complex: a job that dies at hour nine keeps
            # everything it finished, and reruns are idempotent.
            p = recdir / f"gxtb__{r['build_id']}.json"
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(r, default=float) + "\n")
            tmp.replace(p)
            if i % 10 == 0 or i == len(mine):
                print(f"  [{i}/{len(mine)}] ok={n_ok}", flush=True)
    print(f"[gxtb_reopt] shard {args.shard}: {n_ok}/{len(mine)} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
