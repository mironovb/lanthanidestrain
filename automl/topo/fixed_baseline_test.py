#!/usr/bin/env python3
"""Does the topological model still beat the baseline once the baseline is fixed?

**Post-hoc. Not a pre-registered endpoint.**  The pre-registered primary
endpoint compares the SNN against a no-topology control built inside the
topological harness, and it is answered elsewhere (+0.0485 [+0.009, +0.106]).
This asks a different and blunter question: what happens to the *published*
comparison when the published baseline is repaired rather than replaced?

Why it needs asking
-------------------
The published FCNN scores +0.005 on adjacent pairs.  Changing one line of its
pipeline -- ``QuantileTransformer`` to ``StandardScaler`` -- and ensembling 16
seeds exactly as every published arm was, takes the same sklearn model to
+0.2206, against the SNN ensemble's +0.2382.  A gap of +0.018.

Quoted as point estimates that is meaningless: this study has shown repeatedly
that single numbers on this metric are unreadable (the baseline itself spans
0.11 across seed conventions).  So the gap gets the same paired cluster
bootstrap over extractants as every other interval here, and is reported with
the multiplicity correction alongside, since the uncorrected resampling was
measured to be 12-29 % too narrow.

If that interval spans zero, then a one-line change to the baseline erases the
headline comparison, and the paper's claim rests entirely on the pre-registered
control rather than on the published one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.adjacent_test import adj_r2
from automl.topo.compare_arms import attach_meta, collect
from automl.topo.control_factorial import (ensemble, load_cells,
                                           paired_adjacent_fast)

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
OUT = REPORTS / "fixed_baseline_test.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--variant", default="std_scaler_ens16")
    args = ap.parse_args()

    fixed_path = REPORTS / f"oof_fcnn_{args.variant}.parquet"
    if not fixed_path.exists():
        print(f"{fixed_path.name} not found -- run automl.topo.fcnn_diagnostic "
              f"--modes {args.variant} first")
        return 1
    fixed = attach_meta(pd.read_parquet(fixed_path)
                        .drop_duplicates("safe_exp_id").set_index("safe_exp_id"))

    members = load_cells(verbose=False)
    ens = {c: ensemble(m) for c, m in members.items()}
    ens["FCNN"] = attach_meta(collect().get("baseline::mlp::none"))

    def score(d):
        return adj_r2(d["y"].to_numpy(float), d["oof"].to_numpy(float),
                      d["composition_key"].to_numpy(),
                      d["lanthanide_index"].to_numpy())

    print(f"baseline variant: {args.variant}")
    print(f"  adjacent-pair R2 = {score(fixed):+.4f}   "
          f"overall R2 = {ev._r2(fixed['y'].to_numpy(float), fixed['oof'].to_numpy(float)):+.4f}")
    print(f"  (published FCNN  = {score(ens['FCNN']):+.4f})\n")

    rows = []
    for arm in ("S0", "P0", "T0w", "T0", "T1"):
        if ens.get(arm) is None or len(members.get(arm, {})) < 16:
            continue
        r = paired_adjacent_fast(fixed, ens[arm], args.n_boot, seed=0)
        if r is None:
            continue
        verdict = ("arm better" if r["lo"] > 0 else
                   "arm worse" if r["hi"] < 0 else "NOT DISTINGUISHABLE")
        rows.append({"arm": arm, "baseline": args.variant, **r,
                     "verdict": verdict})
        print(f"  {arm:4s} - fixed baseline   delta = {r['delta']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]   P = {r['p_better']:.2f}   "
              f"{verdict}")

    if not rows:
        print("no complete cells to compare")
        return 1
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n[fixed-baseline] wrote {OUT}")
    snn = next((r for r in rows if r["arm"] == "S0"), None)
    if snn:
        print("\nThe published headline was +0.2426 against the FCNN as shipped. "
              f"Against the same model with one line changed it is "
              f"{snn['delta']:+.4f} [{snn['lo']:+.4f}, {snn['hi']:+.4f}].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
