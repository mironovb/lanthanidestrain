#!/usr/bin/env python3
"""Does topology add to the *strongest* baseline in a stack?

Pre-registered in ``automl/reports/STACK_PREREGISTRATION.md``.  No new training:
every out-of-fold vector already exists, so nothing here can be tuned by
re-running an arm.

The three contrasts, and why the third is the one that matters:

    1 (primary)  blend(S0, repaired) - repaired
    2 (control)  blend(T0w, repaired) - repaired
    3 (decisive) blend(S0, repaired) - blend(T0w, repaired)

Contrast 1 alone can be satisfied by *any* model with decorrelated errors -- the
earlier blend analysis found exactly that trap, where an "interior maximum"
credited to topology reproduced, larger, for a plain tabular MLP.  T0w is the
matched tabular control (same harness, folds, seeds and objective as S0, encoder
removed), so contrast 3 isolates the encoder.

Blend weights come from nested leave-one-extractant-out selection: for each
extractant the weight is chosen on the others only, so no row influences the
weight it is scored under.
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
OUT = REPORTS / "stack_test.csv"
GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)


def nested_blend(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Blend ``a`` and ``b`` with a per-extractant weight fitted on the others.

    ``w`` weights ``b``; the returned frame carries the blended oof so it can go
    straight into the same paired bootstrap every other number in this study
    used.
    """
    idx = a.index.intersection(b.index)
    A, B = a.loc[idx], b.loc[idx]
    y = A["y"].to_numpy(float)
    pa, pb = A["oof"].to_numpy(float), B["oof"].to_numpy(float)
    comp = A["composition_key"].to_numpy()
    li = A["lanthanide_index"].to_numpy()
    g = A["extractant_group"].to_numpy()

    out = np.empty(len(A), dtype=float)
    chosen = []
    for grp in pd.unique(g):
        te = g == grp
        tr = ~te
        best_w, best_v = 0.0, -np.inf
        for w in GRID:
            v = adj_r2(y[tr], (1 - w) * pa[tr] + w * pb[tr], comp[tr], li[tr])
            if np.isfinite(v) and v > best_v:
                best_v, best_w = v, w
        out[te] = (1 - best_w) * pa[te] + best_w * pb[te]
        chosen.append(best_w)
    frame = A.copy()
    frame["oof"] = out
    return frame, chosen


def _score(d: pd.DataFrame) -> tuple[float, float]:
    y = d["y"].to_numpy(float); p = d["oof"].to_numpy(float)
    return (adj_r2(y, p, d["composition_key"].to_numpy(),
                   d["lanthanide_index"].to_numpy()), ev._r2(y, p))


def _corrected(delta, lo, hi, n_tests=3):
    """Bonferroni-style interval from the reported 90 % quantiles.

    An approximation, and said to be one: se from the 90 % width, then a
    z for 1 - 0.05/n_tests one-sided.
    """
    from scipy import stats
    se = (hi - lo) / (2 * 1.645)
    z = stats.norm.ppf(1 - 0.05 / n_tests)
    return delta - z * se, delta + z * se


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    cells = load_cells(verbose=False)
    ens = {c: ensemble(m) for c, m in cells.items()}
    fixed = attach_meta(
        pd.read_parquet(REPORTS / "oof_fcnn_std_scaler_ens16.parquet")
        .drop_duplicates("safe_exp_id").set_index("safe_exp_id"))
    cat = attach_meta(collect()["baseline::catboost::none"])

    arms = {"S0": ens.get("S0"), "T0w": ens.get("T0w"), "repaired": fixed,
            "CatBoost": cat}
    # S2, if its 32 seeds are present, as a descriptive secondary.
    from automl.topo.s2_test import load_s2
    s2m = load_s2()
    if len(s2m) >= 30:
        arms["S2"] = ensemble(s2m)

    print("=== arms (adjacent-pair R2 | overall R2) ===")
    for k, v in arms.items():
        if v is None:
            continue
        a, r = _score(v)
        print(f"  {k:10s} n={len(v):5d}  adjR2={a:+.4f}  R2={r:+.4f}")

    print("\n=== nested blends with the repaired baseline ===")
    blends = {}
    for k in ("S0", "T0w", "S2", "CatBoost"):
        if arms.get(k) is None:
            continue
        bl, ws = nested_blend(arms["repaired"], arms[k])
        blends[k] = bl
        a, r = _score(bl)
        print(f"  blend(repaired, {k:8s}) adjR2={a:+.4f}  R2={r:+.4f}  "
              f"median w({k})={np.median(ws):.2f} "
              f"IQR[{np.percentile(ws,25):.2f},{np.percentile(ws,75):.2f}]")

    print("\n=== pre-registered contrasts ===")
    rows = []
    tests = [
        ("1_primary", arms["repaired"], blends.get("S0"),
         "blend(S0, repaired) - repaired : does topology add to the strongest baseline?"),
        ("2_control", arms["repaired"], blends.get("T0w"),
         "blend(T0w, repaired) - repaired : does ANY second neural model add?"),
        ("3_decisive", blends.get("T0w"), blends.get("S0"),
         "blend(S0) - blend(T0w) : does topology add SPECIFICALLY?"),
    ]
    for name, base, arm, q in tests:
        if base is None or arm is None:
            continue
        r = paired_adjacent_fast(base, arm, args.n_boot, seed=0)
        if r is None:
            print(f"  {name}: not comparable")
            continue
        clo, chi = _corrected(r["delta"], r["lo"], r["hi"], 3)
        v = ("ADDS" if r["lo"] > 0 else "worse" if r["hi"] < 0
             else "not distinguishable")
        cv = ("ADDS" if clo > 0 else "worse" if chi < 0
              else "not distinguishable")
        star = "**" if name != "2_control" else "  "
        print(f"{star}{name:11s} delta={r['delta']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}] P={r['p_better']:.2f}  {v}")
        print(f"     3-test corrected [{clo:+.4f}, {chi:+.4f}]  {cv}")
        print(f"     | {q}")
        rows.append({"contrast": name, "question": q, **r,
                     "lo_3test": clo, "hi_3test": chi,
                     "verdict": v, "verdict_3test": cv})

    # Descriptive: same three with S2, and the full stack.
    if blends.get("S2") is not None:
        r = paired_adjacent_fast(arms["repaired"], blends["S2"], args.n_boot, 0)
        if r:
            print(f"\n  [descriptive] blend(S2, repaired) - repaired "
                  f"delta={r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]")
            rows.append({"contrast": "desc_S2", "question": "S2 blend",
                         **r, "verdict": "descriptive"})

    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"\n[stack] wrote {OUT}")
    print("\nContrast 3 is the claim. Contrast 1 positive with 3 spanning zero "
          "means generic ensembling, not topology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
