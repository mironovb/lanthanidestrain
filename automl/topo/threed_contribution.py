#!/usr/bin/env python3
"""How much does 3D structure add, at its best? Point estimates, no inference.

Every published 3D contrast in this study was measured under a stack fitted on
row-level log D.  STACK_FITTING_RESULTS showed that objective allocates weight
by level accuracy and gives the best 3D arm (D0) a weight of ZERO.  So the
published figures may understate what 3D structure is worth, simply because the
meta-learner never used it.

This measures the same contrast under both fitting objectives:

    non-3D pool          CatBoost + repaired      (fingerprints, descriptors)
    + 3D                 CatBoost + repaired + G0 + D0

Point estimates only.  Intervals and verdicts live in stack_test.csv,
encoder_test.csv and dualkey_test.csv; this answers "how large, at best".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/threed_contribution.csv"

NON3D = ("CatBoost", "repaired")
WITH3D = ("CatBoost", "repaired", "G0", "D0")
KEYS = ("composition_key", "strict_composition_key")


def _r2(dy, dp):
    tot = float(((dy - dy.mean()) ** 2).sum())
    return 1.0 - float(((dy - dp) ** 2).sum()) / tot if tot > 0 else float("nan")


def fit(frames, names, key, how: str):
    from automl.topo.pair_stack_probe import pair_vectors, _nnls_weights
    ref = frames[names[0]]
    y = ref["y"].to_numpy(float); g = ref["extractant_group"].to_numpy()
    if how == "row":
        R = np.column_stack([frames[n]["oof"].to_numpy(float) for n in names])
        pred = np.zeros_like(y)
        for gt in pd.unique(g):
            te = g == gt; tr = ~te
            pred[te] = (R[te] @ _nnls_weights(R[tr], y[tr])) if tr.sum() >= 20 \
                else (y[tr].mean() if tr.sum() else 0.0)
        bl = ref.copy(); bl["oof"] = pred
        dy, dp, _ = pair_vectors(bl, key=key)
        return _r2(dy, dp)
    dy, _, gp = pair_vectors(ref, key=key)
    A = np.column_stack([pair_vectors(frames[n], key=key)[1] for n in names])
    dp = np.zeros_like(dy)
    for gt in pd.unique(gp):
        te = gp == gt; tr = ~te
        dp[te] = (A[te] @ _nnls_weights(A[tr], dy[tr])) if tr.sum() >= 20 \
            else (dy[tr].mean() if tr.sum() else 0.0)
    return _r2(dy, dp)


def main() -> int:
    from automl.topo.full_stack import collect_arms
    arms = collect_arms(verbose=False)
    need = sorted(set(NON3D) | set(WITH3D))
    idx = None
    for n in need:
        idx = arms[n].index if idx is None else idx.intersection(arms[n].index)
    F = {n: arms[n].loc[idx] for n in need}

    rows = []
    print("adjacent-pair R2 of a stack, with and without the 3D arms\n")
    print(f"  {'fit':10s} {'key':8s} {'no 3D':>9s} {'+ 3D':>9s} {'3D adds':>9s}")
    for how in ("row", "pair"):
        for key in KEYS:
            tag = "binned" if key == KEYS[0] else "strict"
            a = fit(F, list(NON3D), key, how)
            b = fit(F, list(WITH3D), key, how)
            print(f"  {how+'-fitted':10s} {tag:8s} {a:+9.4f} {b:+9.4f} {b-a:+9.4f}")
            rows.append(dict(fit=how, key=key, without_3d=a, with_3d=b,
                             delta_3d=b - a))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    best = d.loc[d["delta_3d"].idxmax()]
    print(f"\n  largest 3D contribution: {best['delta_3d']:+.4f} "
          f"({best['fit']}-fitted, {'binned' if best['key']==KEYS[0] else 'strict'}: "
          f"{best['without_3d']:+.4f} -> {best['with_3d']:+.4f})")
    r = d[d.fit == "row"]["delta_3d"].max()
    p = d[d.fit == "pair"]["delta_3d"].max()
    print(f"  best under row-fitting {r:+.4f}   best under pair-fitting {p:+.4f}"
          f"   difference {p-r:+.4f}")
    print("\n  Point estimates only. Intervals and verdicts are in stack_test.csv,")
    print("  encoder_test.csv and dualkey_test.csv.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
