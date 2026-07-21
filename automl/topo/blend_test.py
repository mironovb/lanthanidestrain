#!/usr/bin/env python3
"""Does topology carry adjacent-pair information the tabular model lacks?

The seed-ensembled persistence-image model reaches adjacent-pair R2 = +0.204
against CatBoost's +0.144, but the paired interval spans zero, so "better than
CatBoost" is not established.  That is not the same question as "does topology
add anything CatBoost does not already have".

A blend answers the second question directly.  If the two models carry the same
information, averaging them cannot beat the stronger one by more than noise.  If
topology carries a complementary signal, the blend beats CatBoost alone -- and
that is also the configuration anyone would actually deploy.

Guards against fooling ourselves:

* The blend weight is **fixed a priori** at a simple average, and separately
  scanned to show the whole curve rather than reporting only its maximum. A
  weight tuned on the test metric would manufacture the result.
* The same paired cluster bootstrap over extractants scores the blend, so the
  interval is comparable with every other number in this study.
* Overall R2 is reported alongside, because a blend that wins on adjacent pairs
  while destroying overall accuracy is not an improvement worth having.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.compare_arms import collect, attach_meta
from automl.topo.adjacent_test import paired_adjacent, adj_r2
from automl.topo.ensemble_adjacent import config_key, SEED_DIRS, _load

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/adjacent_blend.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--baseline", default="baseline::catboost::none")
    ap.add_argument("--config", default="pi_pair2_sel_picnn_baseline_2d_f3.5_h1")
    args = ap.parse_args()

    arms = collect()
    base = attach_meta(arms[args.baseline])

    members: dict[str, dict[str, pd.DataFrame]] = {}
    for d in SEED_DIRS:
        if d.exists():
            for name, df in _load(d).items():
                members.setdefault(config_key(name), {})[name] = df
    if args.config not in members:
        print(f"config {args.config!r} not found; have {sorted(members)}")
        return 1
    mem = members[args.config]

    idx = None
    for df in mem.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    idx = idx.intersection(base.index)
    stack = np.vstack([mem[k].loc[idx, "oof"].to_numpy(float) for k in sorted(mem)])
    topo = stack.mean(axis=0)

    b = base.loc[idx]
    y = b["y"].to_numpy(float)
    cat = b["oof"].to_numpy(float)
    comp = b["composition_key"].to_numpy()
    li = b["lanthanide_index"].to_numpy()

    print(f"rows={len(idx)}  topo seeds={len(mem)}")
    print(f"  CatBoost alone      adjR2 = {adj_r2(y, cat, comp, li):+.4f}   "
          f"R2 = {ev._r2(y, cat):+.4f}")
    print(f"  topology ensemble   adjR2 = {adj_r2(y, topo, comp, li):+.4f}   "
          f"R2 = {ev._r2(y, topo):+.4f}")
    print("")
    print("  blend curve (w = weight on topology):")
    rows = []
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        p = (1 - w) * cat + w * topo
        rows.append({"w": w, "adj_r2": adj_r2(y, p, comp, li),
                     "r2_overall": ev._r2(y, p)})
        print(f"    w={w:.1f}  adjR2 = {rows[-1]['adj_r2']:+.4f}   "
              f"R2 = {rows[-1]['r2_overall']:+.4f}")

    # The a-priori choice: a plain average, decided before seeing the curve.
    blended = b.copy()
    blended["oof"] = 0.5 * cat + 0.5 * topo
    r = paired_adjacent(base.loc[idx], attach_meta(blended), args.n_boot, seed=0)
    print("")
    if r is None:
        print("  blend could not be paired against the baseline")
        return 1
    verdict = ("BEATS CATBOOST" if r["lo"] > 0 else
               "worse" if r["hi"] < 0 else "not distinguishable")
    print(f"  50/50 blend vs CatBoost:  adjR2 = {r['arm_obs']:+.4f} "
          f"[{r['arm_lo']:+.3f}, {r['arm_hi']:+.3f}]   "
          f"delta = {r['delta']:+.4f} [{r['lo']:+.3f}, {r['hi']:+.3f}]   "
          f"P(better) = {r['p_better']:.2f}   {verdict}")
    print(f"  blend overall R2 = {ev._r2(y, blended['oof'].to_numpy(float)):+.4f} "
          f"(CatBoost alone {ev._r2(y, cat):+.4f})")

    # ---- nested weight selection ------------------------------------------
    # The curve above is descriptive: its optimum was found by looking at the
    # test metric, so "w = 0.2 improves both metrics" cannot be claimed from it.
    # Here the weight is chosen for each extractant using ONLY the other
    # extractants' out-of-fold rows, then applied to that extractant.  No row
    # ever contributes to choosing the weight it is scored under, so this is an
    # honest estimate of what a tuned blend achieves.
    groups = b["extractant_group"].to_numpy()
    uniq = pd.unique(groups)
    grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
    nested = np.empty(len(b), dtype=float)
    chosen = []
    for g in uniq:
        te = groups == g
        tr = ~te
        best_w, best_v = 0.0, -np.inf
        for w in grid:
            ptr = (1 - w) * cat[tr] + w * topo[tr]
            v = adj_r2(y[tr], ptr, comp[tr], li[tr])
            if np.isfinite(v) and v > best_v:
                best_v, best_w = v, w
        nested[te] = (1 - best_w) * cat[te] + best_w * topo[te]
        chosen.append(best_w)
    nb_ = b.copy(); nb_["oof"] = nested
    rn = paired_adjacent(base.loc[idx], attach_meta(nb_), args.n_boot, seed=0)
    print("")
    print(f"  nested-weight blend (w chosen per extractant on the others):")
    print(f"    median w = {float(np.median(chosen)):.2f}  "
          f"IQR [{float(np.percentile(chosen,25)):.2f}, "
          f"{float(np.percentile(chosen,75)):.2f}]")
    if rn is not None:
        vn = ("BEATS CATBOOST" if rn["lo"] > 0 else
              "worse" if rn["hi"] < 0 else "not distinguishable")
        print(f"    adjR2 = {rn['arm_obs']:+.4f} "
              f"[{rn['arm_lo']:+.3f}, {rn['arm_hi']:+.3f}]   "
              f"delta = {rn['delta']:+.4f} [{rn['lo']:+.3f}, {rn['hi']:+.3f}]   "
              f"P(better) = {rn['p_better']:.2f}   {vn}")
    print(f"    overall R2 = {ev._r2(y, nested):+.4f} "
          f"(CatBoost alone {ev._r2(y, cat):+.4f})")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n[blend] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
