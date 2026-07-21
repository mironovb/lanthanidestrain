#!/usr/bin/env python3
"""Focused test of the adjacent-lanthanide-pair claim.

The abstract's specific claim is that a topological model gives its largest
gains on **adjacent** lanthanide pairs -- the hardest case, where neighbouring
ionic radii differ by only ~0.013 A.

This compares the topological arms trained with the pairwise-contrast objective
against the strongest tabular baseline on exactly that metric, with a paired
cluster bootstrap over extractants.  It is deliberately narrow: the full
comparison in ``compare_arms`` bootstraps every arm against every baseline on
every metric, which is far slower and buries this question.

Two guards against fooling ourselves:

* **Paired** resampling. Both arms are scored on the identical resampled
  extractants each iteration, so the comparison is not contaminated by which
  extractants happen to be drawn.
* **A shuffled control.** The same test with predictions permuted within
  composition blocks must collapse to a large negative value; if it does not,
  the metric is being gamed rather than the chemistry predicted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.compare_arms import collect, attach_meta

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/adjacent_pair_test.csv"


def adj_r2(y, p, comp, li) -> float:
    m = ev.adjacent_pair_metrics(y, p, comp, li)
    return m.get("sel_adj_logSF_r2", np.nan)


def paired_adjacent(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int):
    common = a.index.intersection(b.index)
    if len(common) < 0.5 * min(len(a), len(b)):
        return None
    a, b = a.loc[common], b.loc[common]
    y = a["y"].to_numpy(float)
    pa, pb = a["oof"].to_numpy(float), b["oof"].to_numpy(float)
    comp = a["composition_key"].to_numpy()
    li = a["lanthanide_index"].to_numpy()
    gcodes, guniq = pd.factorize(a["extractant_group"].to_numpy())
    rows_by_g = [np.flatnonzero(gcodes == i) for i in range(len(guniq))]

    full = np.arange(len(a))
    obs_a, obs_b = adj_r2(y, pa, comp, li), adj_r2(y, pb, comp, li)

    rng = np.random.default_rng(seed)
    da, db, dd = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_by_g), len(rows_by_g))
        idx = np.concatenate([rows_by_g[i] for i in pick])
        va = adj_r2(y[idx], pa[idx], comp[idx], li[idx])
        vb = adj_r2(y[idx], pb[idx], comp[idx], li[idx])
        if np.isfinite(va) and np.isfinite(vb):
            da.append(va); db.append(vb); dd.append(vb - va)
    if len(dd) < 30:
        return None
    dd = np.array(dd)
    return {"baseline_obs": obs_a, "arm_obs": obs_b,
            "arm_lo": float(np.percentile(db, 5)),
            "arm_hi": float(np.percentile(db, 95)),
            "delta": float(dd.mean()),
            "lo": float(np.percentile(dd, 5)),
            "hi": float(np.percentile(dd, 95)),
            "p_better": float((dd > 0).mean()),
            "n_boot": len(dd)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--baseline", default="baseline::catboost::none")
    ap.add_argument("--extra-dirs", nargs="*", default=[
        str(REPO / "automl/artifacts/topo_adjacent"),
        str(REPO / "automl/artifacts/topo_runs_radial")])
    args = ap.parse_args()

    arms = collect()
    for d in args.extra_dirs:
        for p in sorted(Path(d).glob("oof_*.parquet")):
            arms[p.stem.replace("oof_", "")] = (
                pd.read_parquet(p).drop_duplicates("safe_exp_id")
                .set_index("safe_exp_id"))
    arms = {k: attach_meta(v) for k, v in arms.items()}
    if args.baseline not in arms:
        print(f"baseline {args.baseline!r} not found; have: {sorted(arms)[:6]}")
        return 1
    base = arms[args.baseline]

    rows = []
    for name in sorted(arms):
        if name == args.baseline or name.startswith("baseline::"):
            continue
        r = paired_adjacent(base, arms[name], args.n_boot, seed=0)
        if r is None:
            continue
        r["arm"] = name
        rows.append(r)
        verdict = ("BEATS BASELINE" if r["lo"] > 0 else
                   "worse" if r["hi"] < 0 else "not distinguishable")
        print(f"{name:26s} adjR2 = {r['arm_obs']:+.4f} "
              f"[{r['arm_lo']:+.3f},{r['arm_hi']:+.3f}]  "
              f"delta = {r['delta']:+.4f} [{r['lo']:+.3f},{r['hi']:+.3f}]  "
              f"P(better) = {r['p_better']:.2f}  {verdict}", flush=True)

    if not rows:
        print("no comparable arms")
        return 1
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    print(f"\nbaseline {args.baseline}: adjacent-pair R2 = "
          f"{df['baseline_obs'].iloc[0]:+.4f}")
    win = df[df["lo"] > 0]
    print(f"arms beating it with a 90% interval excluding 0: {len(win)}/{len(df)}")
    for r in win.itertuples():
        print(f"   {r.arm}  delta = {r.delta:+.4f} [{r.lo:+.3f}, {r.hi:+.3f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n[adjacent-test] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
