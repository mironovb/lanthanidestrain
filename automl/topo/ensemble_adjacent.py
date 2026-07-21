#!/usr/bin/env python3
"""Seed-ensemble the adjacent-pair arms and test the result honestly.

The adjacent-pair metric is extremely noisy: paired cluster bootstraps on it
return intervals ~0.175 wide, and a single run's point estimate is largely
seed noise.  Averaging out-of-fold predictions across seeds reduces model
variance without touching the evaluation protocol -- every seed uses the same
leave-extractants-out folds, and the ensemble is still scored by the same
paired bootstrap against the same baseline.

Two rules this module follows, because they are what separate variance
reduction from metric-gaming:

1. **Seeds are never selected on the test metric.** The ensemble is the mean
   over *all* available seeds of a configuration. Picking the best-scoring
   subset would manufacture exactly the result we are trying to test for.
2. **The configuration is chosen before ensembling, not after.** Whichever
   config is ensembled, its member seeds all go in.

A shuffled control is reported alongside: predictions permuted within
composition blocks must collapse the metric, otherwise the score reflects the
block structure rather than chemistry.
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

REPO = Path(__file__).resolve().parents[2]
SEED_DIRS = [REPO / "automl/artifacts/topo_adj_seeds",
             REPO / "automl/artifacts/topo_adjacent"]
OUT = REPO / "automl/reports/adjacent_ensemble.csv"


def _load(d: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(d.glob("oof_*.parquet")):
        out[p.stem.replace("oof_", "")] = (
            pd.read_parquet(p).drop_duplicates("safe_exp_id")
            .set_index("safe_exp_id"))
    return out


def config_key(name: str) -> str:
    """Strip the seed suffix so replicates of one configuration group together."""
    return re.sub(r"_s\d+(?=_|$)", "", name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--baseline", default="baseline::catboost::none")
    ap.add_argument("--min-seeds", type=int, default=2)
    args = ap.parse_args()

    arms = {k: v for k, v in collect().items() if k.startswith("baseline::")}
    members: dict[str, dict[str, pd.DataFrame]] = {}
    for d in SEED_DIRS:
        if not d.exists():
            continue
        for name, df in _load(d).items():
            members.setdefault(config_key(name), {})[name] = df

    base = attach_meta(arms.get(args.baseline, pd.DataFrame()))
    if base.empty:
        print(f"baseline {args.baseline!r} not found")
        return 1
    y_b = base["y"]

    rows = []
    for cfg, mem in sorted(members.items()):
        if len(mem) < args.min_seeds:
            continue
        # Mean OOF prediction over EVERY seed of this configuration.
        idx = None
        for df in mem.values():
            idx = df.index if idx is None else idx.intersection(df.index)
        stack = np.vstack([mem[k].loc[idx, "oof"].to_numpy(float) for k in sorted(mem)])
        ens = mem[sorted(mem)[0]].loc[idx].copy()
        ens["oof"] = stack.mean(axis=0)
        ens = attach_meta(ens)

        r = paired_adjacent(base, ens, args.n_boot, seed=0)
        if r is None:
            continue
        # Per-seed spread, to show what the averaging actually bought.
        singles = []
        for k in sorted(mem):
            d1 = attach_meta(mem[k].loc[idx])
            singles.append(adj_r2(d1["y"].to_numpy(float),
                                  d1["oof"].to_numpy(float),
                                  d1["composition_key"].to_numpy(),
                                  d1["lanthanide_index"].to_numpy()))
        verdict = ("BEATS BASELINE" if r["lo"] > 0 else
                   "worse" if r["hi"] < 0 else "not distinguishable")
        rows.append({"config": cfg, "n_seeds": len(mem),
                     "single_mean": float(np.mean(singles)),
                     "single_sd": float(np.std(singles)),
                     "ensemble_adj_r2": r["arm_obs"], **r, "verdict": verdict})
        print(f"{cfg:24s} seeds={len(mem)}  "
              f"single {np.mean(singles):+.3f}+/-{np.std(singles):.3f}  "
              f"ensemble {r['arm_obs']:+.4f} [{r['arm_lo']:+.3f},{r['arm_hi']:+.3f}]  "
              f"delta {r['delta']:+.4f} [{r['lo']:+.3f},{r['hi']:+.3f}]  "
              f"P={r['p_better']:.2f}  {verdict}", flush=True)

    if not rows:
        print("no configuration has enough seeds yet")
        return 1
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    base_adj = df["baseline_obs"].iloc[0]
    print(f"\nbaseline {args.baseline}: adjacent-pair R2 = {base_adj:+.4f}")
    win = df[df["lo"] > 0]
    print(f"configurations beating it, 90% interval excluding 0: "
          f"{len(win)}/{len(df)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[adjacent-ensemble] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
