#!/usr/bin/env python3
"""Does a converged S0 ensemble clear the repaired baseline?

Pre-registered in ``automl/reports/S0X_PREREGISTRATION.md`` (commit ``86a35eb``),
written before any extra seed finished.  This module only executes what was
fixed there, and was itself written before the extra seeds landed.

**S0X** = every available seed of the *unchanged* S0 configuration: the published
16 (``topo_adj_seeds``, ``topo_adjacent``) plus the extras in
``topo_s0_extra``.  Not a new arm -- a better estimate of the ensemble S0
already defines.

Primary endpoint: S0X - repaired baseline, adjacent-pair R2, paired cluster
bootstrap over extractants, 400 draws, seed 0.  The four-test corrected interval
is printed beside it, because this is the fourth look at the same question.

The published 16-seed S0 must still re-ensemble to +0.2382 before any S0X number
is reported -- the extras live in a separate directory precisely so that stays
true.
"""

from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.adjacent_test import adj_r2
from automl.topo.compare_arms import attach_meta, collect
from automl.topo.control_factorial import (SEEDS as PUBLISHED_SEEDS, ensemble,
                                           load_cells, paired_adjacent_fast)
from automl.topo.stack_test import _corrected

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
EXTRA = REPO / "automl/artifacts/topo_s0_extra"
OUT = REPORTS / "s0x_test.csv"


def load_extra() -> dict[int, pd.DataFrame]:
    """Extra S0 seeds, identified by recorded config rather than by tag."""
    out: dict[int, pd.DataFrame] = {}
    for j in sorted(EXTRA.glob("run_*.json")):
        cfg = json.loads(j.read_text()).get("config", {})
        if cfg.get("arch") != "snn":
            continue
        # the unchanged S0 config, asserted rather than assumed
        if (float(cfg.get("pair_loss_weight") or 0) != 2.0
                or cfg.get("select_on") != "adjacent"
                or int(cfg.get("dim", 96)) != 96
                or int(cfg.get("layers", 3)) != 3
                or int(cfg.get("conformers", 1)) != 1
                or cfg.get("block_centre")):
            continue
        p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
        if p.exists():
            out[int(cfg["seed"])] = (pd.read_parquet(p)
                                     .drop_duplicates("safe_exp_id")
                                     .set_index("safe_exp_id"))
    return out


def _score(d):
    y = d["y"].to_numpy(float); p = d["oof"].to_numpy(float)
    return (adj_r2(y, p, d["composition_key"].to_numpy(),
                   d["lanthanide_index"].to_numpy()), ev._r2(y, p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--min-extra", type=int, default=16,
                    help="refuse to report on fewer than this many extra seeds")
    args = ap.parse_args()

    cells = load_cells(verbose=False)
    published = cells["S0"]
    extra = load_extra()
    overlap = set(published) & set(extra)
    if overlap:
        raise SystemExit(f"extra seeds overlap the published set: {sorted(overlap)}")
    print(f"published S0 seeds: {len(published)}  extra seeds: {len(extra)}")
    if len(extra) < args.min_extra:
        print(f"refusing to report on <{args.min_extra} extra seeds "
              f"(progress check only)")
        return 1

    s0 = ensemble(published)
    a0, r0 = _score(s0)
    print(f"\nharness check: published S0 re-ensembles to {a0:+.4f} "
          f"(must be +0.2382)")
    if abs(a0 - 0.2382) > 5e-4:
        raise SystemExit("published S0 drifted; refusing to report")

    s0x = ensemble({**published, **extra})
    ax, rx = _score(s0x)
    fixed = attach_meta(
        pd.read_parquet(REPORTS / "oof_fcnn_std_scaler_ens16.parquet")
        .drop_duplicates("safe_exp_id").set_index("safe_exp_id"))
    af, rf = _score(fixed)

    print(f"\n=== arms ===")
    print(f"  repaired baseline       adjR2={af:+.4f}  R2={rf:+.4f}")
    print(f"  S0  ({len(published):2d} seeds)        adjR2={a0:+.4f}  R2={r0:+.4f}")
    print(f"  S0X ({len(published)+len(extra):2d} seeds)        adjR2={ax:+.4f}  R2={rx:+.4f}")

    # seed-count curve: has the ensemble converged?
    allm = {**published, **extra}
    order = sorted(allm)
    print(f"\n=== seed-count curve (first n seeds, sorted) ===")
    curve = []
    for n in (4, 8, 16, 24, 32, 40, len(order)):
        if n > len(order):
            continue
        e = ensemble({s: allm[s] for s in order[:n]})
        v, _ = _score(e)
        curve.append({"n_seeds": n, "adj_r2": v})
        print(f"  n={n:3d}  adjR2={v:+.4f}")

    print(f"\n=== PRIMARY endpoint ===")
    rows = []
    for label, arm in (("S0X", s0x), ("S0(published)", s0)):
        r = paired_adjacent_fast(fixed, arm, args.n_boot, seed=0)
        if r is None:
            continue
        clo, chi = _corrected(r["delta"], r["lo"], r["hi"], 4)
        v = ("CLEARS" if r["lo"] > 0 else "worse" if r["hi"] < 0
             else "not distinguishable")
        cv = ("CLEARS" if clo > 0 else "worse" if chi < 0
              else "not distinguishable")
        star = "**" if label == "S0X" else "  "
        print(f"{star}{label:14s} - repaired  delta={r['delta']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  P={r['p_better']:.2f}  {v}")
        print(f"     4-test corrected [{clo:+.4f}, {chi:+.4f}]  {cv}")
        rows.append({"arm": label, "base": "repaired", **r,
                     "lo_4test": clo, "hi_4test": chi,
                     "verdict": v, "verdict_4test": cv})

    r = paired_adjacent_fast(s0, s0x, args.n_boot, seed=0)
    if r:
        print(f"\n  [secondary] S0X - S0(published)  delta={r['delta']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  -- what convergence bought")
        rows.append({"arm": "S0X", "base": "S0_published", **r,
                     "verdict": "secondary"})

    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        pd.DataFrame(curve).to_csv(OUT.with_name("s0x_curve.csv"), index=False)
        print(f"\n[s0x] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
