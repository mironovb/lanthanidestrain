#!/usr/bin/env python3
"""Does ``--deterministic`` actually remove the run-to-run noise?

The floor this attacks
----------------------
``PI_SWEEP_PRECISION.md`` measured an 8-seed ensemble moving by **0.0092**
between two runs of the identical configuration, and showed more seeds does not
fix it: 8 seeds bought a factor of 1.76 where independence would give 2.83,
because part of the noise is shared across every seed inside one process.  That
floor is larger than most differences this study argues about.  It is why
re-running one cell of 25 changed Stage A's winner, and why a 25-configuration
sweep could not select.

The measurement, not the assertion
----------------------------------
The same configuration is run three times in deterministic mode and twice in the
published mode, at four matched seeds.  Reproducibility is then read directly off
the out-of-fold vectors:

* **bit-identical** -- the strongest possible statement, and the only one worth
  making.  Two runs that agree to 1e-12 are still two runs; two runs that agree
  in every bit are one run.
* **ensemble spread** -- the quantity ``pi_precision`` reports, so the new number
  is comparable with the published 0.0092 rather than a new scale.

Arguing determinism from the source would be exactly the mistake this project
has been caught by before, so nothing here is inferred from the code.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from automl.topo.adjacent_test import adj_r2
from automl.topo.compare_arms import attach_meta

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/determinism"
OUT = REPO / "automl/reports/determinism_test.csv"


def load_mode(mode: str) -> dict[int, dict[int, pd.DataFrame]]:
    """{replicate: {seed: oof frame}} for one mode."""
    out: dict[int, dict[int, pd.DataFrame]] = {}
    for d in sorted(ART.glob(f"{mode}_rep*")):
        rep = int(d.name.split("rep")[-1])
        per: dict[int, pd.DataFrame] = {}
        for j in sorted(d.glob("run_*.json")):
            cfg = json.loads(j.read_text()).get("config", {})
            p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
            if p.exists():
                per[int(cfg.get("seed", -1))] = (
                    pd.read_parquet(p).drop_duplicates("safe_exp_id")
                    .set_index("safe_exp_id"))
        if per:
            out[rep] = per
    return out


def ensemble(per: dict[int, pd.DataFrame]) -> pd.DataFrame:
    idx = None
    for f in per.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    stack = np.vstack([per[s].loc[idx, "oof"].to_numpy(float)
                       for s in sorted(per)])
    ens = per[sorted(per)[0]].loc[idx].copy()
    ens["oof"] = stack.mean(axis=0)
    return attach_meta(ens)


def _adj(d: pd.DataFrame) -> float:
    return adj_r2(d["y"].to_numpy(float), d["oof"].to_numpy(float),
                  d["composition_key"].to_numpy(),
                  d["lanthanide_index"].to_numpy())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    rows = []
    print("=== per-seed reproducibility across replicate runs ===")
    print(f"  {'mode':8s} {'seed':>5s} {'reps':>5s} {'bit-identical':>14s} "
          f"{'max|diff|':>12s}")
    summary = {}
    for mode in ("det", "nondet"):
        reps = load_mode(mode)
        if len(reps) < 2:
            print(f"  {mode}: {len(reps)} replicate(s) on disk -- need 2+; "
                  f"nothing to compare")
            continue
        seeds = sorted(set.intersection(*[set(r) for r in reps.values()]))
        bit_all, maxdiff_all = [], []
        for s in seeds:
            fr = [reps[r][s] for r in sorted(reps) if s in reps[r]]
            idx = fr[0].index
            for f in fr[1:]:
                idx = idx.intersection(f.index)
            vals = [f.loc[idx, "oof"].to_numpy(float) for f in fr]
            md = max(float(np.max(np.abs(a - b)))
                     for a, b in combinations(vals, 2))
            bit = md == 0.0
            bit_all.append(bit); maxdiff_all.append(md)
            print(f"  {mode:8s} {s:5d} {len(fr):5d} {str(bit):>14s} "
                  f"{md:12.3e}")
            rows.append({"mode": mode, "seed": s, "n_reps": len(fr),
                         "bit_identical": bit, "max_abs_diff": md})

        # ensemble-level spread, directly comparable with PI_SWEEP_PRECISION
        ens = {r: ensemble(reps[r]) for r in sorted(reps)}
        scores = {r: _adj(ens[r]) for r in ens}
        pairs = [abs(scores[a] - scores[b]) for a, b in combinations(scores, 2)]
        summary[mode] = {
            "all_bit_identical": bool(np.all(bit_all)),
            "max_abs_diff": float(np.max(maxdiff_all)) if maxdiff_all else np.nan,
            "n_seeds_ensembled": len(next(iter(reps.values()))),
            "ensemble_scores": {int(r): float(v) for r, v in scores.items()},
            "mean_ensemble_diff": float(np.mean(pairs)) if pairs else np.nan,
        }
        rows.append({"mode": mode, "seed": -1, "n_reps": len(reps),
                     "bit_identical": summary[mode]["all_bit_identical"],
                     "max_abs_diff": summary[mode]["max_abs_diff"],
                     "mean_ensemble_diff": summary[mode]["mean_ensemble_diff"]})

    if not summary:
        print("\nNo replicate sets on disk yet.")
        return 1

    print("\n=== ensemble-level run-to-run spread "
          "(comparable with PI_SWEEP_PRECISION.md) ===")
    published = 0.0092
    for mode, s in summary.items():
        sc = ", ".join(f"rep{r}={v:+.4f}" for r, v in s["ensemble_scores"].items())
        print(f"  {mode:8s} {s['n_seeds_ensembled']}-seed ensemble: {sc}")
        print(f"           mean |difference| between runs = "
              f"{s['mean_ensemble_diff']:.4f}")
    print(f"  published 8-seed figure for reference          = {published:.4f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)

    print("\n=== verdict ===")
    det = summary.get("det")
    non = summary.get("nondet")
    if det and det["all_bit_identical"]:
        print("  ==> --deterministic REPRODUCES BIT-FOR-BIT. Every re-run of a "
              "configuration returns the identical out-of-fold vector, so the "
              "run-to-run floor is exactly 0 and a sweep can select on "
              "differences of any size.")
    elif det:
        print(f"  ==> --deterministic did NOT reach bit-identity "
              f"(max |diff| = {det['max_abs_diff']:.3e}). Something outside the "
              f"scatter reductions is still order-dependent; the sorted scatter "
              f"is necessary but not sufficient and the remaining source has to "
              f"be found before any sweep relies on it.")
    if non is not None:
        if non["all_bit_identical"]:
            print("  ==> the PUBLISHED mode also reproduces bit-for-bit here, "
                  "which contradicts PI_SWEEP_PRECISION.md. Do not use this "
                  "measurement: either the configurations differ from those "
                  "measured there, or the comparison is not what it appears.")
        else:
            print(f"  ==> the published mode does not "
                  f"(max |diff| = {non['max_abs_diff']:.3e}), which is the "
                  f"premise being tested, so the comparison is meaningful.")
    print(f"\n[determinism] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
