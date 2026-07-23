#!/usr/bin/env python3
"""What is the smallest difference this sweep can actually resolve?

Stage B re-ran a configuration Stage A had already run -- same image set, same
eight seeds, same code, because the two manifests share one grid cell -- and the
8-seed ensemble moved from adj R2 +0.1696 to +0.1587.  ``train.py`` sets
``torch.manual_seed`` and nothing else, so on GPU cuDNN benchmarks its algorithm
choice and several reductions use non-deterministic atomics: identical seeds do
not give identical weights.

One accident is not an estimate.  This reads the deliberate replication -- three
configurations, three independent replicates each, all on the same eight seeds --
and reports the spread.  That spread is the noise floor every "difference between
configurations" in Stage A and Stage B has to clear before it means anything.

Two quantities are separated, because they are different things:

``within-configuration SD``
    Re-running the same configuration.  Pure nondeterminism.  This is the floor.

``between-configuration range``
    The spread Stage A actually reported across its 25 cells.  If it is not
    several times the floor, the sweep's ranking is mostly noise.

Also reports the per-seed spread, which is much larger and is what the 8-seed
ensembling was designed to control -- it is worth seeing that the thing I *did*
control was not the thing that bit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl.topo.best_stack import _score
from automl.topo.compare_arms import attach_meta
from automl.topo.control_factorial import ensemble
from automl.topo import pi_split
from automl.topo.pi_sweep_test import baselines, restrict, mechanism

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
REP_ROOT = REPO / "automl/artifacts/pi_replicate"

LABELS = {"13c391a8bb1e3a40": "shipped anchor (20px, 0.61px)",
          "9d6e4c93026dfa0c": "Stage A winner (96px, 0.50px)",
          "6e229d2419e7c99b": "mid-range (20px, 1.0px)"}


def load_replicates() -> dict[str, dict[int, dict[int, pd.DataFrame]]]:
    """key -> replicate -> seed -> out-of-fold frame."""
    out: dict[str, dict[int, dict[int, pd.DataFrame]]] = {}
    for rep_dir in sorted(REP_ROOT.glob("rep*")):
        rep = int(rep_dir.name.replace("rep", ""))
        for js in sorted(rep_dir.glob("run_*.json")):
            rec = json.loads(js.read_text())
            img = rec.get("resolved", {}).get("pi_images")
            if not img:
                continue
            key = Path(img).stem.replace("img_", "")
            pq = js.with_name(js.name.replace("run_", "oof_")
                              .replace(".json", ".parquet"))
            if not pq.exists():
                continue
            df = pd.read_parquet(pq).drop_duplicates(
                "safe_exp_id").set_index("safe_exp_id")
            out.setdefault(key, {}).setdefault(rep, {})[
                int(rec["config"]["seed"])] = df
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-seeds", type=int, default=8)
    args = ap.parse_args()

    reps = load_replicates()
    if not reps:
        raise SystemExit(f"no replicates under {REP_ROOT}")

    rec = pi_split.load()
    ref = baselines()["repaired"]
    tune = rec["tune"]
    ref_t = restrict(ref, ref, tune)

    rows = []
    print(f"{'configuration':32s} {'rep':>4s} {'seeds':>5s} {'adjR2':>9s} "
          f"{'err corr':>9s}")
    for key, by_rep in sorted(reps.items()):
        for rep, seeds in sorted(by_rep.items()):
            if len(seeds) < args.min_seeds:
                print(f"{LABELS.get(key, key):32s} {rep:4d} {len(seeds):5d}  "
                      f"-- INCOMPLETE, excluded")
                continue
            arm = restrict(ensemble(seeds), ref, tune)
            adj, corr = mechanism(arm, ref_t)
            per = [_score(restrict(s, ref, tune))[0] for s in seeds.values()]
            print(f"{LABELS.get(key, key):32s} {rep:4d} {len(seeds):5d} "
                  f"{adj:+9.4f} {corr:+9.3f}")
            rows.append({"key": key, "label": LABELS.get(key, key),
                         "replicate": rep, "n_seeds": len(seeds),
                         "adj_r2": adj, "err_corr": corr,
                         "per_seed_sd": float(np.std(per, ddof=1)),
                         "per_seed_min": float(np.min(per)),
                         "per_seed_max": float(np.max(per))})

    if not rows:
        print("\nno complete replicates yet")
        return 1
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "pi_precision.csv", index=False)

    print("\n=== within-configuration spread (pure nondeterminism) ===")
    stats = []
    for key, g in df.groupby("key"):
        if len(g) < 2:
            print(f"  {LABELS.get(key, key):32s} only {len(g)} replicate")
            continue
        sd = float(g["adj_r2"].std(ddof=1))
        rng = float(g["adj_r2"].max() - g["adj_r2"].min())
        print(f"  {LABELS.get(key, key):32s} n={len(g)}  "
              f"mean {g['adj_r2'].mean():+.4f}  SD {sd:.4f}  range {rng:.4f}")
        stats.append({"key": key, "sd": sd, "range": rng})

    if not stats:
        print("\nnot enough replicates to estimate the floor yet")
        return 1
    # Pool across configurations rather than averaging SDs: each contributes
    # (n-1) degrees of freedom and n is small, so an unweighted mean of SDs
    # would throw away the little information there is.
    ss = sum(float(((g["adj_r2"] - g["adj_r2"].mean()) ** 2).sum())
             for _, g in df.groupby("key") if len(g) >= 2)
    dof = sum(len(g) - 1 for _, g in df.groupby("key") if len(g) >= 2)
    floor_sd = float(np.sqrt(ss / dof)) if dof else float("nan")
    floor_rng = float(np.max([s["range"] for s in stats]))
    # A *difference* between two independently-run configurations carries the
    # nondeterminism of both, so its standard error is sigma*sqrt(2).  Comparing
    # a difference against a single-measurement sigma overstates significance by
    # 41 %, which is exactly the kind of error this document exists to catch.
    diff_se = floor_sd * np.sqrt(2.0)
    print(f"\nNOISE FLOOR: pooled within-configuration SD {floor_sd:.4f} "
          f"({dof} d.o.f.), worst observed range {floor_rng:.4f}")
    print(f"  a DIFFERENCE between two configurations therefore has SE "
          f"{diff_se:.4f} = sigma*sqrt(2)")
    print(f"  per-seed SD (what the 8-seed ensembling was designed to control): "
          f"{df['per_seed_sd'].mean():.4f}")

    sweep = REPORTS / "pi_sweep_stage_a.csv"
    if sweep.exists():
        a = pd.read_csv(sweep)
        anchor = a[a["is_anchor"]].iloc[0]["adj_r2"]
        best = a["adj_r2"].max()
        rng_a = a["adj_r2"].max() - a["adj_r2"].min()
        print("\n=== every Stage A difference, against that floor ===")
        for lab, val in (("best - anchor", best - anchor),
                         ("full Stage A range", rng_a),
                         ("S0 - best tuned", 0.2429 - best)):
            k = val / diff_se if diff_se > 0 else float("inf")
            verdict = ("resolvable" if k >= 3 else
                       "MARGINAL" if k >= 2 else "NOT RESOLVABLE")
            print(f"  {lab:22s} {val:+.4f}   {k:5.1f} sigma   {verdict}")
        print("\nSigma here is the SE of a difference between two independently"
              "-run\nconfigurations. Under ~2 sigma is not a finding, however "
              "many\nconfigurations it was selected from -- selection makes it "
              "worse, not better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
