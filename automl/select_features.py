#!/usr/bin/env python3
"""Automated feature selection: greedy block search + within-block pruning.

The first sweep showed that handing every 3D column to the model at once is
*worse* than the 2D baseline on selectivity: 2263 columns for 5946 rows, with
859 of them from one derived block, simply dilute the split search.  So the
AutoML has to choose, not just concatenate.  Two selectors run here.

1. ``greedy_blocks`` -- forward stepwise search over whole descriptor blocks.
   Start from the 2D baseline, add the block that improves the objective most
   under the same grouped CV, repeat until nothing improves by more than
   ``--tol``.  This is the "automated representation selection" the
   representation-benchmark literature recommends, at block granularity so the
   answer stays physically interpretable.

2. ``prune_within`` -- given a chosen block set, rank individual columns by
   permutation importance on out-of-fold folds and keep the top-k, then re-score.
   This separates "the block matters" from "three columns in the block matter".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl import models as mz
from automl.dataset import BLOCK_PRESETS, GROUP_COL, TARGET
from automl.experiment import ExperimentSpec, apply_row_filter, prepare_xy, run_cv
from automl.matrix_cache import load_cache

CANDIDATE_BLOCKS = ["g_core", "p3d_phys", "p3d_poly", "g1", "g2", "g3", "g4", "g5",
                    "g6", "g7", "g8", "g9", "g10", "g11", "g12c", "g13c",
                    "g14c", "qc"]


def score_blocks(df, blocks, block_names: Sequence[str], args) -> dict[str, float]:
    spec = ExperimentSpec(
        preset="+".join(block_names), model=args.model,
        params=json.loads(args.params) if args.params else {},
        weight_scheme=args.weight_scheme, n_splits=args.n_splits,
        repeats=args.repeats, seed=args.seed, row_filter=args.row_filter,
        tag="select")
    res = run_cv(df, blocks, spec, n_jobs=args.n_jobs)
    return res.metrics


def greedy_blocks(df, blocks, args) -> dict[str, Any]:
    base = list(BLOCK_PRESETS["baseline_2d"])
    history: list[dict[str, Any]] = []
    current = list(base)
    m = score_blocks(df, blocks, current, args)
    best = m[args.objective]
    history.append({"step": 0, "added": None, "blocks": list(current),
                    "metrics": m})
    print(f"[greedy] start {current} {args.objective}={best:.4f}", flush=True)

    remaining = [b for b in CANDIDATE_BLOCKS if b in blocks.mapping]
    step = 0
    while remaining:
        step += 1
        trials = []
        for cand in remaining:
            t = time.time()
            m = score_blocks(df, blocks, current + [cand], args)
            trials.append((cand, m[args.objective], m, time.time() - t))
            print(f"[greedy] step {step} try +{cand:10s} "
                  f"{args.objective}={m[args.objective]:.4f} "
                  f"(R2={m['r2_overall']:.4f} within={m['r2_within']:.4f}) "
                  f"[{time.time() - t:.0f}s]", flush=True)
        trials.sort(key=lambda t: t[1], reverse=True)
        cand, value, metrics, _ = trials[0]
        if value <= best + args.tol:
            print(f"[greedy] no block improves {args.objective} by > {args.tol}; stop",
                  flush=True)
            break
        current.append(cand)
        remaining.remove(cand)
        best = value
        history.append({"step": step, "added": cand, "blocks": list(current),
                        "metrics": metrics,
                        "all_trials": [{"block": c, "value": v} for c, v, _, _ in trials]})
        print(f"[greedy] step {step} ADD {cand} -> {args.objective}={best:.4f}",
              flush=True)
    return {"selected_blocks": current, "objective": args.objective,
            "best_value": best, "history": history}


def permutation_importance_oof(df, blocks, block_names, args, n_repeats: int = 3
                               ) -> pd.DataFrame:
    """Grouped-CV permutation importance, averaged over folds."""
    sub = apply_row_filter(df, args.row_filter)
    X, y, cols = prepare_xy(sub, blocks, "+".join(block_names))
    groups = sub[GROUP_COL].to_numpy()
    Xv = X.to_numpy(dtype=np.float64)
    weights = mz.sample_weights(sub, args.weight_scheme)
    params = json.loads(args.params) if args.params else {}
    rng = np.random.default_rng(args.seed)

    importance = np.zeros((len(cols),))
    counts = 0
    folds = ev.grouped_folds(groups, n_splits=args.n_splits, seed=args.seed)
    for tr, te in folds:
        model = mz.make_model(args.model, params, seed=args.seed, n_jobs=args.n_jobs)
        try:
            model.fit(Xv[tr], y[tr],
                      **({} if weights is None else {"sample_weight": weights[tr]}))
        except TypeError:
            model.fit(Xv[tr], y[tr])
        base_pred = model.predict(Xv[te])
        base_score = ev._r2(y[te], base_pred)
        for j in range(len(cols)):
            drops = []
            for _ in range(n_repeats):
                Xp = Xv[te].copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops.append(base_score - ev._r2(y[te], model.predict(Xp)))
            importance[j] += float(np.mean(drops))
        counts += 1
    frame = pd.DataFrame({"feature": cols, "importance": importance / max(counts, 1)})
    frame["block"] = frame["feature"].map(_block_of)
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def _block_of(name: str) -> str:
    if name.startswith("feat3d__complex_physical"):
        return "p3d_phys"
    if name.startswith("feat3d__polyhedron"):
        return "p3d_poly"
    if name.startswith("ecfp_"):
        return "ecfp"
    if name.startswith("cond__"):
        return "cond"
    if name.startswith("qc__"):
        return "qc"
    for tag in ("g10", "g11", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"):
        if name.startswith(tag + "__"):
            return tag
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["greedy", "importance"], default="greedy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--params", default="")
    ap.add_argument("--objective", default="r2_overall")
    ap.add_argument("--weight-scheme", default="none")
    ap.add_argument("--row-filter", default="has3d")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.001)
    ap.add_argument("--blocks", default="", help="importance task: block list")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, blocks, info = load_cache()

    if args.task == "greedy":
        result = greedy_blocks(df, blocks, args)
        tag = f"{args.model}_{args.objective}_{args.row_filter}"
        (out_dir / f"greedy_{tag}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"selected": result["selected_blocks"],
                          "best": result["best_value"]}, indent=2))
    else:
        names = (args.blocks.split(",") if args.blocks
                 else list(BLOCK_PRESETS["baseline_2d"]))
        frame = permutation_importance_oof(df, blocks, names, args)
        tag = f"{args.model}_{args.row_filter}"
        frame.to_csv(out_dir / f"importance_{tag}.csv", index=False)
        print(frame.head(40).to_string(index=False))
        print()
        print(frame.groupby("block")["importance"].agg(["sum", "max", "count"])
              .sort_values("sum", ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
