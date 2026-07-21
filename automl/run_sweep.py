#!/usr/bin/env python3
"""Stage B: the AutoML sweep.

Three sweep families, selected with ``--mode``:

  ablation   fixed, reasonable model; vary the feature-block preset.  Answers
             "which 3D signal source actually helps, and where does it help --
             between extractants or within one?"
  models     fixed feature preset; vary the model family and the sample-weight
             scheme.  Answers "what is the right learner for this table?"
  optuna     hyperparameter search for one (preset, model) pair, optimising a
             chosen metric under the same grouped CV.

Every mode writes JSON lines to ``--out-dir/results.jsonl`` so a SLURM array can
fan out and a single reader can merge afterwards.  Work is sharded by index so
array task ``i`` of ``n`` takes ``jobs[i::n]``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from automl.dataset import BLOCK_PRESETS
from automl.evaluation import format_metrics
from automl.experiment import ExperimentSpec, run_and_record
from automl.matrix_cache import load_cache
from automl import models as mz


# ---------------------------------------------------------------------------
# Job builders
# ---------------------------------------------------------------------------
# A mid-strength LightGBM: fast enough for ~50 presets x repeats, strong enough
# that a block that helps will show up.  Ablation conclusions are re-checked
# with the tuned model at the end.
ABLATION_MODEL_PARAMS = {
    "n_estimators": 900, "learning_rate": 0.04, "num_leaves": 63,
    "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.5,
    "reg_lambda": 5.0,
}


def jobs_ablation(args) -> list[ExperimentSpec]:
    presets = (args.presets.split(",") if args.presets
               else list(BLOCK_PRESETS))
    specs = []
    for preset in presets:
        for row_filter in args.row_filters.split(","):
            for model in args.models.split(","):
                specs.append(ExperimentSpec(
                    preset=preset, model=model,
                    params=dict(ABLATION_MODEL_PARAMS) if model.startswith("lgbm") else {},
                    weight_scheme=args.weight_scheme,
                    n_splits=args.n_splits, repeats=args.repeats, seed=args.seed,
                    row_filter=row_filter, tag="ablation"))
    return specs


def jobs_models(args) -> list[ExperimentSpec]:
    models = (args.models.split(",") if args.models else mz.AVAILABLE_MODELS)
    weights = args.weight_schemes.split(",")
    presets = (args.presets.split(",") if args.presets
               else ["baseline_2d", "all_3d"])
    specs = []
    for preset, model, w in itertools.product(presets, models, weights):
        specs.append(ExperimentSpec(
            preset=preset, model=model, params={}, weight_scheme=w,
            n_splits=args.n_splits, repeats=args.repeats, seed=args.seed,
            row_filter=args.row_filters.split(",")[0], tag="models"))
    return specs


def jobs_arch(args) -> list[ExperimentSpec]:
    """Architecture sweep: flat vs two-stage vs anchored-residual vs delta.

    Each architecture is crossed with the feature presets that matter, because
    the right architecture and the right representation are not independent --
    a delta learner can only exploit blocks that actually vary inside a
    composition block.
    """
    presets = (args.presets.split(",") if args.presets else
               ["baseline_2d", "inner_sphere", "selectivity", "all_new_3d"])
    base = args.models.split(",")[0] if args.models else "lgbm"
    fast = dict(ABLATION_MODEL_PARAMS)
    specs: list[ExperimentSpec] = []
    for preset in presets:
        variants: list[tuple[str, dict]] = [(f"{base}", dict(fast))]
        variants.append((f"twostage:{base}", dict(fast)))
        for level in ("extractant", "composition"):
            for sw in (1.0, 0.7, 0.4):
                variants.append((f"anchored:{base}",
                                 {**fast, "level": level, "shape_weight": sw}))
        for pk in ("strict", "binned"):
            for dw in (1.0, 0.6, 0.3):
                variants.append((f"pairwise:{base}",
                                 {**fast, "pair_key": pk, "delta_weight": dw}))
        for model, params in variants:
            tagbits = [f"{k}={v}" for k, v in params.items() if k not in fast]
            specs.append(ExperimentSpec(
                preset=preset, model=model, params=params,
                weight_scheme=args.weight_scheme, n_splits=args.n_splits,
                repeats=args.repeats, seed=args.seed,
                row_filter=args.row_filters.split(",")[0],
                tag="arch:" + ",".join(tagbits) if tagbits else "arch"))
    return specs


def run_optuna(args, df, blocks) -> None:
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir / 'optuna.db'}"
    study_name = f"{args.preset}__{args.model}__{args.objective}"

    def objective(trial: "optuna.Trial") -> float:
        params = mz.suggest_params(trial, args.model)
        weight_scheme = trial.suggest_categorical(
            "weight_scheme", args.weight_schemes.split(","))
        spec = ExperimentSpec(
            preset=args.preset, model=args.model, params=params,
            weight_scheme=weight_scheme, n_splits=args.n_splits,
            repeats=args.repeats, seed=args.seed,
            row_filter=args.row_filters.split(",")[0], tag="optuna")
        rec = run_and_record(df, blocks, spec, out_dir, n_jobs=args.n_jobs)
        if rec["status"] != "ok":
            raise optuna.TrialPruned()
        m = rec["metrics"]
        for k, v in m.items():
            if v is not None:
                trial.set_user_attr(k, v)
        value = m.get(args.objective)
        if value is None or not np.isfinite(value):
            raise optuna.TrialPruned()
        return float(value)

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        load_if_exists=True,
        sampler=TPESampler(seed=args.seed + args.shard, multivariate=True,
                           n_startup_trials=12))
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))
    # A study can legitimately end with zero completed trials (e.g. every trial
    # raised because the learner rejects the matrix).  Report that instead of
    # crashing the worker after it has already done all the work.
    try:
        best = study.best_trial
        payload = {"study": study_name, "best_value": best.value,
                   "best_params": best.params, "n_trials": len(study.trials)}
    except ValueError:
        payload = {"study": study_name, "best_value": None,
                   "n_trials": len(study.trials),
                   "note": "no trial completed; see results.jsonl for the errors"}
    print(json.dumps(payload, indent=2), flush=True)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["ablation", "models", "arch", "optuna"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--presets", type=str, default="")
    ap.add_argument("--models", type=str, default="lgbm")
    ap.add_argument("--weight-scheme", type=str, default="none")
    ap.add_argument("--weight-schemes", type=str, default="none,group_inv,target_lds,combo")
    ap.add_argument("--row-filters", type=str, default="has3d")
    ap.add_argument("--save-oof", action="store_true")
    # optuna only
    ap.add_argument("--preset", type=str, default="all_3d")
    ap.add_argument("--model", type=str, default="lgbm")
    ap.add_argument("--objective", type=str, default="r2_overall")
    ap.add_argument("--n-trials", type=int, default=40)
    args = ap.parse_args()

    t0 = time.time()
    df, blocks, info = load_cache()
    print(f"[sweep] host={socket.gethostname()} matrix={df.shape} "
          f"loaded in {time.time() - t0:.1f}s", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "optuna":
        run_optuna(args, df, blocks)
        return 0

    builder = {"ablation": jobs_ablation, "models": jobs_models, "arch": jobs_arch}[args.mode]
    specs = builder(args)
    mine = specs[args.shard::args.num_shards]
    print(f"[sweep] {len(specs)} specs total, shard {args.shard}/{args.num_shards} "
          f"-> {len(mine)} jobs", flush=True)

    shard_dir = out_dir / f"shard{args.shard:03d}"
    for i, spec in enumerate(mine):
        t = time.time()
        rec = run_and_record(df, blocks, spec, shard_dir, n_jobs=args.n_jobs,
                             save_oof=args.save_oof)
        if rec["status"] == "ok":
            m = {k: v for k, v in rec["metrics"].items() if v is not None}
            print(f"[{i + 1}/{len(mine)}] {spec.key()}  {format_metrics(m)} "
                  f"[{time.time() - t:.0f}s]", flush=True)
        else:
            print(f"[{i + 1}/{len(mine)}] {spec.key()}  FAILED {rec.get('error')}",
                  flush=True)
    print(f"[sweep] shard {args.shard} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
