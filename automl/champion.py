#!/usr/bin/env python3
"""Stage E: re-run the shortlisted configurations at high repeat count.

The sweeps use 2-3 repeats so they can cover a lot of ground; that leaves a
fold-to-fold spread of roughly +/-0.01 R^2, which is the same size as some of
the differences being compared.  This stage re-runs only the shortlist with 5
repeats x 5 folds (25 fits each) and reports a bootstrap confidence interval on
every headline metric, so the final claims are properly separated from noise.

It also writes the per-extractant and per-metal breakdowns used in the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.experiment import ExperimentSpec, apply_row_filter, run_cv
from automl.matrix_cache import load_cache

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "automl/artifacts/champion"

TUNED_LGBM = {"n_estimators": 1200, "learning_rate": 0.03, "num_leaves": 63,
              "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.5,
              "reg_lambda": 5.0}

# (label, preset, model, extra params, weight scheme).
# The model sweep showed CatBoost with inverse-extractant weighting is a much
# stronger base learner than LightGBM on this table (R2 0.518 vs 0.459 on the
# same 2D features), so the shortlist carries both: LightGBM rows keep the
# screening sweeps comparable, CatBoost rows are the ones that decide the
# headline number.
SHORTLIST = [
    # --- references ------------------------------------------------------
    ("baseline 2D, LightGBM",              "baseline_2d",   "lgbm",          {}, "none"),
    ("baseline 2D, CatBoost",              "baseline_2d",   "catboost",      {}, "none"),
    ("baseline 2D, CatBoost + group wts",  "baseline_2d",   "catboost",      {}, "group_inv"),
    ("+ shipped 3D blocks",                "plus_p3d_all",  "catboost",      {}, "group_inv"),
    # --- single blocks on the strong learner -------------------------------
    ("+ G5 xTB electronics",               "plus_g5",       "catboost",      {}, "group_inv"),
    ("+ G14c metal-free family means",     "plus_g14c",     "catboost",      {}, "group_inv"),
    ("+ G13c metal-free family slopes",    "plus_g13c",     "catboost",      {}, "group_inv"),
    ("+ G15c CN-effect removed",           "plus_g15c",     "catboost",      {}, "group_inv"),
    ("+ metal-free 3D only",               "ligand3d_only", "catboost",      {}, "group_inv"),
    ("+ curated g_core",                   "core3d_qc",     "catboost",      {}, "group_inv"),
    ("+ CN-free & metal-free",             "cnfree_ligand", "catboost",      {}, "group_inv"),
    ("+ G5 & metal-free",                  "g5_ligand",     "catboost",      {}, "group_inv"),
    ("+ all 3D (dilution control)",        "all_3d",        "catboost",      {}, "group_inv"),
    # --- architectures on the strong learner -------------------------------
    ("anchored, 2D only",                  "baseline_2d",   "anchored:catboost", {"shape_weight": 0.7}, "group_inv"),
    ("anchored + G5",                      "plus_g5",       "anchored:catboost", {"shape_weight": 0.7}, "group_inv"),
    ("anchored + CN-free",                 "plus_g15c",     "anchored:catboost", {"shape_weight": 0.7}, "group_inv"),
    ("anchored + CN-free & metal-free",    "cnfree_ligand", "anchored:catboost", {"shape_weight": 0.7}, "group_inv"),
    ("delta-learning + G5",                "plus_g5",       "pairwise:catboost", {"pair_key": "binned", "delta_weight": 0.6}, "group_inv"),
    ("delta-learning 2D only",             "baseline_2d",   "pairwise:catboost", {"pair_key": "binned", "delta_weight": 0.6}, "group_inv"),
    # --- LightGBM comparison rows -----------------------------------------
    ("LightGBM anchored + CN-free",        "plus_g15c",     "anchored:lgbm", {"shape_weight": 0.7}, "none"),
    ("LightGBM + metal-free 3D",           "plus_g14c",     "lgbm",          {}, "none"),
    ("LightGBM delta + inner-sphere",      "inner_sphere",  "pairwise:lgbm", {"pair_key": "binned", "delta_weight": 0.6}, "none"),
    # --- best-of-breed: stack every independent win found -------------------
    # CatBoost + inverse-extractant weights (biggest lever, sec.7)
    # + anchored-residual architecture (fixes the within component, sec.5)
    # + training-target winsorisation at log D = -6 (sec.7.2)
    # + the 3D block that survived on the strong learner.
    ("BEST 2D: cat+wts+anchored+clip",     "baseline_2d",   "anchored:catboost", {"shape_weight": 0.7, "_clip": 6.0}, "group_inv"),
    ("BEST +G5",                           "plus_g5",       "anchored:catboost", {"shape_weight": 0.7, "_clip": 6.0}, "group_inv"),
    ("BEST +metal-free 3D",                "plus_g14c",     "anchored:catboost", {"shape_weight": 0.7, "_clip": 6.0}, "group_inv"),
    ("BEST +CN-free",                      "plus_g15c",     "anchored:catboost", {"shape_weight": 0.7, "_clip": 6.0}, "group_inv"),
    ("BEST +CN-free & metal-free",         "cnfree_ligand", "anchored:catboost", {"shape_weight": 0.7, "_clip": 6.0}, "group_inv"),
    ("BEST +G5 & metal-free",              "g5_ligand",     "anchored:catboost", {"shape_weight": 0.7, "_clip": 6.0}, "group_inv"),
    ("BEST 2D clip only (control)",        "baseline_2d",   "catboost",      {"_clip": 6.0}, "group_inv"),
]


def bootstrap_ci(y: np.ndarray, p: np.ndarray, meta: pd.DataFrame,
                 n_boot: int = 200, seed: int = 0) -> dict[str, tuple[float, float]]:
    """Cluster bootstrap over extractants (the CV unit), 90 % interval."""
    rng = np.random.default_rng(seed)
    groups = meta["extractant_group"].to_numpy()
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    keys = ("r2_overall", "r2_between", "r2_within", "r2_within_composition")
    samples: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        m = ev.variance_decomposed_r2(y[idx], p[idx], groups[idx])
        sub = meta.iloc[idx]
        comp = sub["composition_key"].to_numpy()
        fr = pd.DataFrame({"y": y[idx], "p": p[idx], "c": comp})
        gm = fr.groupby("c")[["y", "p"]].transform("mean")
        yc = fr["y"].to_numpy() - gm["y"].to_numpy()
        pc = fr["p"].to_numpy() - gm["p"].to_numpy()
        ss = float(np.sum(yc ** 2))
        m["r2_within_composition"] = (1 - float(np.sum((yc - pc) ** 2)) / ss
                                      if ss > 0 else np.nan)
        for k in keys:
            v = m.get(k)
            if v is not None and np.isfinite(v):
                samples[k].append(float(v))
    return {k: (float(np.percentile(v, 5)), float(np.percentile(v, 95)))
            for k, v in samples.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--row-filter", default="has3d")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, blocks, info = load_cache()

    jobs = SHORTLIST[args.shard::args.num_shards]
    print(f"[champion] {len(jobs)} configs on shard {args.shard}/{args.num_shards}",
          flush=True)

    for label, preset, model, extra, wscheme in jobs:
        # "_clip" is a run-level option (winsorise the training target), not a
        # learner hyperparameter; strip it before it reaches the model.
        clip = float(extra.pop("_clip", 0.0)) if "_clip" in extra else 0.0
        # CatBoost has its own parameter names; do not hand it the LightGBM dict.
        base_family = model.partition(":")[2] or model
        params = ({**extra} if base_family == "catboost"
                  else {**TUNED_LGBM, **extra})
        spec = ExperimentSpec(preset=preset, model=model, params=params,
                              target_clip=clip,
                              weight_scheme=wscheme, n_splits=args.n_splits,
                              repeats=args.repeats, seed=42,
                              row_filter=args.row_filter, tag="champion")
        try:
            res = run_cv(df, blocks, spec, n_jobs=args.n_jobs)
        except Exception as exc:
            print(f"[champion] {label}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        sub = apply_row_filter(df, args.row_filter)
        y = sub["log_D"].to_numpy(dtype=float)
        ci = bootstrap_ci(y, res.oof, sub, n_boot=args.n_boot)
        record = {
            "label": label, "preset": preset, "model": model, "params": params,
            "row_filter": args.row_filter, "repeats": args.repeats,
            "metrics": {k: (None if not np.isfinite(v) else float(v))
                        for k, v in res.metrics.items()},
            "ci90": {k: list(v) for k, v in ci.items()},
        }
        safe = label.replace(" ", "_").replace("/", "-")
        (out_dir / f"champ_{args.shard}_{safe}.json").write_text(json.dumps(record, indent=2))
        pd.DataFrame({
            "safe_exp_id": sub["safe_exp_id"].to_numpy(),
            "y": y, "oof": res.oof,
            "extractant_group": sub["extractant_group"].to_numpy(),
            "composition_key": sub["composition_key"].to_numpy(),
            "metal": sub["metal"].to_numpy(),
            "lanthanide_index": sub["lanthanide_index"].to_numpy(),
        }).to_parquet(out_dir / f"oof_{args.shard}_{safe}.parquet", index=False)
        m = res.metrics
        lo, hi = ci.get("r2_overall", (np.nan, np.nan))
        print(f"[champion] {label:36s} R2={m['r2_overall']:.4f} "
              f"[{lo:.4f},{hi:.4f}] between={m['r2_between']:.4f} "
              f"within={m['r2_within']:.4f} withinComp={m['r2_within_composition']:.4f} "
              f"selSF={m.get('sel_logSF_r2', float('nan')):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
