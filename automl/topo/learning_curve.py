#!/usr/bin/env python3
"""Learning curve over the number of TRAINING extractants.

Is the adjacent-pair score data-bound?  Retrain the best tabular system (cell
anch_q60_q60: level + shape CatBoost, CHAMP params) under the usual
leave-extractants-out CV, but keep only a fraction f of each fold's training
extractants; test folds are untouched, so every run is scored on the same 905
legacy pairs with the canonical construction.

Draws are NESTED: for a given draw seed the fold's training extractants are
permuted once and the first ceil(f * n) are kept, so within a draw the curve
is monotone in data and the draw-to-draw spread is sampling noise.

Arms:  both        level and shape models see the subset (primary)
       shape_only  level model on the full training fold, shape on the subset

The f = 1.0 run of arm 'both' must reproduce the stored
automl/artifacts/anchored_champ/oof_anch_q60_q60_s42.parquet (same folds,
seeds, params, thread count); the script checks this and reports it.

    PYTHONPATH=$PWD python3 -m automl.topo.learning_curve --arm both --workers 4
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.topo.anchored_champion import CHAMP, load_table, _cb

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/learning_curve"
REPORTS = REPO / "automl/reports"
REF_OOF = REPO / "automl/artifacts/anchored_champ/oof_anch_q60_q60_s42.parquet"
SEED_SD = 0.0285      # per-seed sd of the full-data champion (8 seeds, I14)

_G = {}          # data shared with forked workers


def score(df: pd.DataFrame, oof: np.ndarray) -> dict:
    y = df["log_D"].to_numpy(float)
    dy, dp = ev.adjacent_pair_arrays(y, oof, df["composition_key"].to_numpy(),
                                     df["lanthanide_index"].to_numpy())
    return {"adj_r2": float(ev._r2(dy, dp)),
            "adj_pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2),
            "adj_disp": float(np.std(dp) / np.std(dy)),
            "logd_r2": float(ev._r2(y, oof)), "n_pairs": int(len(dy))}


def groups_with_pairs(df: pd.DataFrame) -> set:
    out = set()
    for g, sub in df.groupby("extractant_group"):
        dy, _ = ev.adjacent_pair_arrays(sub["log_D"].to_numpy(float),
                                        np.zeros(len(sub)),
                                        sub["composition_key"].to_numpy(),
                                        sub["lanthanide_index"].to_numpy())
        if len(dy):
            out.add(g)
    return out


def run_one(task: dict) -> dict:
    df, X = _G["df"], _G["X"]
    gp = _G["groups_with_pairs"]
    params = dict(CHAMP, iterations=task["iterations"])
    y = df["log_D"].to_numpy(float)
    g = df["extractant_group"].to_numpy()
    oof = np.zeros(len(y)); cnt = np.zeros(len(y))
    per_fold = []
    frac, draw, arm = task["frac"], task["draw"], task["arm"]
    for rep in range(task["repeats"]):
        folds = ev.grouped_folds(g, task["folds"], seed=task["seed"] + rep)
        for fi, (tr_full, te) in enumerate(folds):
            groups = np.unique(g[tr_full])
            if frac < 1.0:
                rng = np.random.default_rng([draw, rep, fi, 20260909])
                perm = rng.permutation(groups)
                kept = set(perm[:math.ceil(frac * len(groups))].tolist())
                tr = tr_full[np.isin(g[tr_full], list(kept))]
            else:
                tr = tr_full
            tr_level = tr_full if arm == "shape_only" else tr
            per_fold.append({"n_train_groups": int(len(np.unique(g[tr]))),
                             "n_train_groups_pairs": int(len(set(g[tr]) & gp)),
                             "n_train_rows": int(len(tr))})
            base = _cb(params, task["seed"] + rep).fit(X[tr_level], y[tr_level])
            ktr = pd.Series(g[tr])
            resid = y[tr] - ktr.map(pd.Series(y[tr]).groupby(ktr).mean()).to_numpy()
            rm = _cb(params, task["seed"] + rep).fit(X[tr], resid)
            bp = pd.Series(base.predict(X[te])); sp = pd.Series(rm.predict(X[te]))
            kte = pd.Series(g[te])
            anchor = bp.groupby(kte).transform("mean")
            p = anchor + (sp - sp.groupby(kte).transform("mean"))
            oof[te] += p.to_numpy(); cnt[te] += 1
    oof = oof / np.maximum(cnt, 1)
    rec = dict(task, **score(df, oof))
    for k in ("n_train_groups", "n_train_groups_pairs", "n_train_rows"):
        rec[k] = float(np.mean([f[k] for f in per_fold]))
    ART.mkdir(parents=True, exist_ok=True)
    tag = f"{arm}_f{frac:.3f}_d{draw}" + ("_smoke" if task["smoke"] else "")
    pd.DataFrame({"safe_exp_id": df["safe_exp_id"], "y": y, "oof": oof}).to_parquet(
        ART / f"oof_{tag}.parquet", index=False)
    return rec


def fits(runs: list[dict]) -> dict:
    """Curve fits on the draw means; x = mean training extractants with pairs."""
    R = pd.DataFrame(runs)
    m = R.groupby("frac").agg(n=("n_train_groups_pairs", "mean"),
                              r2=("adj_r2", "mean"), sd=("adj_r2", "std"),
                              lo=("adj_r2", "min"), hi=("adj_r2", "max")).reset_index()
    n, r2 = m["n"].to_numpy(float), m["r2"].to_numpy(float)
    out = {"points": m.fillna(0.0).to_dict("records")}
    n_full = float(n.max())
    # hyperbolic  R2 = Rinf - k / n   (linear in 1/n)
    A = np.column_stack([np.ones_like(n), -1.0 / n])
    coef, *_ = np.linalg.lstsq(A, r2, rcond=None)
    Rinf, k = float(coef[0]), float(coef[1])
    hyp = {"R_inf": Rinf, "k": k,
           "pred_1x": Rinf - k / n_full, "pred_2x": Rinf - k / (2 * n_full),
           "pred_4x": Rinf - k / (4 * n_full),
           "slope_per_extractant_at_full": k / n_full ** 2}
    out["hyperbolic"] = hyp
    # power law  R2 = Rinf - k * n^-c   (c in (0.2, 2]) by grid + lstsq
    best = None
    for c in np.linspace(0.2, 2.0, 91):
        A = np.column_stack([np.ones_like(n), -n ** (-c)])
        cf, res, *_ = np.linalg.lstsq(A, r2, rcond=None)
        sse = float(((A @ cf - r2) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, c, cf)
    sse, c, cf = best
    pw = {"R_inf": float(cf[0]), "k": float(cf[1]), "c": float(c), "sse": sse,
          "pred_1x": float(cf[0] - cf[1] * n_full ** (-c)),
          "pred_2x": float(cf[0] - cf[1] * (2 * n_full) ** (-c)),
          "pred_4x": float(cf[0] - cf[1] * (4 * n_full) ** (-c))}
    out["power"] = pw
    # verdict from the numbers: gain from doubling vs draw sd at the largest subsampled fraction
    sub = m[m.frac < 1.0]
    sd_ref = float(sub.loc[sub.frac.idxmax(), "sd"]) if len(sub) else float("nan")
    gain2 = 0.5 * ((hyp["pred_2x"] - hyp["pred_1x"]) + (pw["pred_2x"] - pw["pred_1x"]))
    gain4 = 0.5 * ((hyp["pred_4x"] - hyp["pred_1x"]) + (pw["pred_4x"] - pw["pred_1x"]))
    last_step = float(r2[-1] - r2[-2]) if len(r2) >= 2 else float("nan")
    out.update({"gain_from_doubling_mean_of_fits": float(gain2),
                "gain_from_quadrupling_mean_of_fits": float(gain4),
                "last_step_observed": last_step,
                "draw_sd_at_largest_subsample": sd_ref, "seed_sd_reference": SEED_SD})
    if not np.isfinite(sd_ref) or sd_ref == 0:
        verdict = "insufficient draws for a verdict"
    elif gain2 > 2 * max(sd_ref, SEED_SD):
        verdict = (f"still rising: doubling the extractants is worth about {gain2:+.3f} R2, "
                   f"more than twice the draw sd ({sd_ref:.3f}) and the seed sd ({SEED_SD:.3f})")
    elif gain2 > max(sd_ref, SEED_SD):
        verdict = (f"rising slowly: doubling worth {gain2:+.3f} R2, above the draw sd ({sd_ref:.3f}) "
                   f"and the seed sd ({SEED_SD:.3f}) but less than twice either")
    else:
        verdict = (f"saturating: doubling worth {gain2:+.3f} R2, within the draw sd ({sd_ref:.3f}) "
                   f"or the seed sd ({SEED_SD:.3f}); observed step over the last fraction {last_step:+.3f}")
    out["verdict"] = verdict
    return out


def aggregate(arm: str) -> int:
    files = sorted(REPORTS.glob(f"learning_curve_{arm}_s*.json"))
    files = [f for f in files if "smoke" not in f.name]
    runs, seeds = [], []
    for f in files:
        d = json.loads(f.read_text()); seeds.append(d["seed"])
        for r in d["runs"]:
            runs.append(dict(r, seed=d["seed"]))
    out = {"arm": arm, "seeds": seeds, "n_runs": len(runs), "runs": runs, "fits": fits(runs)}
    print(f"[lc] aggregate over seeds {seeds}: {len(runs)} runs")
    for p in out["fits"]["points"]:
        print(f"  f={p['frac']:.3f}  n={p['n']:5.1f}  R2 {p['r2']:+.4f} sd {p['sd']:.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]")
    print("[lc] fits:", json.dumps({k: v for k, v in out["fits"].items() if k != "points"}, indent=1))
    (REPORTS / f"learning_curve_{arm}.json").write_text(json.dumps(out, indent=1))
    print("wrote", REPORTS / f"learning_curve_{arm}.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("both", "shape_only"), default="both")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.125, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true",
                    help="pool learning_curve_<arm>_s*.json over seeds and refit; no training")
    args = ap.parse_args()
    if args.aggregate:
        return aggregate(args.arm)
    iterations = CHAMP["iterations"]
    if args.smoke:
        args.fracs, args.draws, args.folds, args.repeats, iterations = [0.25, 1.0], 1, 2, 1, 100

    df, X, _ = load_table("ok_only")
    _G["df"], _G["X"] = df, X
    _G["groups_with_pairs"] = groups_with_pairs(df)
    print(f"[lc] {len(df)} rows, {df.extractant_group.nunique()} extractants, "
          f"{len(_G['groups_with_pairs'])} with pairs; arm={args.arm}", flush=True)

    tasks = [dict(frac=f, draw=d, arm=args.arm, seed=args.seed, repeats=args.repeats,
                  folds=args.folds, iterations=iterations, smoke=args.smoke)
             for f in args.fracs for d in (range(args.draws) if f < 1.0 else [0])]
    tasks.sort(key=lambda t: -t["frac"])                  # biggest first
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fu in as_completed(futs):
            r = fu.result(); runs.append(r)
            print(f"  f={r['frac']:.3f} d={r['draw']}  adj_R2={r['adj_r2']:+.4f} "
                  f"P2={r['adj_pearson2']:.4f} disp={r['adj_disp']:.3f} "
                  f"train ext={r['n_train_groups']:.1f} (with pairs {r['n_train_groups_pairs']:.1f}) "
                  f"rows={r['n_train_rows']:.0f}", flush=True)
    runs.sort(key=lambda r: (r["frac"], r["draw"]))
    out = {"arm": args.arm, "seed": args.seed, "folds": args.folds, "repeats": args.repeats,
           "iterations": iterations, "smoke": args.smoke, "runs": runs}
    # reproduction check against the stored champion OOF (same folds/seed/params)
    if REF_OOF.exists() and not args.smoke and args.arm == "both" and 1.0 in args.fracs:
        ref = pd.read_parquet(REF_OOF).set_index("safe_exp_id").loc[df["safe_exp_id"]]
        ref_score = score(df, ref["oof"].to_numpy())
        mine = [r for r in runs if r["frac"] == 1.0][0]
        out["reference_check"] = {"stored_adj_r2": ref_score["adj_r2"], "rerun_adj_r2": mine["adj_r2"],
                                  "abs_diff": abs(ref_score["adj_r2"] - mine["adj_r2"])}
        print(f"[lc] reference check: stored {ref_score['adj_r2']:+.4f} vs rerun {mine['adj_r2']:+.4f}", flush=True)
    if len(set(r["frac"] for r in runs)) >= 3:
        out["fits"] = fits(runs)
        print("[lc] fits:", json.dumps({k: v for k, v in out["fits"].items() if k != "points"}, indent=1))
        print("[lc] VERDICT:", out["fits"]["verdict"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    name = f"learning_curve_{args.arm}_s{args.seed}" + ("_smoke" if args.smoke else "") + ".json"
    (REPORTS / name).write_text(json.dumps(out, indent=1))
    print("wrote", REPORTS / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
