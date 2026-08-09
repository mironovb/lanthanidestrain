"""Re-tune the two non-3D stack partners on the metric they are scored by.

Both were tuned for overall log D.  CatBoost reaches +0.4987 there and only
+0.1422 on adjacent pairs; the repaired fingerprint network reaches +0.3218 and
+0.2206.  ``metric_tension.py`` already measured what that costs: selecting the
sweep2 cells on overall R2 rather than adjacent R2 would have picked a cell that
was **-0.0303** on the quantity the study exists to predict.  The same argument
applies one level down, to the partners themselves, and it has never been run.

Nothing here touches the published arms.  New OOF parquets are written under
``automl/artifacts/c6_partners/`` with their own names, and the published
``oof_fcnn_std_scaler_ens16.parquet`` is left exactly where it is.

CPU only -- MLPRegressor and CatBoost, no GPU -- so this runs on xeon-p8
concurrently with the GPU waves rather than competing with them.

    python3 -m automl.topo.c6_partners --which fcnn
    python3 -m automl.topo.c6_partners --which catboost
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from automl import evaluation as ev
from automl.dataset import GROUP_COL, TARGET
from automl.topo.train import build_row_table

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/artifacts/c6_partners"
SPLIT = REPO / "automl/artifacts/c6_split"

SEEDS = (7, 11, 23, 37, 42, 51, 67, 83,
         211, 223, 233, 241, 251, 263, 271, 281)

# --- FCNN grid ------------------------------------------------------------
# The shipped repaired arm is hidden=(256,128), alpha=1e-3, lr=1e-3, and was
# never varied after the StandardScaler repair took it from +0.005 to +0.221.
# Every variant here keeps StandardScaler -- the rank transform is a settled
# question, not an axis.
FCNN_GRID = {
    "shipped":     dict(hidden=(256, 128), alpha=1e-3, lr=1e-3),
    "wide":        dict(hidden=(512, 256), alpha=1e-3, lr=1e-3),
    "deep":        dict(hidden=(256, 128, 64), alpha=1e-3, lr=1e-3),
    "narrow":      dict(hidden=(128, 64), alpha=1e-3, lr=1e-3),
    "a1e2":        dict(hidden=(256, 128), alpha=1e-2, lr=1e-3),
    "a1e4":        dict(hidden=(256, 128), alpha=1e-4, lr=1e-3),
    "lr3e4":       dict(hidden=(256, 128), alpha=1e-3, lr=3e-4),
    "lr3e3":       dict(hidden=(256, 128), alpha=1e-3, lr=3e-3),
    "wide_a1e2":   dict(hidden=(512, 256), alpha=1e-2, lr=1e-3),
    "longer":      dict(hidden=(256, 128), alpha=1e-3, lr=1e-3, max_iter=1200),
}

# --- CatBoost grid --------------------------------------------------------
# models.py defaults: 1500 iters, lr 0.04, depth 7, l2 3.0, rsm 0.6.
CAT_GRID = {
    "shipped":  dict(depth=7, lr=0.04, l2=3.0, rsm=0.6),
    "shallow":  dict(depth=4, lr=0.04, l2=3.0, rsm=0.6),
    "deep":     dict(depth=9, lr=0.04, l2=3.0, rsm=0.6),
    "slow":     dict(depth=7, lr=0.015, l2=3.0, rsm=0.6, iters=4000),
    "l2_10":    dict(depth=7, lr=0.04, l2=10.0, rsm=0.6),
    "rsm_03":   dict(depth=7, lr=0.04, l2=3.0, rsm=0.3),
    "shallow_slow": dict(depth=4, lr=0.015, l2=3.0, rsm=0.6, iters=4000),
    "mae":      dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="MAE"),
    # --- re-grid AROUND the winner -------------------------------------------
    # MAE was worth +0.1066 adjacent and +0.0115 log D, and survived on the
    # held-out third.  The grid above tuned depth/lr/l2/rsm under RMSE, so none
    # of those settings has ever been chosen for the loss that actually works.
    "mae_deep":   dict(depth=9, lr=0.04, l2=3.0, rsm=0.6, loss="MAE"),
    "mae_shallow": dict(depth=5, lr=0.04, l2=3.0, rsm=0.6, loss="MAE"),
    "mae_slow":   dict(depth=7, lr=0.015, l2=3.0, rsm=0.6, iters=4000, loss="MAE"),
    "mae_rsm03":  dict(depth=7, lr=0.04, l2=3.0, rsm=0.3, loss="MAE"),
    "mae_l2_10":  dict(depth=7, lr=0.04, l2=10.0, rsm=0.6, loss="MAE"),
    # Quantile(0.5) IS MAE; Huber sits between MAE and RMSE and localises how
    # much of the gain is robustness rather than the L1 gradient specifically.
    "huber":      dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Huber:delta=1"),
    "huber_d03":  dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Huber:delta=0.3"),
    # H2: Quantile(0.5) IS median regression and must MATCH MAE if the median is
    # the mechanism.  0.3 and 0.7 are just as robust but target a different
    # quantile -- if they lose, it is the MEDIAN specifically.
    "q50": dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Quantile:alpha=0.5"),
    "q30": dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Quantile:alpha=0.3"),
    "q70": dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Quantile:alpha=0.7"),
    # B3 was FALSIFIED by q70 > q50: the median is not optimal, an UPPER
    # quantile is.  Sweep to locate the optimum and check it is a smooth
    # interior maximum rather than one noisy cell.
    "q60": dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Quantile:alpha=0.6"),
    "q65": dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Quantile:alpha=0.65"),
    "q75": dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Quantile:alpha=0.75"),
    "q80": dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Quantile:alpha=0.8"),
    "q85": dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Quantile:alpha=0.85"),
    "q90": dict(depth=7, lr=0.04, l2=3.0, rsm=0.6, loss="Quantile:alpha=0.9"),
}


def _fcnn(seed: int, p: dict) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=p["hidden"], alpha=p["alpha"],
            learning_rate_init=p["lr"], batch_size=128,
            max_iter=int(p.get("max_iter", 400)), early_stopping=True,
            n_iter_no_change=25, validation_fraction=0.12,
            random_state=seed)),
    ])


def _catboost(seed: int, p: dict):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=int(p.get("iters", 1500)), learning_rate=p["lr"],
        depth=p["depth"], l2_leaf_reg=p["l2"], rsm=p["rsm"],
        loss_function=p.get("loss", "RMSE"), bagging_temperature=1.0,
        random_seed=seed, verbose=0, allow_writing_files=False)


def oof_for(kind: str, name: str, params: dict, df, X, y, groups,
            seeds, folds: int, repeats: int) -> np.ndarray:
    """Mean OOF over every seed -- the same convention the arms use."""
    acc = np.zeros(len(df))
    cnt = np.zeros(len(df))
    for s in seeds:
        for rep in range(repeats):
            for tr, te in ev.grouped_folds(groups, n_splits=folds,
                                           seed=s + rep):
                if kind == "fcnn":
                    m = _fcnn(s + rep, params)
                    m.fit(X[tr], y[tr])
                    p = m.predict(X[te])
                else:
                    m = _catboost(s + rep, params)
                    m.fit(X[tr], y[tr])
                    p = m.predict(X[te])
                acc[te] += p
                cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=("fcnn", "catboost"), required=True)
    ap.add_argument("--seeds", type=int, default=8,
                    help="how many of the 16 published seeds to use")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--restrict", default="screen",
                    help="c6_split partition to SCORE on; training always uses "
                         "the restricted rows too, matching the GPU screen")
    ap.add_argument("--only", default=None, help="run one grid entry")
    args = ap.parse_args()

    df, X, cols = build_row_table("baseline_2d", "snn")
    # "full" trains on all 162 extractants.  Needed for the endpoint: the stack
    # is scored on the report third, and an arm fitted only on screen+select
    # has no out-of-fold prediction there at all, so it could not enter.
    if args.restrict and args.restrict != "full":
        keep = set((SPLIT / f"{args.restrict}_extractants.txt").read_text().split())
        m = df[GROUP_COL].isin(keep).to_numpy()
        df, X = df[m].reset_index(drop=True), X[m]
        print(f"[partners] restricted to {args.restrict}: {len(df)} rows, "
              f"{df[GROUP_COL].nunique()} extractants")
    y = df[TARGET].to_numpy(dtype=float)
    groups = df[GROUP_COL].to_numpy()
    comp = df["composition_key"].to_numpy()
    li = df["lanthanide_index"].to_numpy()
    seeds = list(SEEDS[:args.seeds])

    grid = FCNN_GRID if args.which == "fcnn" else CAT_GRID
    if args.only:
        grid = {args.only: grid[args.only]}

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, params in grid.items():
        t0 = time.time()
        oof = oof_for(args.which, name, params, df, X, y, groups,
                      seeds, args.folds, args.repeats)
        dy, dp = ev.adjacent_pair_arrays(y, oof, comp, li)
        rec = {"which": args.which, "variant": name,
               "adj_r2": ev._r2(dy, dp), "n_pairs": len(dy),
               "logD_r2": ev._r2(y, oof), "seconds": round(time.time() - t0, 1),
               "seeds": len(seeds), "params": json.dumps(params, default=str)}
        rows.append(rec)
        print(f"  {args.which:8s} {name:14s} adj={rec['adj_r2']:+.4f} "
              f"logD={rec['logD_r2']:+.4f}  [{rec['seconds']:.0f}s]",
              flush=True)
        pd.DataFrame({"safe_exp_id": df["safe_exp_id"], "y": y, "oof": oof,
                      "extractant_group": groups, "composition_key": comp,
                      "metal": df["metal"], "lanthanide_index": li}
                     ).to_parquet(
            OUT / f"oof_c6p_{args.which}_{name}_{args.restrict}.parquet",
            index=False)

    res = pd.DataFrame(rows).sort_values("adj_r2", ascending=False)
    p = REPO / f"automl/reports/c6_partners_{args.which}_{args.restrict}.csv"
    res.to_csv(p, index=False)
    print(f"\n[partners] wrote {p}")
    print(res[["variant", "adj_r2", "logD_r2"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))
    # The tension, restated on this grid: the log-D winner is usually not the
    # adjacent-pair winner, which is the entire reason this module exists.
    best_adj = res.iloc[0]["variant"]
    best_lvl = res.sort_values("logD_r2", ascending=False).iloc[0]["variant"]
    print(f"[partners] adjacent-pair winner: {best_adj}; "
          f"log D winner: {best_lvl}"
          + ("  (same)" if best_adj == best_lvl else "  (DIFFERENT)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
