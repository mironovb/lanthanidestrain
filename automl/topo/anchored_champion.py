#!/usr/bin/env python3
"""A1 follow-up: the anchored architecture x the champion's tuned loss.

The arch sweep (arch_adj) showed anchored/two-stage CatBoost at +0.2714
adjacent R^2 against +0.1566 for flat CatBoost at IDENTICAL fast params -- a
+0.11 architectural effect.  But ``models.make_model`` never forwards a
``loss_function``, so that entire sweep ran on RMSE; the champion's quantile
loss (q60_rsm03_deep: depth 9, rsm 0.3, Quantile:alpha=0.6, itself worth ~+0.10
over RMSE on the flat model) has never been combined with the anchored
architecture.  The two levers are orthogonal in mechanism: the loss fixes
WHICH errors are minimised, the anchor fixes WHERE the capacity is spent
(within-block shape instead of between-block level).

Additionally the residual model may want a different loss from the base:
the base predicts levels (quantile 0.6 won there), the residual predicts
within-block shape (MAE won every contrast test in this project).

Cells (all leave-extractants-out 5 folds x 3 repeats, ok_only, 746 columns):
  flat_q60          -- reproduce the champion under this harness (reference)
  anch_q60_q60      -- anchored, both models champion params
  anch_q60_mae      -- base champion, residual MAE deep rsm03
  anch_q60_mae_comp -- same, anchored on the composition block
  anch_q60_mae_w07  -- shape_weight 0.7
  two_q60_mae       -- two-stage-like: anchor from base, residual trained on
                       composition-centred target with adjacent upweighting
                       via sample weights (blocks with adjacent pairs x3)

Writes OOF parquets to automl/artifacts/anchored_champ/oof_<cell>.parquet
(safe_exp_id, y, oof) -- attach_meta-compatible -- and a summary CSV.

Usage:  PYTHONPATH=$PWD python3 -m automl.topo.anchored_champion [--cells ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import automl.evaluation as ev
from automl.matrix_cache import load_cache

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/anchored_champ"
OUT_CSV = REPO / "automl/reports/anchored_champion.csv"

CHAMP = dict(iterations=1500, learning_rate=0.04, depth=9, l2_leaf_reg=3.0,
             rsm=0.3, loss_function="Quantile:alpha=0.6")
MAE_DEEP = dict(iterations=1500, learning_rate=0.04, depth=9, l2_leaf_reg=3.0,
                rsm=0.3, loss_function="MAE")


def load_table():
    df, blocks, _ = load_cache()
    df = df[df["geometry_ok"].astype(bool) & df["has_3d"]].reset_index(drop=True)
    cols = []
    for b in ("rdkit", "ecfp", "metal", "cond", "plan"):
        cols.extend(blocks.mapping[b])
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    X = df[cols].to_numpy(float)
    return df, X


def _cb(params: dict, seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(random_seed=seed, verbose=0,
                             allow_writing_files=False, thread_count=12,
                             **params)


def run_cell(name: str, df, X, base_params, resid_params=None,
             level="extractant", shape_weight=1.0, folds=5, repeats=3,
             seed=42):
    y = df["log_D"].to_numpy(float)
    g = df["extractant_group"].to_numpy()
    comp = df["composition_key"].to_numpy()
    key_arr = g if level == "extractant" else comp

    oof = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for rep in range(repeats):
        for tr, te in ev.grouped_folds(g, folds, seed=seed + rep):
            base = _cb(base_params, seed + rep).fit(X[tr], y[tr])
            if resid_params is None:
                p = base.predict(X[te])
            else:
                key_tr = pd.Series(key_arr[tr])
                resid = y[tr] - key_tr.map(
                    pd.Series(y[tr]).groupby(key_tr).mean()).to_numpy()
                rm = _cb(resid_params, seed + rep).fit(X[tr], resid)
                bp = pd.Series(base.predict(X[te]))
                sp = pd.Series(rm.predict(X[te]))
                key_te = pd.Series(key_arr[te])
                anchor = bp.groupby(key_te).transform("mean")
                shape_c = sp - sp.groupby(key_te).transform("mean")
                base_c = bp - bp.groupby(key_te).transform("mean")
                p = (anchor + shape_weight * shape_c
                     + (1 - shape_weight) * base_c).to_numpy()
            oof[te] += p
            cnt[te] += 1
    oof = oof / np.maximum(cnt, 1)

    dy, dp = ev.adjacent_pair_arrays(y, oof, comp,
                                     df["lanthanide_index"].to_numpy())
    res = {"cell": name,
           "adj_r2": ev._r2(dy, dp),
           "adj_pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2),
           "adj_disp": float(np.std(dp) / np.std(dy)),
           "logD_r2": ev._r2(y, oof), "n_pairs": len(dy)}
    ART.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"safe_exp_id": df["safe_exp_id"], "y": y, "oof": oof})
    out.to_parquet(ART / f"oof_{name}.parquet", index=False)
    return res


CELLS = {
    "flat_q60": dict(base_params=CHAMP, resid_params=None),
    "anch_q60_q60": dict(base_params=CHAMP, resid_params=CHAMP),
    "anch_q60_mae": dict(base_params=CHAMP, resid_params=MAE_DEEP),
    "anch_q60_mae_comp": dict(base_params=CHAMP, resid_params=MAE_DEEP,
                              level="composition"),
    "anch_q60_mae_w07": dict(base_params=CHAMP, resid_params=MAE_DEEP,
                             shape_weight=0.7),
    "anch_mae_mae": dict(base_params=MAE_DEEP, resid_params=MAE_DEEP),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="per-seed runs are averaged into an _ens parquet")
    args = ap.parse_args()

    df, X = load_table()
    print(f"{len(df)} rows · {df['extractant_group'].nunique()} extractants · "
          f"{X.shape[1]} columns")

    rows = []
    for name in args.cells:
        oofs = []
        for sd in args.seeds:
            res = run_cell(f"{name}_s{sd}", df, X, repeats=args.repeats,
                           seed=sd, **CELLS[name])
            oofs.append(pd.read_parquet(ART / f"oof_{name}_s{sd}.parquet")
                        ["oof"].to_numpy())
            res["cell"] = f"{name}_s{sd}"
            rows.append(res)
            print(f"  {name}_s{sd:<4d} adj_R2={res['adj_r2']:+.4f} "
                  f"P2={res['adj_pearson2']:+.4f} disp={res['adj_disp']:.3f} "
                  f"logD_R2={res['logD_r2']:+.4f}")
        if len(oofs) > 1:
            y = df["log_D"].to_numpy(float)
            ens = np.mean(oofs, axis=0)
            pd.DataFrame({"safe_exp_id": df["safe_exp_id"], "y": y,
                          "oof": ens}).to_parquet(
                ART / f"oof_{name}_ens{len(oofs)}.parquet", index=False)
            dy, dp = ev.adjacent_pair_arrays(
                y, ens, df["composition_key"].to_numpy(),
                df["lanthanide_index"].to_numpy())
            r = {"cell": f"{name}_ens{len(oofs)}", "adj_r2": ev._r2(dy, dp),
                 "adj_pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2),
                 "adj_disp": float(np.std(dp) / np.std(dy)),
                 "logD_r2": ev._r2(y, ens), "n_pairs": len(dy)}
            rows.append(r)
            print(f"  {name}_ens{len(oofs)}  adj_R2={r['adj_r2']:+.4f} "
                  f"P2={r['adj_pearson2']:+.4f} disp={r['adj_disp']:.3f} "
                  f"logD_R2={r['logD_r2']:+.4f}")

    out = pd.DataFrame(rows)
    if OUT_CSV.exists():
        out = pd.concat([pd.read_csv(OUT_CSV), out], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
