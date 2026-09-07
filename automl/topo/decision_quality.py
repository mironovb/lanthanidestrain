#!/usr/bin/env python3
"""Decision quality of the best system: ranking, enrichment, and calibrated
uncertainty -- computed from out-of-fold predictions already on disk.

Part 1, ranking (what a screening user needs).  For each adjacent pair
position (La-Ce ... Yb-Lu) rank the held-out extractants by predicted
separation and compare with the measured ranking: Spearman rho, and
top-quartile enrichment (fraction of the predicted top quartile that is in
the measured top quartile; chance = 0.25).  Also pooled over all pairs,
and per extractant (rank the 12 positions).

Part 2, calibrated uncertainty.  Split-conformal intervals on the pair
residuals, calibrated leave-one-extractant-out: for each held-out
extractant the interval half-width is the (1-alpha) quantile of |residual|
over the other extractants' pairs.  Reported: empirical coverage at 80/90 %,
and an abstention curve -- R^2 on the fraction of pairs with the smallest
ensemble disagreement (8-seed spread of the anchored model's pair
prediction), which is the only per-pair uncertainty signal we have.

Both use the current best system: anchor + 0.65 tabular shape + 0.35
distance-encoder shape.  Writes automl/reports/decision_quality.json.

Usage:  PYTHONPATH=$PWD python3 -m automl.topo.decision_quality
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import automl.evaluation as ev
from automl.topo.topo_shape import load_cell

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/anchored_champ"
OUT = REPO / "automl/reports/decision_quality.json"
W = 0.35
LN = ["", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
      "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
SEEDS = [42, 51, 67, 83, 91, 103, 107, 109]


def pair_table() -> pd.DataFrame:
    """One row per adjacent pair: dy, blend prediction, per-seed blend
    predictions (for disagreement), position, extractant."""
    meta = pd.read_parquet(REPO / "automl/artifacts/matrix/matrix.parquet",
                           columns=["safe_exp_id", "composition_key",
                                    "lanthanide_index", "extractant_group"])
    dist, _ = load_cell("topo_c15", "c15_plw4")
    per_seed = {s: pd.read_parquet(ART / f"oof_anch_q60_q60_s{s}.parquet")
                for s in SEEDS}
    base = per_seed[SEEDS[0]][["safe_exp_id", "y"]].copy()
    base["ens"] = np.mean([per_seed[s]["oof"].to_numpy() for s in SEEDS], axis=0)
    for s in SEEDS:
        base[f"s{s}"] = per_seed[s]["oof"].to_numpy()
    df = (base.merge(dist[["safe_exp_id", "oof"]].rename(columns={"oof": "enc"}),
                     on="safe_exp_id").merge(meta, on="safe_exp_id"))
    key = pd.Series(df["composition_key"])

    def blend_col(tab_col):
        anchor = pd.Series(df[tab_col]).groupby(key).transform("mean")
        st = pd.Series(df[tab_col]) - anchor
        se = (pd.Series(df["enc"])
              - pd.Series(df["enc"]).groupby(key).transform("mean"))
        return (anchor + (1 - W) * st + W * se).to_numpy()

    df["blend"] = blend_col("ens")
    for s in SEEDS:
        df[f"b{s}"] = blend_col(f"s{s}")

    rows = []
    for ck, blk in df.groupby("composition_key"):
        cols = ["y", "blend"] + [f"b{s}" for s in SEEDS]
        c = blk.groupby("lanthanide_index", as_index=False)[cols].mean()
        li = c["lanthanide_index"].to_numpy(int)
        ex = blk["extractant_group"].iloc[0]
        for a in range(len(li) - 1):
            if li[a + 1] - li[a] != 1:
                continue
            rec = {"ex": ex, "l_lo": int(li[a]),
                   "dy": c["y"].iloc[a] - c["y"].iloc[a + 1],
                   "dp": c["blend"].iloc[a] - c["blend"].iloc[a + 1]}
            seeds = np.array([c[f"b{s}"].iloc[a] - c[f"b{s}"].iloc[a + 1]
                              for s in SEEDS])
            rec["spread"] = float(seeds.std())
            rows.append(rec)
    return pd.DataFrame(rows)


def ranking(P: pd.DataFrame) -> dict:
    out = {"by_position": {}}
    rhos, enrich = [], []
    for lo, g in P.groupby("l_lo"):
        # one value per extractant at this position (mean over its blocks)
        e = g.groupby("ex")[["dy", "dp"]].mean()
        if len(e) < 8:
            continue
        rho = float(stats.spearmanr(e["dy"], e["dp"]).statistic)
        q = max(2, len(e) // 4)
        top_true = set(e["dy"].nlargest(q).index)
        top_pred = set(e["dp"].nlargest(q).index)
        enr = len(top_true & top_pred) / q
        out["by_position"][f"{LN[lo]}-{LN[lo+1]}"] = {
            "n_extractants": int(len(e)), "spearman": round(rho, 3),
            "top_quartile_hit_rate": round(enr, 3), "q": int(q)}
        rhos.append(rho); enrich.append(enr)
    out["mean_spearman_over_positions"] = round(float(np.mean(rhos)), 3)
    out["mean_top_quartile_hit_rate"] = round(float(np.mean(enrich)), 3)
    out["chance_hit_rate"] = 0.25
    out["pooled_spearman_all_pairs"] = round(
        float(stats.spearmanr(P["dy"], P["dp"]).statistic), 3)
    # sign accuracy on pairs with a clear measured direction
    clear = P[P["dy"].abs() > 0.1]
    out["sign_accuracy_|dy|>0.1"] = round(
        float(np.mean(np.sign(clear["dy"]) == np.sign(clear["dp"]))), 3)
    out["n_pairs_|dy|>0.1"] = int(len(clear))
    # per-extractant: does the model order the 12 positions?
    per_ex = []
    for ex, g in P.groupby("ex"):
        if g["l_lo"].nunique() >= 6:
            e = g.groupby("l_lo")[["dy", "dp"]].mean()
            per_ex.append(float(stats.spearmanr(e["dy"], e["dp"]).statistic))
    out["median_within_extractant_spearman"] = round(float(np.median(per_ex)), 3)
    out["n_extractants_with_>=6_positions"] = len(per_ex)
    return out


def conformal(P: pd.DataFrame) -> dict:
    res = (P["dy"] - P["dp"]).to_numpy()
    ex = P["ex"].to_numpy()
    out = {"coverage": {}, "abstention": []}
    for alpha in (0.2, 0.1):
        covered, widths = [], []
        for g in np.unique(ex):
            te = ex == g
            q = float(np.quantile(np.abs(res[~te]), 1 - alpha))
            covered.extend((np.abs(res[te]) <= q).tolist())
            widths.append(q)
        out["coverage"][f"{int((1-alpha)*100)}%"] = {
            "empirical": round(float(np.mean(covered)), 3),
            "median_half_width": round(float(np.median(widths)), 3)}
    # abstention curve on ensemble disagreement
    order = np.argsort(P["spread"].to_numpy())
    dy, dp = P["dy"].to_numpy(), P["dp"].to_numpy()
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(20, int(round(frac * len(P))))
        idx = order[:k]
        out["abstention"].append({
            "kept_fraction": frac, "n": int(k),
            "r2": round(ev._r2(dy[idx], dp[idx]), 3),
            "mae": round(float(np.mean(np.abs(dy[idx] - dp[idx]))), 3),
            "sd_dy_kept": round(float(np.std(dy[idx])), 3)})
    out["spread_vs_abs_residual_spearman"] = round(
        float(stats.spearmanr(P["spread"], np.abs(res)).statistic), 3)
    return out


def main() -> int:
    P = pair_table()
    print(f"{len(P)} adjacent pairs · {P['ex'].nunique()} extractants · "
          f"R2 {ev._r2(P['dy'], P['dp']):+.4f}")
    R = ranking(P)
    C = conformal(P)
    print("\nranking:")
    for k, v in R.items():
        if k != "by_position":
            print(f"  {k}: {v}")
    print("  by position (spearman / top-quartile hit rate, chance 0.25):")
    for k, v in R["by_position"].items():
        print(f"    {k:6s} n={v['n_extractants']:2d}  rho={v['spearman']:+.2f}  "
              f"hit={v['top_quartile_hit_rate']:.2f}")
    print("\nconformal:")
    print(" ", C["coverage"])
    print(f"  spread vs |residual| spearman: {C['spread_vs_abs_residual_spearman']}")
    for a in C["abstention"]:
        print(f"  keep {a['kept_fraction']:.2f} (n={a['n']}): R2={a['r2']:+.3f} "
              f"MAE={a['mae']:.3f} sd(dy)={a['sd_dy_kept']:.3f}")
    OUT.write_text(json.dumps({"ranking": R, "conformal": C,
                               "n_pairs": int(len(P))}, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
