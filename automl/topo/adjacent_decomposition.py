#!/usr/bin/env python3
"""Where does the missing adjacent-pair R^2 live?

No per-extractant or per-position breakdown of ``sel_adj_logSF_r2`` exists in
the repo.  This reconstructs the current best system -- the nested pair-fitted
NNLS stack over (CatBoost q60_rsm03_deep, c15_plw4, fcnn_repaired), the
+0.3132 in ``c17_stack_new.csv`` -- from the OOF parquets on disk, and
decomposes its error by:

  * extractant (which chemistries hold the residual variance),
  * pair position (which adjacent steps are worst -- ties to series_shape.py),
  * block size (do full-series blocks predict better than sparse ones),
  * pair magnitude (are large true separations under-predicted -- the known
    0.42x dispersion compression).

Purely diagnostic: reads existing artefacts, fits nothing new outside the
established nested-stack procedure.  Writes
``automl/reports/adjacent_decomposition.csv`` (per extractant) and
``adjacent_decomposition_position.csv`` (per pair position), prints the story.

Usage:  module load anaconda/Python-ML-2025a
        PYTHONPATH=$PWD python3 -m automl.topo.adjacent_decomposition
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.topo.c6_final import (ART, REPORTS, FCNN, align, nested_pair_stack,
                                  pair_matrix)
from automl.topo.compare_arms import attach_meta
from automl.topo.lift_report import ensemble, load_dirs

OUT_EX = REPORTS / "adjacent_decomposition.csv"
OUT_POS = REPORTS / "adjacent_decomposition_position.csv"

LN = ["", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
      "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]


def load_best_arms() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    cells = load_dirs(["topo_c15"])
    for k, slot in cells.items():
        name = sorted(slot["tags"])[0].rsplit("_s", 1)[0]
        if name == "c15_plw4" and len(slot["runs"]) >= 8:
            frames["c15_plw4"] = ensemble(slot["runs"])
    frames["fcnn_repaired"] = attach_meta(
        pd.read_parquet(FCNN).drop_duplicates("safe_exp_id")
        .set_index("safe_exp_id"))
    p = ART / "c6_partners/oof_c6p_catboost_q60_rsm03_deep_full.parquet"
    frames["cpu_catboost_q60_rsm03_deep"] = attach_meta(
        pd.read_parquet(p).drop_duplicates("safe_exp_id")
        .set_index("safe_exp_id"))
    assert set(frames) == {"c15_plw4", "fcnn_repaired",
                           "cpu_catboost_q60_rsm03_deep"}, sorted(frames)
    return align(frames)


def pair_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per adjacent pair with dy, stack dp, arm dps, extractant, l_lo."""
    names = sorted(frames)
    dy, pred, _ = nested_pair_stack(frames, names)
    _, A, grp = pair_matrix(frames, names)

    # recover l_lo per pair by re-walking the same enumeration order
    ref = frames[names[0]]
    y = ref["y"].to_numpy(float)
    comp = ref["composition_key"].to_numpy()
    li = ref["lanthanide_index"].to_numpy()
    g = ref["extractant_group"].to_numpy()
    lo_all, ck_all = [], []
    for grp_name in pd.unique(g):
        m = g == grp_name
        frame = pd.DataFrame({"y": y[m], "c": comp[m], "m": li[m]})
        for ck, blk in frame.groupby("c"):
            blk = blk.groupby("m", as_index=False)["y"].mean()
            mm = blk["m"].to_numpy()
            i, j = np.triu_indices(len(blk), k=1)
            adj = np.abs(mm[i] - mm[j]) == 1
            lo_all.extend(np.minimum(mm[i][adj], mm[j][adj]).tolist())
            ck_all.extend([ck] * int(adj.sum()))
    assert len(lo_all) == len(dy), (len(lo_all), len(dy))

    out = pd.DataFrame({"extractant_group": grp, "l_lo": np.array(lo_all, int),
                        "composition_key": ck_all,
                        "dy": dy, "dp_stack": pred})
    for k, n in enumerate(names):
        out[f"dp_{n}"] = A[:, k]
    out["pair"] = out["l_lo"].map(lambda v: f"{LN[v]}-{LN[v + 1]}")
    return out


def group_decomposition(pf: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-group error shares against the global variance."""
    ss_tot = float(np.sum((pf["dy"] - pf["dy"].mean()) ** 2))
    rows = []
    for gname, gg in pf.groupby(key):
        err = float(np.sum((gg["dy"] - gg["dp_stack"]) ** 2))
        rows.append({key: gname, "n_pairs": len(gg),
                     "sd_dy": float(gg["dy"].std()) if len(gg) > 1 else 0.0,
                     "sse": err, "share_of_total_error": np.nan,
                     "r2_local": (1 - err / np.sum((gg["dy"] - gg["dy"].mean()) ** 2))
                     if len(gg) > 2 and gg["dy"].std() > 0 else np.nan})
    df = pd.DataFrame(rows)
    sse_tot = df["sse"].sum()
    df["share_of_total_error"] = df["sse"] / sse_tot
    r2_now = 1 - sse_tot / ss_tot
    df["global_r2_if_solved"] = 1 - (sse_tot - df["sse"]) / ss_tot
    df["r2_gain_if_solved"] = df["global_r2_if_solved"] - r2_now
    return df.sort_values("sse", ascending=False)


def main() -> int:
    frames = load_best_arms()
    pf = pair_frame(frames)

    ss_tot = float(np.sum((pf["dy"] - pf["dy"].mean()) ** 2))
    sse = float(np.sum((pf["dy"] - pf["dp_stack"]) ** 2))
    r2 = 1 - sse / ss_tot
    print(f"reconstructed stack: {len(pf)} pairs, adj R^2 = {r2:+.4f} "
          f"(published +0.3132)")

    # dispersion
    slope = float(np.polyfit(pf['dy'], pf['dp_stack'], 1)[0])
    print(f"prediction sd / truth sd = {pf['dp_stack'].std() / pf['dy'].std():.3f}"
          f" · regression slope dp~dy = {slope:.3f}")

    ex = group_decomposition(pf, "extractant_group")
    pos = group_decomposition(pf, "pair")

    ex.to_csv(OUT_EX, index=False)
    pos.to_csv(OUT_POS, index=False)

    print("\n--- top 15 extractants by error share ---")
    top = ex.head(15)[["extractant_group", "n_pairs", "sd_dy",
                       "share_of_total_error", "r2_local", "r2_gain_if_solved"]]
    print(top.to_string(index=False,
                        formatters={"extractant_group": lambda s: s[:48],
                                    "share_of_total_error": "{:.1%}".format,
                                    "r2_local": "{:+.3f}".format,
                                    "r2_gain_if_solved": "{:+.4f}".format,
                                    "sd_dy": "{:.3f}".format}))
    cum = ex["share_of_total_error"].head(10).sum()
    print(f"top 10 extractants carry {cum:.1%} of the stack's squared error")

    print("\n--- by pair position ---")
    print(pos[["pair", "n_pairs", "sd_dy", "share_of_total_error", "r2_local",
               "r2_gain_if_solved"]].to_string(
        index=False, formatters={"share_of_total_error": "{:.1%}".format,
                                 "r2_local": "{:+.3f}".format,
                                 "r2_gain_if_solved": "{:+.4f}".format,
                                 "sd_dy": "{:.3f}".format}))

    # magnitude quartiles
    pf["absbin"] = pd.qcut(pf["dy"].abs(), 4,
                           labels=["q1_small", "q2", "q3", "q4_large"])
    print("\n--- by |dy| quartile ---")
    for b, gg in pf.groupby("absbin", observed=True):
        e = float(np.sum((gg["dy"] - gg["dp_stack"]) ** 2))
        print(f"  {b:>9}: n={len(gg):3d} sd={gg['dy'].std():.3f} "
              f"share_of_error={e / sse:.1%}")

    print(f"\nwrote {OUT_EX} and {OUT_POS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
