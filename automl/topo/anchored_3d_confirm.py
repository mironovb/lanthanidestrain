#!/usr/bin/env python3
"""The fresh-444 confirmation of I15, rule declared BEFORE the data exists.

Written 14 Aug, while C19 (the expanded-population c15_plw4 retrain) is
still in the queue.  Primary endpoint, fixed in advance:

  blend(w = 0.35) vs tabular-only (w = 0), both built from
    anchor + shape mixes of
    - anchored champion trained on has3d (oof_anch_q60_q60_has3d_ens4)
    - C19 encoder ensemble (all seeds present in automl/artifacts/topo_c19)
  scored on the frozen fresh 444 pairs (fresh_eval.load_fresh).

  w = 0.35 is the median nested weight from the legacy-population analysis
  (I15); it is NOT re-fitted here.  PASS = blend R2 > tabular R2 on the
  fresh pairs (sign of the contrast, not a magnitude threshold; the fresh
  population is small and hard, so the pre-declared claim is the direction).

Secondary (reported alongside, not the endpoint): the nested equal-extractant
procedure of anchored_3d.py run on the legacy 905 of the expanded OOFs, and
the fresh-444 score of that nested system.

Usage:  PYTHONPATH=$PWD python3 -m automl.topo.anchored_3d_confirm
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.topo.fresh_eval import load_fresh

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/anchored_champ"
OUT = REPO / "automl/reports/anchored_3d_confirm.json"

W_FIXED = 0.35


def blend_frame(w: float) -> pd.DataFrame:
    anch = pd.read_parquet(ART / "oof_anch_q60_q60_has3d_ens4.parquet")
    import glob
    paths = sorted(glob.glob(str(
        REPO / "automl/artifacts/topo_c19/oof_c19_plw4h3d_s*.parquet")))
    if not paths:
        raise SystemExit("no c19_plw4h3d seed parquets found yet")
    frames = [pd.read_parquet(p).drop_duplicates("safe_exp_id")
              .set_index("safe_exp_id") for p in paths]
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    enc = frames[0].loc[idx, ["y"]].copy()
    enc["oof"] = np.mean([f.loc[idx, "oof"].to_numpy() for f in frames], axis=0)
    enc = enc.reset_index()
    n_seeds = len(paths)
    meta = pd.read_parquet(
        REPO / "automl/artifacts/matrix/matrix.parquet",
        columns=["safe_exp_id", "composition_key", "lanthanide_index"])
    df = (anch.merge(enc[["safe_exp_id", "oof"]].rename(columns={"oof": "enc"}),
                     on="safe_exp_id").merge(meta, on="safe_exp_id"))
    key = pd.Series(df["composition_key"])
    anchor = pd.Series(df["oof"]).groupby(key).transform("mean")
    st = pd.Series(df["oof"]) - anchor
    se = pd.Series(df["enc"]) - pd.Series(df["enc"]).groupby(key).transform("mean")
    df["blend"] = (anchor + (1 - w) * st + w * se).to_numpy()
    print(f"[confirm] joined {len(df)} rows; encoder seeds: {n_seeds}")
    return df


def score(df: pd.DataFrame, col: str, which: str) -> dict:
    fresh = load_fresh()
    dy_all, dp_all = [], []
    for ck, blk in df.groupby("composition_key"):
        blk = blk.groupby("lanthanide_index", as_index=False)[["y", col]].mean()
        idx = blk["lanthanide_index"].to_numpy()
        yv, pv = blk["y"].to_numpy(), blk[col].to_numpy()
        i, j = np.triu_indices(len(blk), k=1)
        adj = np.abs(idx[i] - idx[j]) == 1
        for a, b in zip(i[adj], j[adj]):
            lo, hi = sorted((int(idx[a]), int(idx[b])))
            is_fresh = (str(ck), lo, hi) in fresh
            if which == "fresh" and not is_fresh:
                continue
            if which == "legacy" and is_fresh:
                continue
            dy_all.append(yv[a] - yv[b]); dp_all.append(pv[a] - pv[b])
    dy, dp = np.asarray(dy_all), np.asarray(dp_all)
    out = {"n": int(len(dy)), "r2": ev._r2(dy, dp)}
    if np.std(dp) > 0:
        out["pearson2"] = float(np.corrcoef(dy, dp)[0, 1] ** 2)
    return out


def main() -> int:
    df = blend_frame(W_FIXED)
    df["tab"] = np.nan
    # tabular-only = w = 0 path through the same arithmetic
    key = pd.Series(df["composition_key"])
    anchor = pd.Series(df["oof"]).groupby(key).transform("mean")
    df["tab"] = (anchor + (pd.Series(df["oof"]) - anchor)).to_numpy()

    out = {"w_fixed": W_FIXED, "results": {}}
    for which in ("fresh", "legacy", "all"):
        b = score(df, "blend", which)
        t = score(df, "tab", which)
        out["results"][which] = {"blend": b, "tabular": t,
                                 "contrast": b["r2"] - t["r2"]}
        print(f"[{which:6s}] blend R2={b['r2']:+.4f} tabular R2={t['r2']:+.4f} "
              f"contrast={b['r2'] - t['r2']:+.4f} (n={b['n']})")
    verdict = out["results"]["fresh"]["contrast"] > 0
    out["primary_pass"] = bool(verdict)
    print(f"PRIMARY ({'PASS' if verdict else 'FAIL'}): "
          f"fresh-444 blend-vs-tabular contrast "
          f"{out['results']['fresh']['contrast']:+.4f}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
