#!/usr/bin/env python3
"""One dataset, two splits, two targets: the ladder from the R^2 that the
literature reports to the one this project scores.

Same rows (legacy population, 4,746 rows / 162 extractants), same learner
(the champion CatBoost, quantile 0.6, depth 9, 1,500 iterations, trained on
log D directly):

  * random rows held out (5-fold KFold, the protocol of the literature's
    R^2 = 0.85-0.96): log D R^2 / MAE, and the adjacent-pair R^2 of the
    same predictions;
  * extractants held out (grouped 5-fold, the project's protocol): log D
    R^2 / MAE and adjacent-pair R^2.

Also the variance decomposition of log D into the between-block level and
the within-block shape (the part the adjacent-pair score sees), and the
replicate scatter of adjacent separations measured more than once.

Writes automl/reports/split_ladder.json.  CPU; run on a compute node.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import automl.evaluation as ev
from automl.topo.anchored_champion import CHAMP, _cb, load_table

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/split_ladder.json"


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def run(X, y, splits, comp, midx, seed):
    oof = np.zeros(len(y))
    for tr, te in splits:
        oof[te] = _cb(CHAMP, seed).fit(X[tr], y[tr]).predict(X[te])
    dy, dp = ev.adjacent_pair_arrays(y, oof, comp, midx)
    return {"logD_r2": r2(y, oof), "logD_mae": float(np.abs(y - oof).mean()),
            "adj_r2": ev._r2(dy, dp),
            "adj_pearson": float(np.corrcoef(dy, dp)[0, 1]),
            "n_pairs": int(len(dy))}


def main() -> int:
    df, X, _ = load_table("ok_only")
    y = df["log_D"].to_numpy(float)
    g = df["extractant_group"].to_numpy()
    comp = df["composition_key"].to_numpy()
    midx = df["lanthanide_index"].to_numpy()
    out = {"n_rows": int(len(df)), "n_extractants": int(len(set(g))),
           "learner": "CatBoost champion on log D: " + json.dumps(CHAMP)}

    # variance decomposition: block means vs within-block deviations
    bm = pd.Series(y).groupby(comp).transform("mean").to_numpy()
    tot = ((y - y.mean()) ** 2).sum()
    out["variance"] = {"between_block_level": float(((bm - y.mean()) ** 2).sum() / tot),
                       "within_block_shape": float(((y - bm) ** 2).sum() / tot),
                       "n_blocks": int(len(set(comp)))}

    # replicate scatter of adjacent separations measured more than once
    cells = (pd.DataFrame({"y": y, "c": comp, "m": midx})
             .groupby(["c", "m"])["y"].agg(list).reset_index())
    reps, spread = [], []
    for _, blk in cells.groupby("c"):
        blk = blk.sort_values("m")
        for a, b in zip(blk.itertuples(), blk.iloc[1:].itertuples()):
            if b.m - a.m != 1:
                continue
            spread.append(np.mean(a.y) - np.mean(b.y))
            if len(a.y) > 1 and len(b.y) > 1:
                # scatter of the separation across replicate combinations
                d = np.array([ya - yb for ya in a.y for yb in b.y])
                reps.append(d.std(ddof=1))
    out["replicates"] = {"n_pairs_with_replicates": len(reps),
                         "median_sd_of_separation": float(np.median(reps)) if reps else None,
                         "sd_of_all_separations": float(np.std(spread))}

    seeds = [42, 51, 67]
    for name, maker in (("random_rows", lambda s: list(KFold(5, shuffle=True, random_state=s).split(X))),
                        ("extractants_held_out", lambda s: list(ev.grouped_folds(g, 5, seed=s)))):
        recs = []
        for s in seeds:
            t0 = time.time()
            recs.append(run(X, y, maker(s), comp, midx, s))
            print(f"[{name}] seed {s}: logD R2 {recs[-1]['logD_r2']:.3f} MAE {recs[-1]['logD_mae']:.3f} "
                  f"adj R2 {recs[-1]['adj_r2']:+.3f} ({time.time()-t0:.0f}s)", flush=True)
        out[name] = {"per_seed": recs,
                     **{k: float(np.mean([r[k] for r in recs])) for k in ("logD_r2", "logD_mae", "adj_r2", "adj_pearson")},
                     "sd_logD_r2": float(np.std([r["logD_r2"] for r in recs], ddof=1)),
                     "sd_adj_r2": float(np.std([r["adj_r2"] for r in recs], ddof=1))}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "learner"}, indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
