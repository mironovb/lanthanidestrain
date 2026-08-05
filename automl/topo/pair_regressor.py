#!/usr/bin/env python3
"""Train directly on adjacent pairs, with features no row model can see.

EXPLORATORY until pre-registered.  Tune half for selection; the confirm half is
not touched here.

The idea
--------
Every model in this study predicts a LEVEL per row, and the metric differences
block-averaged levels afterwards.  Campaign 3 showed that bolting a pair head
onto a level model fails, because the head sits on a pathway the metric never
reads.  This does the opposite: it makes the pair the training example.

That buys two things a row model cannot have:

1. **The target is the scored quantity.** dy is regressed directly, so the loss
   and `sel_adj_logSF_r2` are the same object.
2. **Difference features exist.** For a pair of adjacent lanthanides in one
   block, the DIFFERENCE of their complexes' 3D descriptors is a real feature
   vector. A row model sees each complex separately and can only recover such a
   difference implicitly, through two forward passes and a subtraction it was
   never trained to make.

The cost is data: 905 pairs, against 4,746 rows.  So the estimator is
deliberately small -- ridge and a shallow gradient booster -- and every fit is
leave-extractants-out, because a pair belongs to a block which belongs to an
extractant.

Usage
-----
    python3 -m automl.topo.pair_regressor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
OUT = REPORTS / "pair_regressor.csv"

# Shannon ionic radii, Ln(III), CN 8, Angstrom -- the physical axis the whole
# separation problem sits on.
IONIC_R = {1: 1.160, 2: 1.143, 3: 1.126, 4: 1.109, 6: 1.079, 7: 1.066,
           8: 1.053, 9: 1.040, 10: 1.027, 11: 1.015, 12: 1.004, 13: 0.994,
           14: 0.985, 15: 0.977}


def build_pair_table(preset: str = "baseline_2d_shape"):
    """One row per adjacent pair: block features, both metals, and their delta."""
    from automl.topo.train import build_row_table
    df, X, cols = build_row_table(preset=preset, arch="snn")
    X = np.asarray(X, dtype=np.float64)

    # Columns that describe the LIGAND and CONDITIONS are constant within a
    # block; columns that describe the COMPLEX vary with the metal.  Only the
    # latter can produce a meaningful difference feature.
    is_cplx = np.array([c.split("__")[0] in ("feat3d", "g3", "g4", "g8", "p3d_poly",
                                             "gE", "p3d", "g9")
                        for c in cols])
    blk_cols = np.flatnonzero(~is_cplx)
    cpx_cols = np.flatnonzero(is_cplx)

    rows, feats = [], []
    for blk, sub in df.groupby("composition_key"):
        g = sub["extractant_group"].iloc[0]
        by_metal = {}
        for m, s in sub.groupby("lanthanide_index"):
            by_metal[int(m)] = (float(s["log_D"].mean()), s.index.to_numpy())
        ks = sorted(by_metal)
        for a, b in zip(ks[:-1], ks[1:]):
            if b - a != 1:
                continue
            ya, ia = by_metal[a]
            yb, ib = by_metal[b]
            xa = np.nanmean(X[ia], axis=0)
            xb = np.nanmean(X[ib], axis=0)
            ra, rb = IONIC_R.get(a, np.nan), IONIC_R.get(b, np.nan)
            feats.append(np.concatenate([
                xa[blk_cols],                    # ligand + conditions
                xa[cpx_cols] - xb[cpx_cols],     # THE difference features
                (xa[cpx_cols] + xb[cpx_cols]) / 2.0,   # and their mean
                [ra, rb, ra - rb, (ra + rb) / 2.0, float(a), float(b)],
            ]))
            rows.append(dict(block=blk, group=g, m_light=a, m_heavy=b,
                             dy=ya - yb))
    meta = pd.DataFrame(rows)
    F = np.vstack(feats)
    names = ([f"blk::{cols[i]}" for i in blk_cols]
             + [f"d::{cols[i]}" for i in cpx_cols]
             + [f"mean::{cols[i]}" for i in cpx_cols]
             + ["r_light", "r_heavy", "d_radius", "mean_radius",
                "idx_light", "idx_heavy"])
    return meta, F, names


def _r2(dy, dp):
    tot = float(((dy - dy.mean()) ** 2).sum())
    return 1.0 - float(((dy - dp) ** 2).sum()) / tot if tot > 0 else float("nan")


def _standardise(tr, *others):
    med = np.nanmedian(tr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    a = np.where(np.isfinite(tr), tr, med)
    mu, sd = a.mean(0), a.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    out = [(a - mu) / sd]
    for o in others:
        b = np.where(np.isfinite(o), o, med)
        out.append((b - mu) / sd)
    return out


def oof_predict(meta, F, model: str, alpha: float = 10.0, seed: int = 7):
    """Leave-extractants-out out-of-fold predictions of dy."""
    g = meta["group"].to_numpy()
    dy = meta["dy"].to_numpy(float)
    pred = np.zeros_like(dy)
    for gtest in pd.unique(g):
        te = g == gtest
        tr = ~te
        if tr.sum() < 30:
            pred[te] = dy[tr].mean() if tr.sum() else 0.0
            continue
        Xtr, Xte = _standardise(F[tr], F[te])
        ytr = dy[tr]
        if model == "ridge":
            n = Xtr.shape[1]
            w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(n), Xtr.T @ (ytr - ytr.mean()))
            pred[te] = Xte @ w + ytr.mean()
        elif model == "mean":
            pred[te] = ytr.mean()
        else:
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(iterations=400, depth=4, learning_rate=0.05,
                                  l2_leaf_reg=10.0, random_seed=seed,
                                  verbose=False, allow_writing_files=False)
            m.fit(Xtr, ytr)
            pred[te] = m.predict(Xte)
    return pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="baseline_2d_shape")
    args = ap.parse_args()

    from automl.topo.objective_test import load_split
    tune, conf = load_split()
    meta, F, names = build_pair_table(args.preset)
    print(f"[pair-reg] {len(meta)} pairs, {F.shape[1]} features, "
          f"{meta['group'].nunique()} extractants")
    ndiff = sum(1 for n in names if n.startswith("d::"))
    print(f"[pair-reg] of which {ndiff} are DIFFERENCE features between the two "
          f"complexes -- the part no row model sees\n")

    is_tune = meta["group"].isin(tune).to_numpy()
    rows = []
    for model, alpha in (("mean", 0.0), ("ridge", 100.0), ("ridge", 1000.0),
                         ("catboost", 0.0)):
        try:
            p = oof_predict(meta, F, model, alpha=alpha)
        except ImportError:
            print(f"  {model}: unavailable"); continue
        dy = meta["dy"].to_numpy(float)
        rt = _r2(dy[is_tune], p[is_tune])
        lab = f"{model}" + (f" a={alpha:g}" if model == "ridge" else "")
        print(f"  {lab:16s} tune adj-R2 = {rt:+.4f}")
        rows.append(dict(model=lab, tune_adj_r2=rt, n_pairs=int(is_tune.sum())))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n  reference: best single arm on the same tune pairs = +0.2702 (D0)")
    print(f"  EXPLORATORY -- tune half only, no decision rule.")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
