#!/usr/bin/env python3
"""Phase B of the post-0.313 campaign: the all-pairs delta model.

The first model in this project whose TRAINING population matches the metric's
structure.  One training example = one within-block cell pair (block = the
metric's own ``composition_key``; cell = block x metal, replicate-averaged
exactly as ``evaluation.adjacent_pair_arrays`` averages).  Training uses ALL
within-block pairs -- 6,389 under the binned key, 7.06x the 905 adjacent ones
the failed 2026-07 ``pair_regressor`` had -- and evaluation is on the adjacent
subset, which IS the metric.

Features per pair (lo = lighter metal, hi = heavier):
  * pair identity: indices, d_radius, mean radius, f-counts, tetrad
    coordinates, Gd-crossing, CN-break crossing -- banks the label-side series
    shape (series_shape.py: LOEO floor +0.066);
  * delta-conditions: cell-mean differences of the numeric condition columns
    (the strongest measured within-block signal, |r| up to 0.36) plus their
    block-level means;
  * ligand context: rdkit + plan + ECFP columns (constant within a block;
    trees learn ligand x identity interactions = ligand-dependent slopes);
  * optionally --energy: adjacent interaction-energy steps from the full
    metal-substitution probe (automl/artifacts/xtb_reference/metal_probe.csv),
    joined on the pair's cells once the full probe has landed.

Protocol: leave-extractants-out grouped folds (5 x --repeats), pairs grouped by
extractant so no extractant spans train and test.  Scored on adjacent pairs
with the canonical R^2; Pearson^2 and dispersion ratio reported alongside.

Outputs: one row per config in ``automl/reports/pair_model.csv`` and an OOF
pair parquet ``automl/artifacts/pair_model/oof_pairs_<tag>.parquet``
(composition_key, l_lo, l_hi, dy, dp) for stack integration.

Usage:
  PYTHONPATH=$PWD python3 -m automl.topo.pair_model --loss mae --adj-weight 3
  PYTHONPATH=$PWD python3 -m automl.topo.pair_model --sweep     # the full grid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import automl.evaluation as ev

REPO = Path(__file__).resolve().parents[2]
MATRIX = REPO / "automl/artifacts/matrix/matrix.parquet"
BLOCKS = REPO / "automl/artifacts/matrix/blocks.json"
PROBE = REPO / "automl/artifacts/xtb_reference/metal_probe.csv"
ART = REPO / "automl/artifacts/pair_model"
OUT_CSV = REPO / "automl/reports/pair_model.csv"

META = ["safe_exp_id", "extractant_group", "composition_key",
        "lanthanide_index", "log_D", "has_3d", "geometry_ok",
        "Ionic Radius_metal", "build_id"]

LN = ["", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
      "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
# 4f electron count of Ln(III): La(0) ... Lu(14); index 1-based, Pm=5 present
# for completeness though absent from data.
NF = {i: i - 1 for i in range(1, 16)}


def _block_cols() -> dict[str, list[str]]:
    return json.loads(BLOCKS.read_text())["blocks"] if "blocks" in \
        json.loads(BLOCKS.read_text()) else json.loads(BLOCKS.read_text())


def load_blocks() -> dict[str, list[str]]:
    raw = json.loads(BLOCKS.read_text())
    return raw.get("blocks", raw)


def build_pairs(strict: bool = False, population: str = "ok_only",
                energy: bool = False) -> pd.DataFrame:
    """The pair table.  Cached; rebuild with --rebuild."""
    blocks = load_blocks()
    lig_cols = blocks["rdkit"] + blocks["plan"] + blocks["ecfp"]
    cond_cols = blocks["cond"]
    key = "strict_composition_key" if strict else "composition_key"

    cols = sorted(set(META + lig_cols + cond_cols + ([key] if strict else [])))
    m = pd.read_parquet(MATRIX, columns=[c for c in cols])
    if population == "ok_only":
        m = m[m["geometry_ok"].astype(bool) & m["has_3d"]]
    elif population == "has3d":
        m = m[m["has_3d"].astype(bool)]
    m = m.reset_index(drop=True)

    # cell = (block, metal): y and conditions averaged, ligand features first
    agg = {c: "mean" for c in cond_cols}
    agg.update({c: "first" for c in lig_cols})
    agg["log_D"] = "mean"
    agg["Ionic Radius_metal"] = "first"
    agg["build_id"] = "first"
    cells = (m.groupby(["extractant_group", key, "lanthanide_index"],
                       as_index=False).agg(agg))

    probe = None
    if energy and PROBE.exists():
        pr = pd.read_csv(PROBE)
        if len(pr) > 100:            # the full probe, not the 12-row pilot
            pr["bid"] = pr["path"].str.extract(
                r"_([0-9a-f]{12})(?:_[A-Za-z0-9_]+)?\.xyz$")
            probe = pr.set_index("bid")

    rows = []
    for (ex, ck), blk in cells.groupby(["extractant_group", key]):
        blk = blk.sort_values("lanthanide_index")
        n = len(blk)
        if n < 2:
            continue
        idx = blk["lanthanide_index"].to_numpy(int)
        yv = blk["log_D"].to_numpy(float)
        rad = blk["Ionic Radius_metal"].to_numpy(float)
        bid = blk["build_id"].to_numpy()
        C = blk[cond_cols].to_numpy(float)
        lig = blk.iloc[0][lig_cols]
        cond_block_mean = C.mean(axis=0)
        for a in range(n):
            for b in range(a + 1, n):
                lo, hi = idx[a], idx[b]
                rec = {"extractant_group": ex, "composition_key": ck,
                       "l_lo": lo, "l_hi": hi, "dl": hi - lo,
                       "dy": yv[a] - yv[b],
                       "d_radius": rad[a] - rad[b],
                       "mean_radius": 0.5 * (rad[a] + rad[b]),
                       "nf_lo": NF[lo], "nf_hi": NF[hi],
                       "tetrad_q1_lo": abs(NF[lo] - 3.5),
                       "tetrad_q2_lo": abs(NF[lo] - 7.0),
                       "tetrad_q3_lo": abs(NF[lo] - 10.5),
                       "tetrad_q1_hi": abs(NF[hi] - 3.5),
                       "tetrad_q2_hi": abs(NF[hi] - 7.0),
                       "tetrad_q3_hi": abs(NF[hi] - 10.5),
                       "gd_cross": float(lo <= 8 <= hi),
                       "cn_cross": float(lo <= 8 < hi)}
                for k, c in enumerate(cond_cols):
                    rec[f"dcond__{c}"] = C[a, k] - C[b, k]
                    rec[f"bcond__{c}"] = cond_block_mean[k]
                if probe is not None:
                    e = {}
                    for tag, i in (("lo", a), ("hi", b)):
                        h = str(bid[i])[-12:] if bid[i] is not None else ""
                        e[tag] = probe.loc[h] if h in probe.index else None
                    de = []
                    for src in ("lo", "hi"):
                        if e[src] is not None:
                            va = e[src].get(f"eint_{LN[lo]}")
                            vb = e[src].get(f"eint_{LN[hi]}")
                            if pd.notna(va) and pd.notna(vb):
                                de.append(float(va) - float(vb))
                    rec["d_eint"] = float(np.mean(de)) if de else np.nan
                    rec["d_eint_n"] = float(len(de))
                rec.update(lig.to_dict())
                rows.append(rec)
    pf = pd.DataFrame(rows)
    return pf


def cached_pairs(args) -> pd.DataFrame:
    ART.mkdir(parents=True, exist_ok=True)
    tag = f"{args.population}_{'strict' if args.strict else 'binned'}" \
          f"{'_energy' if args.energy else ''}"
    cache = ART / f"pairs_{tag}.parquet"
    if cache.exists() and not args.rebuild:
        return pd.read_parquet(cache)
    pf = build_pairs(strict=args.strict, population=args.population,
                     energy=args.energy)
    pf.to_parquet(cache, index=False)
    print(f"[pairs] built {len(pf)} pairs -> {cache}")
    return pf


IDENTITY_COLS = ("l_lo", "l_hi", "dl", "d_radius", "mean_radius", "nf_lo",
                 "nf_hi", "tetrad_q1_lo", "tetrad_q2_lo", "tetrad_q3_lo",
                 "tetrad_q1_hi", "tetrad_q2_hi", "tetrad_q3_hi", "gd_cross",
                 "cn_cross", "d_eint", "d_eint_n")


def run_config(pf: pd.DataFrame, loss: str, adj_weight: float, folds: int,
               repeats: int, seed: int, learner: str = "catboost",
               features: str = "full") -> tuple[dict, pd.DataFrame]:
    feat_cols = [c for c in pf.columns
                 if c not in ("extractant_group", "composition_key", "dy")]
    if features == "identity_cond":
        feat_cols = [c for c in feat_cols
                     if c in IDENTITY_COLS or c.startswith(("dcond__",
                                                            "bcond__"))]
    elif features == "identity":
        feat_cols = [c for c in feat_cols if c in IDENTITY_COLS]
    X = pf[feat_cols].to_numpy(float)
    y = pf["dy"].to_numpy(float)
    g = pf["extractant_group"].to_numpy()
    adj = (pf["dl"] == 1).to_numpy()
    w = np.where(adj, adj_weight, 1.0)

    oof = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for rep in range(repeats):
        for tr, te in ev.grouped_folds(g, folds, seed=seed + rep):
            if learner == "catboost":
                from catboost import CatBoostRegressor
                cb_loss = {"mae": "MAE", "rmse": "RMSE"}.get(
                    loss, f"Quantile:alpha={loss[1:]}" if loss.startswith("q")
                    else loss)
                mo = CatBoostRegressor(
                    iterations=1500, learning_rate=0.05, depth=8,
                    loss_function=cb_loss, random_seed=seed + rep,
                    verbose=0, thread_count=8)
                mo.fit(X[tr], y[tr], sample_weight=w[tr])
            else:
                from lightgbm import LGBMRegressor
                objective = {"mae": "l1", "rmse": "l2"}.get(loss, "l1")
                mo = LGBMRegressor(n_estimators=1200, learning_rate=0.05,
                                   num_leaves=127, objective=objective,
                                   random_state=seed + rep, n_jobs=8)
                mo.fit(X[tr], y[tr], sample_weight=w[tr])
            oof[te] += mo.predict(X[te])
            cnt[te] += 1
    oof = oof / np.maximum(cnt, 1)

    dy_a, dp_a = y[adj], oof[adj]
    res = {"n_pairs_train": int(len(y)), "n_pairs_adj": int(adj.sum()),
           "adj_r2": ev._r2(dy_a, dp_a),
           "adj_pearson2": float(np.corrcoef(dy_a, dp_a)[0, 1] ** 2)
           if np.std(dp_a) > 0 else float("nan"),
           "adj_disp": float(np.std(dp_a) / np.std(dy_a)),
           "all_r2": ev._r2(y, oof)}
    oof_df = pf[["composition_key", "l_lo", "l_hi", "dl"]].copy()
    oof_df["dy"] = y
    oof_df["dp"] = oof
    oof_df["extractant_group"] = g
    return res, oof_df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loss", default="mae",
                    help="mae | rmse | q<alpha> (e.g. q0.6)")
    ap.add_argument("--adj-weight", type=float, default=3.0)
    ap.add_argument("--learner", default="catboost",
                    choices=("catboost", "lgbm"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--population", default="ok_only",
                    choices=("ok_only", "has3d"))
    ap.add_argument("--energy", action="store_true",
                    help="join d_eint from the full metal probe (needs the "
                         "full metal_probe.csv, not the 12-row pilot)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="loss x adj-weight x learner grid")
    ap.add_argument("--features", default="full",
                    choices=("full", "identity_cond", "identity"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    pf = cached_pairs(args)
    print(f"[pairs] {len(pf)} pairs · {pf['extractant_group'].nunique()} "
          f"extractants · {int((pf['dl'] == 1).sum())} adjacent")

    grid = ([(args.loss, args.adj_weight, args.learner)] if not args.sweep else
            [(l, w, ln) for l in ("mae", "q0.5", "q0.6", "q0.7", "rmse")
             for w in (1.0, 3.0, 10.0) for ln in ("catboost",)]
            + [("mae", 3.0, "lgbm")])

    rows = []
    for loss, aw, ln in grid:
        res, oof_df = run_config(pf, loss, aw, args.folds, args.repeats,
                                 args.seed, learner=ln,
                                 features=args.features)
        tag = (args.tag or
               f"{ln}_{loss}_w{aw:g}_{args.population}"
               f"{'' if args.features == 'full' else '_' + args.features}"
               f"{'_strict' if args.strict else ''}"
               f"{'_energy' if args.energy else ''}")
        res.update({"tag": tag, "loss": loss, "adj_weight": aw,
                    "learner": ln, "population": args.population,
                    "strict": args.strict, "energy": args.energy})
        rows.append(res)
        oof_path = ART / f"oof_pairs_{tag}.parquet"
        oof_df.to_parquet(oof_path, index=False)
        print(f"  {tag:42s} adj_R2={res['adj_r2']:+.4f} "
              f"P2={res['adj_pearson2']:+.4f} disp={res['adj_disp']:.3f} "
              f"all_R2={res['all_r2']:+.4f}")

    df = pd.DataFrame(rows)
    if OUT_CSV.exists():
        df = pd.concat([pd.read_csv(OUT_CSV), df], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
