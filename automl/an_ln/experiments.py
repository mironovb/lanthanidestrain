#!/usr/bin/env python3
"""An/Ln transfer experiments on the Am+Ln row table.

Three questions, one harness (CatBoost level/shape split, the current best
tabular system; features = RDKit 10 + ECFP 2048 + metal 3 + conditions 64
+ is_actinide; the 4 geometry-plan columns are dropped for everyone since
Am has no structures):

  ln_only   train on lanthanide rows only.  Lanthanide metric on the legacy
            905-pair subset (for comparability with +0.318); Am rows in the
            test folds are scored ZERO-SHOT with Am as a pseudo-lanthanide
            at its radius (Am/Eu separation factor, extractants held out).
  ln_am     train on lanthanide + Am (+ Cm) rows with the is_actinide flag.
            Same two scores: does the extra chemistry help the lanthanide
            target, and how well is Am/Eu predicted when Am is in training?

Blocks: composition_key = canonical_smiles || the geom_cond__* bins, exactly
as automl/dataset.py builds it (verified against matrix.parquet for the
lanthanide rows).  Folds are grouped by canonical_smiles over the UNION of
rows, so an extractant's Am and Ln measurements are held out together.

Am/Eu pairs: blocks containing both an Am cell and a Eu cell;
dy = mean log D(Am) - mean log D(Eu); pairs with dy exactly 0 are dropped
(detection-limit rows carrying identical D for both metals).

Writes automl/reports/an_ln/<config>.json and OOF parquets under
automl/artifacts/an_ln/.

Usage:  PYTHONPATH=$PWD python3 -m automl.an_ln.experiments --config ln_only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy import stats

import automl.evaluation as ev

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/an_ln"
REPORTS = REPO / "automl/reports/an_ln"
MATRIX = REPO / "automl/artifacts/matrix/matrix.parquet"
LN3D = REPO / "data/processed/final_ml_dataset_3d.parquet"

GEOM_COND = ["geom_cond__acid_class", "geom_cond__acid_strength_bin",
             "geom_cond__nitrate_activity", "geom_cond__pH_bin",
             "geom_cond__diluent_family", "geom_cond__modifier_class",
             "geom_cond__temperature_bin", "geom_cond__contact_time_bin",
             "geom_cond__phase_ratio_bin", "geom_cond__metal_conc_bin"]
RDKIT10 = ["MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
           "NumAromaticRings", "NumAliphaticRings", "RingCount",
           "FractionCSP3", "MolLogP"]
METAL3 = ["Atomic Number_metal", "lanthanide_index", "Ionic Radius_metal"]
CHAMP = dict(iterations=1500, learning_rate=0.04, depth=9, l2_leaf_reg=3.0,
             rsm=0.3, loss_function="Quantile:alpha=0.6")


def load() -> tuple[pd.DataFrame, list[str]]:
    from automl.dataset import GEOM_COND_COLS
    df = pd.read_parquet(ART / "anln_rows.parquet")
    gc = [c for c in GEOM_COND_COLS if c in df.columns]
    assert len(gc) == len(GEOM_COND_COLS), f"missing geom_cond columns: {set(GEOM_COND_COLS) - set(gc)}"
    df["extractant_group"] = df["canonical_smiles"].astype(str)
    df["composition_key"] = (df["extractant_group"] + "||"
                             + df[gc].astype(str).agg("|".join, axis=1))
    # the row id is '{source_file}:{exp_id}', identical to the shipped
    # safe_exp_id -> exact join for the legacy mask and the matrix's key
    mx = pd.read_parquet(MATRIX, columns=["safe_exp_id", "composition_key",
                                          "geometry_ok", "has_3d"])
    mx = mx.rename(columns={"safe_exp_id": "row_id",
                            "composition_key": "ck_matrix"})
    df = df.merge(mx, on="row_id", how="left")
    ln = df[df["is_actinide"] == 0]
    matched = ln["ck_matrix"].notna()
    agree = (ln.loc[matched, "ck_matrix"] == ln.loc[matched, "composition_key"]).mean()
    print(f"[load] {len(df)} rows · Ln {int((df.is_actinide == 0).sum())} "
          f"(matched to shipped ids {int(matched.sum())}) · An "
          f"{int((df.is_actinide == 1).sum())} · reconstructed composition_key "
          f"agrees with matrix.parquet on {agree:.1%} of matched Ln rows")
    assert agree > 0.99, "composition_key reconstruction does not match"
    df["legacy"] = (df["geometry_ok"].fillna(False).astype(bool)
                    & df["has_3d"].fillna(False).astype(bool)
                    & (df["is_actinide"] == 0))
    ecfp = sorted((c for c in df.columns if c.startswith("ecfp_")),
                  key=lambda s: int(s.split("_")[1]))
    cond = sorted(c for c in df.columns if c.startswith("cond__"))
    feats = [c for c in RDKIT10 if c in df.columns] + ecfp + METAL3 + cond + ["is_actinide"]
    print(f"[load] {len(feats)} feature columns · legacy Ln rows {int(df.legacy.sum())}")
    return df, feats


def _cb(seed):
    return CatBoostRegressor(random_seed=seed, verbose=0,
                             allow_writing_files=False, thread_count=12, **CHAMP)


def run(df, feats, train_mask, folds, repeats, seed):
    X = df[feats].to_numpy(float)
    y = df["log_D"].to_numpy(float)
    g = df["extractant_group"].to_numpy()
    oof = np.zeros(len(y)); cnt = np.zeros(len(y))
    for rep in range(repeats):
        for tr, te in ev.grouped_folds(g, folds, seed=seed + rep):
            tr = tr[train_mask[tr]]                   # training subset only
            base = _cb(seed + rep).fit(X[tr], y[tr])
            ktr = pd.Series(g[tr])
            resid = y[tr] - ktr.map(pd.Series(y[tr]).groupby(ktr).mean()).to_numpy()
            rm = _cb(seed + rep).fit(X[tr], resid)
            bp = pd.Series(base.predict(X[te])); sp = pd.Series(rm.predict(X[te]))
            kte = pd.Series(g[te])
            anchor = bp.groupby(kte).transform("mean")
            p = anchor + (sp - sp.groupby(kte).transform("mean"))
            oof[te] += p.to_numpy(); cnt[te] += 1
    return oof / np.maximum(cnt, 1)


def score_ln(df, oof):
    m = df["legacy"].to_numpy()
    dy, dp = ev.adjacent_pair_arrays(df["log_D"].to_numpy()[m], oof[m],
                                     df["composition_key"].to_numpy()[m],
                                     df["lanthanide_index"].to_numpy()[m])
    return {"n_pairs": int(len(dy)), "r2": ev._r2(dy, dp),
            "pearson2": float(np.corrcoef(dy, dp)[0, 1] ** 2)}


def score_an(df, oof, an="Am", ln="Eu"):
    d = df.assign(oof=oof)
    cells = d.groupby(["composition_key", "metal"], as_index=False).agg(
        y=("log_D", "mean"), p=("oof", "mean"), ex=("extractant_group", "first"))
    rows = []
    for ck, blk in cells.groupby("composition_key"):
        b = blk.set_index("metal")
        if an in b.index and ln in b.index:
            dy = b.loc[an, "y"] - b.loc[ln, "y"]
            if dy == 0.0:
                continue                      # detection-limit artefact
            rows.append({"ex": b.loc[an, "ex"], "dy": dy,
                         "dp": b.loc[an, "p"] - b.loc[ln, "p"]})
    P = pd.DataFrame(rows)
    if len(P) < 10:
        return {"n_pairs": int(len(P))}
    mae_ex = P.assign(a=(P.dy - P.dp).abs()).groupby("ex")["a"].mean()
    return {"pair": f"{an}/{ln}", "n_pairs": int(len(P)),
            "n_extractants": int(P["ex"].nunique()),
            "r2": ev._r2(P["dy"].to_numpy(), P["dp"].to_numpy()),
            "spearman": float(stats.spearmanr(P.dy, P.dp).statistic),
            "sign_accuracy": float(np.mean(np.sign(P.dy) == np.sign(P.dp))),
            "macro_mae": float(mae_ex.mean()),
            "sd_dy": float(P.dy.std()), "sd_dp": float(P.dp.std())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=("ln_only", "ln_am"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 51, 67, 83])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)

    df, feats = load()
    if args.config == "ln_only":
        train_mask = (df["is_actinide"] == 0).to_numpy()
    else:
        train_mask = np.ones(len(df), bool)
    print(f"[{args.config}] training rows {int(train_mask.sum())}")

    oofs = []
    per_seed = []
    for sd in args.seeds:
        oof = run(df, feats, train_mask, args.folds, args.repeats, sd)
        oofs.append(oof)
        r = {"seed": sd, "ln_legacy": score_ln(df, oof),
             "am_eu": score_an(df, oof, "Am", "Eu"),
             "cm_eu": score_an(df, oof, "Cm", "Eu"),
             "am_cm": score_an(df, oof, "Am", "Cm")}
        per_seed.append(r)
        print(f"  seed {sd}: Ln legacy R2 {r['ln_legacy']['r2']:+.4f} · "
              f"Am/Eu R2 {r['am_eu'].get('r2', float('nan')):+.4f} "
              f"(n={r['am_eu']['n_pairs']}, rho={r['am_eu'].get('spearman', float('nan')):+.3f})",
              flush=True)
    ens = np.mean(oofs, axis=0)
    out = {"config": args.config, "seeds": args.seeds, "per_seed": per_seed,
           "ensemble": {"ln_legacy": score_ln(df, ens),
                        "am_eu": score_an(df, ens, "Am", "Eu"),
                        "cm_eu": score_an(df, ens, "Cm", "Eu"),
                        "am_cm": score_an(df, ens, "Am", "Cm")}}
    print(f"[{args.config}] ensemble: {json.dumps(out['ensemble'], indent=1)}")
    (REPORTS / f"{args.config}.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame({"row_id": df["row_id"], "y": df["log_D"], "oof": ens,
                  "metal": df["metal"], "composition_key": df["composition_key"],
                  "extractant_group": df["extractant_group"],
                  "legacy": df["legacy"]}).to_parquet(
        ART / f"oof_{args.config}_ens{len(oofs)}.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
