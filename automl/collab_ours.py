#!/usr/bin/env python3
"""Our best system, evaluated under the collaborator's protocol.

His cohort (see ``automl/collab_repro.py``) scores pair-level log SF on
34 extractants with equal-extractant macro MAE.  Our August-campaign models
predict per-row log D out-of-fold under leave-extractants-out CV -- strictly
harder than his 5-fold extractant-grouped CV, so injecting our OOF as an arm
is leakage-free by construction: the prediction for any pair never saw that
pair's extractant in training.

Arms injected (per-row OOF -> cell median over the cell's source rows ->
pair difference p_A - p_B):

  OURS_anchored     anchored CatBoost q60/q60 trained on the has3d
                    population (4-seed ensemble)
  OURS_encoder      C19 distance encoder on the expanded population
                    (8-seed ensemble)
  OURS_anchored3D   the confirmed blend: anchor + 0.65 tab + 0.35 encoder
                    shape (I15, w fixed at the pre-declared 0.35)

Scored with his metrics on exactly the pairs both sides cover; his A2
numbers are recomputed on the same covered subset so the comparison is
apples-to-apples.  Writes automl/reports/collab_ours/{coverage.json,
leaderboard_ours.csv,paired_bootstrap_ours.csv}.

Usage:  PYTHONPATH=$PWD python3 -m automl.collab_ours \
            [--repro-oof automl/reports/collab_repro/oof_predictions.parquet]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl.collab_repro import (DATA, LN_ORDER, Z_OF, build_cohort,
                                 paired_bootstrap, seed_metrics)

REPO = Path(__file__).resolve().parents[1]
OUTD = REPO / "automl/reports/collab_ours"
ANCH = REPO / "automl/artifacts/anchored_champ/oof_anch_q60_q60_has3d_ens4.parquet"
C19 = REPO / "automl/artifacts/topo_c19/oof_c19_plw4h3d_s*.parquet"
W_ENC = 0.35


def cell_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the cohort's cell membership, keeping safe_exp_id lists."""
    name = df["extractant_name"].astype(str).str.upper()
    todga = df.loc[name == "TODGA", "canonical_smiles"].mode().iloc[0]
    df = df[~((df["canonical_smiles"] == todga) & (name != "TODGA"))]
    df = df[df["metal"].isin(LN_ORDER) & np.isfinite(df["log_D"])]
    cond = sorted(c for c in df.columns if c.startswith("cond__"))
    df = df[df[cond].notna().all(axis=1)]
    key = (df["canonical_smiles"] + "|"
           + df[cond].apply(lambda r: "|".join(f"{v:.10g}" for v in r), axis=1))
    out = df[["safe_exp_id", "metal", "canonical_smiles"]].copy()
    out["cell_key"] = key + "|" + df["metal"]
    return out


def our_row_predictions() -> pd.DataFrame:
    anch = pd.read_parquet(ANCH)
    paths = sorted(glob.glob(str(C19)))
    frames = [pd.read_parquet(p).drop_duplicates("safe_exp_id")
              .set_index("safe_exp_id") for p in paths]
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    enc = pd.DataFrame({"safe_exp_id": idx,
                        "enc": np.mean([f.loc[idx, "oof"].to_numpy()
                                        for f in frames], axis=0)})
    meta = pd.read_parquet(REPO / "automl/artifacts/matrix/matrix.parquet",
                           columns=["safe_exp_id", "composition_key"])
    d = anch.merge(enc, on="safe_exp_id").merge(meta, on="safe_exp_id")
    key = pd.Series(d["composition_key"])
    anchor = pd.Series(d["oof"]).groupby(key).transform("mean")
    st = pd.Series(d["oof"]) - anchor
    se = pd.Series(d["enc"]) - pd.Series(d["enc"]).groupby(key).transform("mean")
    d["p_anchored"] = d["oof"]
    d["p_encoder"] = d["enc"]
    d["p_anchored3d"] = (anchor + (1 - W_ENC) * st + W_ENC * se).to_numpy()
    return d[["safe_exp_id", "p_anchored", "p_encoder", "p_anchored3d"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repro-oof",
                    default=str(REPO / "automl/reports/collab_repro"
                                       "/oof_predictions.parquet"))
    args = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA)
    frame, audit = build_cohort(df)
    cells = cell_rows(df)
    ours = our_row_predictions()
    cells = cells.merge(ours, on="safe_exp_id", how="left")
    cond = sorted(c for c in df.columns if c.startswith("cond__"))

    # cell-level medians of our per-row predictions
    cm = (cells.groupby("cell_key")[["p_anchored", "p_encoder",
                                     "p_anchored3d"]].median())
    cov_cells = float(cm["p_anchored3d"].notna().mean())

    # rebuild the pair-side cell keys the same way build_cohort groups
    sub = df.copy()
    name = sub["extractant_name"].astype(str).str.upper()
    todga = sub.loc[name == "TODGA", "canonical_smiles"].mode().iloc[0]
    sub = sub[~((sub["canonical_smiles"] == todga) & (name != "TODGA"))]
    sub = sub[sub["metal"].isin(LN_ORDER) & np.isfinite(sub["log_D"])]
    sub = sub[sub[cond].notna().all(axis=1)]
    ck = (sub["canonical_smiles"] + "|"
          + sub[cond].apply(lambda r: "|".join(f"{v:.10g}" for v in r), axis=1))
    # frame rows carry base__cond__* + extractant; reconstruct their keys
    fk = (frame["extractant"] + "|"
          + frame[[f"base__{c}" for c in cond]]
          .apply(lambda r: "|".join(f"{v:.10g}" for v in r), axis=1))
    for arm, col in (("OURS_anchored", "p_anchored"),
                     ("OURS_encoder", "p_encoder"),
                     ("OURS_anchored3D", "p_anchored3d")):
        pa = (fk + "|" + frame["metal_A"]).map(cm[col])
        pb = (fk + "|" + frame["metal_B"]).map(cm[col])
        frame[f"prediction_{arm}"] = (pa - pb).to_numpy()
    covered = frame["prediction_OURS_anchored3D"].notna()
    cov = {"cells_covered": cov_cells,
           "pairs_total": int(len(frame)),
           "pairs_covered": int(covered.sum())}
    print(f"[coverage] {cov}")

    # his A2/TP/PAIRMEAN on the same covered subset, from the repro OOF
    rep = Path(args.repro_oof)
    have_repro = rep.exists()
    sc = frame[covered].copy()
    rows, boots = [], []
    if have_repro:
        ro = pd.read_parquet(rep)
        for arm in ("A2", "A2_TP", "PAIRMEAN"):
            per_pair = (ro.groupby("pair_id")[f"prediction_{arm}"].mean())
            sc[f"prediction_{arm}"] = sc["pair_id"].map(per_pair)
        sc = sc[sc["prediction_A2"].notna()]
    arms = (["A2", "A2_TP", "PAIRMEAN"] if have_repro else []) + \
        ["OURS_anchored", "OURS_encoder", "OURS_anchored3D"]
    for arm in arms:
        rows.append(seed_metrics(sc, arm))
    L = pd.DataFrame(rows)
    print(L.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    L.to_csv(OUTD / "leaderboard_ours.csv", index=False)
    if have_repro:
        for ref, cand in (("A2", "OURS_anchored3D"),
                          ("A2_TP", "OURS_anchored3D"),
                          ("OURS_anchored", "OURS_anchored3D"),
                          ("A2", "OURS_anchored")):
            b = paired_bootstrap(sc, ref, cand)
            boots.append(b)
            print(f"[boot] {ref} vs {cand}: delta {b['delta_mean']:+.6f} "
                  f"CI [{b['ci95_low']:+.6f}, {b['ci95_high']:+.6f}] "
                  f"p_better {b['p_better']:.4f} "
                  f"({b['extractants_improved']}/{b['n_extractants']})")
        pd.DataFrame(boots).to_csv(OUTD / "paired_bootstrap_ours.csv",
                                   index=False)
    (OUTD / "coverage.json").write_text(json.dumps(cov, indent=1))
    print(f"wrote {OUTD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
