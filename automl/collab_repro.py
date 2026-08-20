#!/usr/bin/env python3
"""Reproduce the collaborator's headline metrics from his 2026-08-18 spec.

The collaborator's repo (branch ``descriptor-arm-metal-site``) is not on this
machine; ``collaborator_update/`` holds only his dataset.parquet (SHA-256
verified against the hash pinned in his document), 1,155 geometries and the
reproduction instructions (``metrics_reproduction_20260818.md``, in Russian).
This module is a from-scratch reimplementation of his §§2-6:

  cohort   5,992 rows -> quarantine(-129) -> complete-conditions(-2,851)
           -> 2,520 cells (median-aggregated) -> exact-condition pairs,
           A lighter than B -> geometry_ok both / replicate-unique /
           3D-complete -> 6,699 pairs / 34 extractants   (his audit numbers)
  model    A2 = 64 cond + 8 Ln + 10 RDKit + 2,048 ECFP = 2,130 columns;
           AntisymmetricExtraTrees (impute-median+indicator -> 200 trees),
           antisymmetrised train (reversed copies, negated target),
           prediction (f(A,B) - f(B,A))/2, group-balanced weights;
           StratifiedGroupKFold(5) by pair_label / extractant, inner 3-fold
           grid {(0.35,2),(0.70,2),(1.00,4)} on inner macro MAE;
           5 split seeds 104729/130363/155921/196613/262147, model seed 42
  extras   PAIRMEAN baseline (leave-fold-out pair_label table) and
           transitive projection (per-(extractant, condition) lstsq)
  metrics  equal-extractant macro MAE (primary), pooled MAE/R2, sign acc,
           dispersion ratio, adjacent/nonadjacent MAE; extractant-resampled
           paired bootstrap (rng 8675309, 10,000 replicates)

Expected (his gen4_candidates_20260817T003449Z): A2 macro 0.3192 +- 0.0078,
A2+TP 0.3175, PAIRMEAN 0.4482; tolerance per his §8 is the bootstrap CI, not
bitwise (ExtraTrees thread nondeterminism ~3rd-4th decimal).

Usage:
  PYTHONPATH=$PWD python3 -m automl.collab_repro --quick        # smoke
  PYTHONPATH=$PWD python3 -m automl.collab_repro                # full 5 seeds
Outputs under automl/reports/collab_repro/: cohort_audit.json,
leaderboard.csv, per_seed_metrics.csv, paired_bootstrap.csv,
oof_predictions.parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "collaborator_update/dataset.parquet"
OUTD = REPO / "automl/reports/collab_repro"

SPLIT_SEEDS = [104729, 130363, 155921, 196613, 262147]
MODEL_SEED = 42
GRID = [(0.35, 2), (0.70, 2), (1.00, 4)]
RDKIT10 = ["MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
           "NumAromaticRings", "NumAliphaticRings", "RingCount",
           "FractionCSP3", "MolLogP"]
LN_ORDER = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
            "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
Z_OF = {m: 57 + i for i, m in enumerate(LN_ORDER)}


# --------------------------------------------------------------------------
# cohort (his §2)
# --------------------------------------------------------------------------
def build_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    audit: dict = {"start_rows": len(df)}
    name = df["extractant_name"].astype(str).str.upper()
    todga_smiles = df.loc[name == "TODGA", "canonical_smiles"].mode().iloc[0]
    q = (df["canonical_smiles"] == todga_smiles) & (name != "TODGA")
    audit["quarantined"] = int(q.sum())
    df = df[~q]
    df = df[df["metal"].isin(LN_ORDER) & np.isfinite(df["log_D"])]
    cond = sorted(c for c in df.columns if c.startswith("cond__"))
    complete = df[cond].notna().all(axis=1)
    audit["dropped_nan_conditions"] = int((~complete).sum())
    df = df[complete]
    audit["cohort_rows"] = len(df)

    # proxy for his "compact-invariant" 3D set: feat3d columns that are not
    # slot-indexed/CN-dependent (donor_09 etc. are NaN by construction on CN-8)
    # and not entirely empty -- i.e. near-complete on geometry_ok rows
    all3d = sorted(c for c in df.columns if c.startswith("feat3d__"))
    ok_rows = df[df["geometry_ok"].astype(bool)]
    feat3d = [c for c in all3d if ok_rows[c].isna().mean() < 0.005]
    grp_cols = ["canonical_smiles"] + cond
    agg = {c: "median" for c in feat3d}
    agg.update({"log_D": "median", "Ionic Radius_metal": "median",
                "geometry_ok": "all"})
    agg.update({c: "first" for c in RDKIT10})
    agg.update({c: "first" for c in df.columns if c.startswith("ecfp_")})
    cells = (df.groupby(grp_cols + ["metal"], dropna=False, sort=False)
             .agg(agg).reset_index())
    cells["n_replicates"] = (df.groupby(grp_cols + ["metal"], dropna=False,
                                        sort=False).size().to_numpy())
    audit["cells"] = len(cells)
    audit["cells_replicated"] = int((cells["n_replicates"] > 1).sum())

    ecfp = sorted((c for c in df.columns if c.startswith("ecfp_")),
                  key=lambda s: int(s.split("_")[1]))
    rows = []
    n_cand = n_geo = n_rep = n_f3d = 0
    for key, blk in cells.groupby(grp_cols, dropna=False, sort=False):
        if len(blk) < 2:
            continue
        smiles = key[0]
        cond_vals = dict(zip(cond, key[1:]))
        cid = hashlib.sha256(
            (smiles + "|" + "|".join(f"{v:.10g}" for v in key[1:]))
            .encode()).hexdigest()[:20]
        blk = blk.sort_values("metal", key=lambda s: s.map(Z_OF))
        rec = blk.to_dict("records")
        for a in range(len(rec)):
            for b in range(a + 1, len(rec)):
                A, B = rec[a], rec[b]
                n_cand += 1
                if not (A["geometry_ok"] and B["geometry_ok"]):
                    n_geo += 1
                    continue
                if A["n_replicates"] != 1 or B["n_replicates"] != 1:
                    n_rep += 1
                    continue
                if any(pd.isna(A[c]) for c in feat3d) or \
                   any(pd.isna(B[c]) for c in feat3d):
                    n_f3d += 1
                    continue
                row = {"extractant": smiles, "condition_id": cid,
                       "metal_A": A["metal"], "metal_B": B["metal"],
                       "pair_label": f"{A['metal']}-{B['metal']}",
                       "log_SF_A_over_B": A["log_D"] - B["log_D"],
                       "pair__Z_A": Z_OF[A["metal"]],
                       "pair__Z_B": Z_OF[B["metal"]],
                       "pair__ionic_radius_A": A["Ionic Radius_metal"],
                       "pair__ionic_radius_B": B["Ionic Radius_metal"]}
                row["pair__Z_mean"] = 0.5 * (row["pair__Z_A"] + row["pair__Z_B"])
                row["pair__delta_Z"] = row["pair__Z_B"] - row["pair__Z_A"]
                row["pair__ionic_radius_mean"] = 0.5 * (
                    row["pair__ionic_radius_A"] + row["pair__ionic_radius_B"])
                row["pair__delta_ionic_radius"] = (
                    row["pair__ionic_radius_B"] - row["pair__ionic_radius_A"])
                row["pair_id"] = hashlib.sha256(
                    f"{cid}|{A['metal']}|{B['metal']}".encode()).hexdigest()[:20]
                for c in cond:
                    row[f"base__{c}"] = cond_vals[c]
                for c in RDKIT10:
                    row[f"base__{c}"] = A[c]
                for c in ecfp:
                    row[f"base__{c}"] = A[c]
                rows.append(row)
    audit.update({"candidate_pairs": n_cand, "dropped_geometry": n_geo,
                  "dropped_replicates": n_rep, "dropped_feat3d": n_f3d})
    frame = pd.DataFrame(rows)
    audit["pairs"] = len(frame)
    audit["extractants"] = int(frame["extractant"].nunique())
    audit["condition_ids"] = int(frame["condition_id"].nunique())
    audit["pair_labels"] = int(frame["pair_label"].nunique())
    return frame, audit


def a2_columns(frame: pd.DataFrame) -> list[str]:
    cond = sorted(c for c in frame.columns if c.startswith("base__cond__"))
    ln = ["pair__Z_A", "pair__Z_B", "pair__Z_mean", "pair__delta_Z",
          "pair__ionic_radius_A", "pair__ionic_radius_B",
          "pair__ionic_radius_mean", "pair__delta_ionic_radius"]
    rd = [f"base__{c}" for c in RDKIT10]
    ec = sorted((c for c in frame.columns if c.startswith("base__ecfp_")),
                key=lambda s: int(s.rsplit("_", 1)[1]))
    return cond + ln + rd + ec


# --------------------------------------------------------------------------
# model (his §4)
# --------------------------------------------------------------------------
def reverse_features(X: pd.DataFrame) -> pd.DataFrame:
    R = X.copy()
    for a in [c for c in X.columns if c.endswith("_A")]:
        b = a[:-2] + "_B"
        R[a], R[b] = X[b], X[a]
    for d in [c for c in X.columns if c.startswith("pair__delta_")]:
        R[d] = -X[d]
    return R


def make_est(max_features, min_leaf, seed, n_jobs):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", ExtraTreesRegressor(
            n_estimators=200, max_features=max_features,
            min_samples_leaf=min_leaf, random_state=seed, n_jobs=n_jobs))])


def group_weights(extractant: pd.Series) -> np.ndarray:
    counts = extractant.value_counts()
    n, g = len(extractant), extractant.nunique()
    return (n / (g * extractant.map(counts))).to_numpy(float)


def fit_predict_antisym(Xtr, ytr, wtr, Xte, params, seed, n_jobs):
    mf, ml = params
    est = make_est(mf, ml, seed, n_jobs)
    Xaug = pd.concat([Xtr, reverse_features(Xtr)], ignore_index=True)
    yaug = np.concatenate([ytr, -ytr])
    waug = np.concatenate([wtr, wtr])
    est.fit(Xaug, yaug, model__sample_weight=waug)
    return 0.5 * (est.predict(Xte) - est.predict(reverse_features(Xte)))


def macro_mae(y, p, extractant) -> float:
    d = pd.DataFrame({"e": extractant, "a": np.abs(np.asarray(y) - np.asarray(p))})
    return float(d.groupby("e")["a"].mean().mean())


def run_seed(frame: pd.DataFrame, cols: list[str], split_seed: int,
             n_jobs: int, n_estimators_quick: bool = False) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold
    X = frame[cols]
    y = frame["log_SF_A_over_B"].to_numpy(float)
    grp = frame["extractant"]
    strat = frame["pair_label"]
    out = frame[["pair_id", "extractant", "condition_id", "pair_label",
                 "metal_A", "metal_B", "pair__Z_A", "pair__Z_B",
                 "log_SF_A_over_B"]].copy()
    out["split_seed"] = split_seed
    out["outer_fold"] = -1
    out["prediction_A2"] = np.nan
    out["prediction_PAIRMEAN"] = np.nan

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                 random_state=split_seed)
    for k, (tr, te) in enumerate(outer.split(X, strat, grp)):
        # inner CV over the grid
        best, best_mae = GRID[0], np.inf
        inner = StratifiedGroupKFold(
            n_splits=3, shuffle=True,
            random_state=split_seed + k * 10_007)
        for params in GRID:
            maes = []
            for itr, ite in inner.split(X.iloc[tr], strat.iloc[tr],
                                        grp.iloc[tr]):
                itr_g, ite_g = tr[itr], tr[ite]
                p = fit_predict_antisym(
                    X.iloc[itr_g], y[itr_g],
                    group_weights(grp.iloc[itr_g]), X.iloc[ite_g],
                    params, MODEL_SEED, n_jobs)
                maes.append(macro_mae(y[ite_g], p, grp.iloc[ite_g]))
            m = float(np.mean(maes))
            if m < best_mae:
                best_mae, best = m, params
        seed = MODEL_SEED + k * 1009 + 9_999_991
        p = fit_predict_antisym(X.iloc[tr], y[tr],
                                group_weights(grp.iloc[tr]),
                                X.iloc[te], best, seed, n_jobs)
        out.iloc[te, out.columns.get_loc("prediction_A2")] = p
        out.iloc[te, out.columns.get_loc("outer_fold")] = k
        # PAIRMEAN baseline: train-fold mean y per (pair_label, extractant),
        # averaged over extractants -> per-pair_label table
        trd = frame.iloc[tr]
        tab = (trd.groupby(["pair_label", "extractant"])["log_SF_A_over_B"]
               .mean().groupby("pair_label").mean())
        pm = frame.iloc[te]["pair_label"].map(tab)
        pm = pm.fillna(trd["log_SF_A_over_B"].mean())
        out.iloc[te, out.columns.get_loc("prediction_PAIRMEAN")] = pm.to_numpy()
        print(f"    fold {k}: n_te={len(te)} params={best} "
              f"inner_macro={best_mae:.4f}", flush=True)

    # transitive projection on the completed OOF frame
    out["prediction_A2_TP"] = out["prediction_A2"]
    for (_, _), g in out.groupby(["extractant", "condition_id"]):
        if len(g) < 2:
            continue
        metals = sorted(set(g["metal_A"]) | set(g["metal_B"]))
        mi = {m: i for i, m in enumerate(metals)}
        M = np.zeros((len(g), len(metals)))
        for r, (_, row) in enumerate(g.iterrows()):
            M[r, mi[row["metal_A"]]] = 1.0
            M[r, mi[row["metal_B"]]] = -1.0
        s, *_ = np.linalg.lstsq(M, g["prediction_A2"].to_numpy(), rcond=None)
        out.loc[g.index, "prediction_A2_TP"] = M @ s
    return out


# --------------------------------------------------------------------------
# metrics (his §6)
# --------------------------------------------------------------------------
def seed_metrics(oof: pd.DataFrame, arm: str) -> dict:
    from sklearn.metrics import r2_score
    y = oof["log_SF_A_over_B"].to_numpy(float)
    p = oof[f"prediction_{arm}"].to_numpy(float)
    e = oof["extractant"]
    adj = (oof["pair__Z_B"] - oof["pair__Z_A"]) == 1
    return {"arm": arm,
            "macro_mae": macro_mae(y, p, e),
            "pooled_mae": float(np.mean(np.abs(y - p))),
            "pooled_r2": float(r2_score(y, p)),
            "sign_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
            "dispersion_ratio": float(np.std(p) / np.std(y)),
            "adjacent_mae": float(np.mean(np.abs(y - p)[adj])),
            "nonadjacent_mae": float(np.mean(np.abs(y - p)[~adj]))}


def paired_bootstrap(oof: pd.DataFrame, ref: str, cand: str,
                     n: int = 10_000) -> dict:
    y = oof["log_SF_A_over_B"].to_numpy(float)
    d = pd.DataFrame({
        "e": oof["extractant"],
        "ref": np.abs(y - oof[f"prediction_{ref}"].to_numpy(float)),
        "cand": np.abs(y - oof[f"prediction_{cand}"].to_numpy(float))})
    per = d.groupby("e")[["ref", "cand"]].mean()
    delta = (per["ref"] - per["cand"]).to_numpy()
    rng = np.random.default_rng(8675309)
    stats = np.array([delta[rng.integers(0, len(delta), size=len(delta))]
                      .mean() for _ in range(n)])
    return {"reference": ref, "candidate": cand,
            "delta_mean": float(delta.mean()),
            "ci95_low": float(np.quantile(stats, 0.025)),
            "ci95_high": float(np.quantile(stats, 0.975)),
            "p_better": float(np.mean(stats > 0)),
            "extractants_improved": int((delta > 0).sum()),
            "n_extractants": len(delta)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="one seed, no inner CV grid (first params)")
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=SPLIT_SEEDS)
    args = ap.parse_args()

    OUTD.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert sha == ("fefbefc6fe993aa9ce9db1a0c338adb9e5f58a8b75bab084cc1df4e0"
                   "24faf5dd"), f"dataset hash mismatch: {sha}"
    df = pd.read_parquet(DATA)
    frame, audit = build_cohort(df)
    cols = a2_columns(frame)
    audit["a2_columns"] = len(cols)
    (OUTD / "cohort_audit.json").write_text(json.dumps(audit, indent=1))
    print("[cohort]", json.dumps(audit), flush=True)

    global GRID
    seeds = args.seeds[:1] if args.quick else args.seeds
    if args.quick:
        GRID = GRID[:1]

    oofs = []
    for s in seeds:
        print(f"[seed {s}]", flush=True)
        oofs.append(run_seed(frame, cols, s, args.n_jobs))
    oof = pd.concat(oofs, ignore_index=True)
    oof.to_parquet(OUTD / "oof_predictions.parquet", index=False)

    per_seed, leader = [], []
    for arm in ("A2", "A2_TP", "PAIRMEAN"):
        rows = [dict(seed_metrics(g, arm), split_seed=s)
                for s, g in oof.groupby("split_seed")]
        per_seed.extend(rows)
        agg = pd.DataFrame(rows).drop(columns=["arm", "split_seed"]).mean()
        leader.append({"arm": arm, **agg.to_dict(),
                       "macro_mae_split_sd": float(
                           pd.DataFrame(rows)["macro_mae"].std())})
    pd.DataFrame(per_seed).to_csv(OUTD / "per_seed_metrics.csv", index=False)
    L = pd.DataFrame(leader)
    L.to_csv(OUTD / "leaderboard.csv", index=False)
    print(L.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    boots = [paired_bootstrap(oof, "A2", "A2_TP"),
             paired_bootstrap(oof, "A2", "PAIRMEAN"),
             paired_bootstrap(oof, "PAIRMEAN", "A2")]
    pd.DataFrame(boots).to_csv(OUTD / "paired_bootstrap.csv", index=False)
    for b in boots:
        print(f"[boot] {b['reference']} vs {b['candidate']}: "
              f"delta {b['delta_mean']:+.6f} CI [{b['ci95_low']:+.6f}, "
              f"{b['ci95_high']:+.6f}] p_better {b['p_better']:.4f}")
    print("\nexpected (his run): A2 macro 0.319221+-0.007794, "
          "A2_TP 0.317464, PAIRMEAN 0.448181")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
