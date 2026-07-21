#!/usr/bin/env python3
"""Does the water->octanol reorganisation block add to overall log D?

Pre-registered in ``automl/reports/WO_PREREGISTRATION.md``, committed before any
fit.  Self-contained, like ``fcnn_diagnostic.py``: it does not touch the
``dataset.py`` block system.  It loads the tabular design matrix, hstacks the
reorganisation block by build id, and runs the study's standard protocol --
leave-extractants-out folds, CatBoost (the overall-log D champion),
``full_metrics`` and the multiset cluster bootstrap in ``compare.paired_bootstrap``.

Four arms, on the identical both-solvents row subset and identical folds:

    A0  baseline_2d (ECFP + RDKit)
    A1  baseline_2d + single-solvent 3D block (feat3d__*)
    A2  baseline_2d + water<->octanol block          <- primary vs A0
    A3  baseline_2d + feat3d + water<->octanol block <- secondary vs A1

Primary endpoint  : A2 - A0 overall log D R^2  (does the block add over 2D?)
Secondary endpoint: A3 - A1                    (does it add beyond feat3d?)

baseline_2d already contains ``Ionic Radius_metal`` and ``lanthanide_index``, so
A2 - A0 controls for the ionic radius -- any gain is beyond what the metal's size
already provides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl import models as mz
from automl.compare import paired_bootstrap
from automl.dataset import BLOCK_PRESETS, GROUP_COL, TARGET
from automl.matrix_cache import load_cache

REPO = Path(__file__).resolve().parents[2]
WO = REPO / "automl/artifacts/water_octanol/features.parquet"
OUT = REPO / "automl/reports/wo_test.csv"


def _oof_frame(df, X, groups, y, *, folds, repeats, seed, n_jobs) -> np.ndarray:
    """Averaged out-of-fold predictions, leave-extractants-out, like the sweep."""
    oof_sum = np.zeros(len(df)); oof_cnt = np.zeros(len(df))
    for rep in range(repeats):
        for tr, te in ev.grouped_folds(groups, n_splits=folds, seed=seed + rep):
            m = mz.make_model("catboost", seed=seed + rep, n_jobs=n_jobs)
            m.fit(X[tr], y[tr])
            oof_sum[te] += np.asarray(m.predict(X[te]), dtype=float)
            oof_cnt[te] += 1
    return oof_sum / np.maximum(oof_cnt, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    if not WO.exists():
        print(f"missing {WO}; run automl.qc.water_octanol_features first")
        return 1
    wo = pd.read_parquet(WO).drop_duplicates("geometry_feature_build_id") \
        .set_index("geometry_feature_build_id")
    wo_cols = [c for c in wo.columns if c.startswith("wo_")]

    df, blocks, _ = load_cache()
    key = df["geometry_feature_build_id"].astype(str)
    df = df.assign(_bid=key)
    # The both-solvents subset: every arm is scored on exactly these rows, so the
    # comparison is never contaminated by a differing row set.
    sub = df[df["_bid"].isin(wo.index)].reset_index(drop=True)
    print(f"both-solvents subset: {len(sub)} rows, "
          f"{sub[GROUP_COL].nunique()} extractants, "
          f"{sub['_bid'].nunique()} complexes", flush=True)

    base_cols = blocks.select(BLOCK_PRESETS["baseline_2d"])
    feat3d_cols = [c for c in df.columns if c.startswith("feat3d__")]
    Xbase = sub[base_cols].to_numpy(np.float32)
    X3d = sub[feat3d_cols].to_numpy(np.float32)
    Xwo = sub.join(wo, on="_bid")[wo_cols].to_numpy(np.float32)

    arms = {
        "A0_baseline2d": Xbase,
        "A1_plus_feat3d": np.hstack([Xbase, X3d]),
        "A2_plus_wo": np.hstack([Xbase, Xwo]),
        "A3_feat3d_plus_wo": np.hstack([Xbase, X3d, Xwo]),
    }
    y = sub[TARGET].to_numpy(float)
    groups = sub[GROUP_COL].to_numpy()
    meta = pd.DataFrame({
        "safe_exp_id": sub["safe_exp_id"].to_numpy(), "y": y,
        "extractant_group": groups,
        "composition_key": sub["composition_key"].to_numpy(),
    }).set_index("safe_exp_id")

    oof = {}
    print("\n=== per-arm overall log D (leave-extractants-out) ===")
    rows = []
    for name, X in arms.items():
        p = _oof_frame(sub, X, groups, y, folds=args.folds, repeats=args.repeats,
                       seed=args.seed, n_jobs=args.n_jobs)
        oof[name] = meta.assign(oof=p)
        m = ev.full_metrics(y, p, sub)
        rows.append({"arm": name, "n_feat": X.shape[1], **m})
        print(f"  {name:20s} nfeat={X.shape[1]:4d}  R2={m['r2_overall']:+.4f}  "
              f"within={m.get('r2_within', float('nan')):+.4f}  "
              f"between={m.get('r2_between', float('nan')):+.4f}", flush=True)

    print("\n=== pre-registered endpoints (paired cluster bootstrap over extractants) ===")
    tests = [("primary", "A0_baseline2d", "A2_plus_wo",
              "water<->octanol block over 2D"),
             ("secondary", "A1_plus_feat3d", "A3_feat3d_plus_wo",
              "water<->octanol block beyond feat3d")]
    out = []
    for kind, a, b, q in tests:
        res = paired_bootstrap(oof[a], oof[b], n_boot=args.n_boot, seed=0)
        d, lo, hi, P = res["r2_overall"]
        verdict = ("adds" if lo > 0 else "worse" if hi < 0 else "not distinguishable")
        star = "**" if kind == "primary" else "  "
        print(f"{star}{b} - {a}:  dR2 = {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"P = {P:.2f}  {verdict}   | {q}")
        out.append({"kind": kind, "base": a, "arm": b, "question": q,
                    "delta": d, "lo": lo, "hi": hi, "p_better": P,
                    "verdict": verdict})
        # the within-extractant component, where a per-complex feature can help
        dw, lw, hw, Pw = res["r2_within"]
        print(f"     within-extractant dR2 = {dw:+.4f} [{lw:+.4f}, {hw:+.4f}]  P = {Pw:.2f}")
        out[-1].update({"within_delta": dw, "within_lo": lw, "within_hi": hw})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT.with_name("wo_cells.csv"), index=False)
    pd.DataFrame(out).to_csv(OUT, index=False)
    print(f"\n[wo-test] wrote {OUT} and wo_cells.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
