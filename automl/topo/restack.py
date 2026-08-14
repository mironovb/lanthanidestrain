#!/usr/bin/env python3
"""A3 of the post-0.313 campaign: squeeze the stack itself.

Three moves, all on the legacy 905-pair iteration population (fresh pairs are
never touched here), all nested leave-one-extractant-out so nothing is fitted
on the pairs it is scored on:

1. **Reproduce** the 3-arm pair-fitted NNLS (+0.3132) as the reference.
2. **Add the series-profile arm**: the label-side pair-identity lookup from
   ``series_shape.py`` (LOEO mean dy per pair position) enters the NNLS as a
   fourth prediction column.  This banks the ligand-independent series shape
   (+0.066 standalone) if it is complementary to the learned arms.
3. **Stratify the weights by series half** (light: l_lo <= 7, heavy >= 8).
   The decomposition shows the two halves have opposite difficulty profiles;
   a single weight vector averages them away.

Writes ``automl/reports/restack.csv``; prints every variant with R^2,
Pearson^2 (scale-free check) and the dispersion ratio.

Usage:  module load anaconda/Python-ML-2025a
        PYTHONPATH=$PWD python3 -m automl.topo.restack [--wide]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.topo.adjacent_decomposition import load_best_arms, pair_frame
from automl.topo.c6_final import REPORTS, _nnls

OUT = REPORTS / "restack.csv"


def profile_column(pf: pd.DataFrame) -> np.ndarray:
    """Nested LOEO pair-identity lookup: for each pair, the mean dy of its
    position over all OTHER extractants (fallback: global training mean)."""
    out = np.zeros(len(pf))
    for ex in pf["extractant_group"].unique():
        te = pf["extractant_group"] == ex
        tr = pf[~te]
        prof = tr.groupby("l_lo")["dy"].mean()
        const = tr["dy"].mean()
        out[te.to_numpy()] = pf.loc[te, "l_lo"].map(prof).fillna(const).to_numpy()
    return out


def nested_stack(pf: pd.DataFrame, cols: list[str],
                 strata: pd.Series | None = None) -> np.ndarray:
    """Nested LOEO NNLS over the given prediction columns; optionally with
    separate weight vectors per stratum."""
    A = pf[cols].to_numpy(float)
    dy = pf["dy"].to_numpy(float)
    grp = pf["extractant_group"].to_numpy()
    pred = np.zeros_like(dy)
    for ex in pd.unique(grp):
        te = grp == ex
        tr = ~te
        if strata is None:
            if tr.sum() < 20:
                pred[te] = A[te].mean(axis=1)
            else:
                pred[te] = A[te] @ _nnls(A[tr], dy[tr])
        else:
            s = strata.to_numpy()
            for lev in pd.unique(s):
                trs = tr & (s == lev)
                tes = te & (s == lev)
                if not tes.any():
                    continue
                if trs.sum() < 20:
                    pred[tes] = A[tes].mean(axis=1)
                else:
                    pred[tes] = A[tes] @ _nnls(A[trs], dy[trs])
    return pred


def score(dy: np.ndarray, dp: np.ndarray) -> dict[str, float]:
    out = {"r2": ev._r2(dy, dp)}
    if np.std(dp) > 0:
        out["pearson2"] = float(np.corrcoef(dy, dp)[0, 1] ** 2)
        out["disp"] = float(np.std(dp) / np.std(dy))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wide", action="store_true",
                    help="also try a wider arm pool (extra CatBoost variants + "
                         "snn plw4) -- exploratory, worth checking once")
    args = ap.parse_args()

    frames = load_best_arms()
    pf = pair_frame(frames)
    dy = pf["dy"].to_numpy(float)
    arm_cols = [c for c in pf.columns if c.startswith("dp_") and c != "dp_stack"]

    pf["dp_profile"] = profile_column(pf)
    light = (pf["l_lo"] <= 7).map({True: "light", False: "heavy"})

    variants: list[tuple[str, np.ndarray]] = []
    variants.append(("3-arm (reproduction)", nested_stack(pf, arm_cols)))
    variants.append(("3-arm + series profile", nested_stack(pf, arm_cols + ["dp_profile"])))
    variants.append(("profile alone (floor)", pf["dp_profile"].to_numpy()))
    variants.append(("3-arm, stratified light/heavy",
                     nested_stack(pf, arm_cols, strata=light)))
    variants.append(("3-arm + profile, stratified",
                     nested_stack(pf, arm_cols + ["dp_profile"], strata=light)))

    if args.wide:
        from automl.topo.c6_final import ART, align
        from automl.topo.compare_arms import attach_meta
        from automl.topo.lift_report import ensemble, load_dirs
        wide = dict(frames)
        for p in sorted((ART / "c6_partners").glob("oof_c6p_*_full.parquet")):
            name = p.stem.replace("oof_c6p_", "cpu_").replace("_full", "")
            if name not in wide:
                wide[name] = attach_meta(
                    pd.read_parquet(p).drop_duplicates("safe_exp_id")
                    .set_index("safe_exp_id"))
        for k, slot in load_dirs(["topo_c17"]).items():
            name = sorted(slot["tags"])[0].rsplit("_s", 1)[0]
            if name in ("c17_plw4", "c17_plw2") and len(slot["runs"]) >= 8:
                wide[name] = ensemble(slot["runs"])
        wide = align(wide)
        from automl.topo.c6_final import pair_matrix
        names = sorted(wide)
        wdy, WA, wgrp = pair_matrix(wide, names)
        wpf = pd.DataFrame({"extractant_group": wgrp, "dy": wdy})
        for k, n in enumerate(names):
            wpf[f"dp_{n}"] = WA[:, k]
        wcols = [f"dp_{n}" for n in names]
        pred = np.zeros_like(wdy)
        grp = wpf["extractant_group"].to_numpy()
        for ex in pd.unique(grp):
            te = grp == ex
            tr = ~te
            A = wpf[wcols].to_numpy(float)
            pred[te] = (A[te] @ _nnls(A[tr], wdy[tr])) if tr.sum() >= 20 else A[te].mean(axis=1)
        s = score(wdy, pred)
        print(f"wide pool ({len(names)} arms): R2={s['r2']:+.4f} "
              f"P2={s.get('pearson2', float('nan')):+.4f} "
              f"(n={len(wdy)} pairs; population may differ by alignment)")

    rows = []
    print(f"\n{len(pf)} pairs · reference (published) = +0.3132")
    for name, dp in variants:
        s = score(dy, dp)
        rows.append({"variant": name, **s})
        print(f"  {name:34s} R2={s['r2']:+.4f}  "
              f"P2={s.get('pearson2', float('nan')):+.4f}  "
              f"disp={s.get('disp', float('nan')):.3f}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
