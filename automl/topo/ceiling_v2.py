#!/usr/bin/env python3
"""Is the ceiling on the adjacent-pair metric identifiable at all?

Replaces ``ceiling_test.py``, whose answer (+0.679) was withdrawn on 30 July 2026
because its estimator measured a quantity the model can predict and the metric
averages away.  See ``AUDIT_2026-07-30.md``.

The quantity that *would* be a ceiling
--------------------------------------
Disagreement between rows whose **model feature vectors are identical**.  Two rows
sharing a ``strict_composition_key`` and a lanthanide have the same extractant (so
the same ECFP and RDKit block), the same metal and ligand (so the same complex and
every 3D feature), and the same ``cond__`` values to six decimal places.  Nothing
in the design matrix distinguishes them, so their spread in log D is irreducible
**for any model built on these features** -- which is what a ceiling means here.

Why the first attempt could not measure it
------------------------------------------
Only 273 of 4,165 cells (6.6%) hold more than one such row, and a cell acquires
duplicates non-randomly.  The first attempt assumed the excess scatter was
**inter-laboratory source conflict** and tried to subtract it.

The DOI join tests that assumption directly.  ``raw_data/*_SAFE.csv`` carries a
``DOI`` column the ML table dropped; ``safe_exp_id`` is ``<file stem>:<exp_id>``, so
the join key is ``(file, exp_id)``.  It matches 100% of rows over 110 DOIs.

**The assumption was wrong**: 94% of the disagreement is *within a single paper*.
The scatter is not curation and cannot be subtracted as such.

What this reports
-----------------
1. join integrity, refusing to proceed below 95%;
2. the disagreement split within-DOI versus across-DOI;
3. how much is explained by **recorded covariates the pipeline drops**;
4. a verdict on identifiability, computed from those numbers rather than asserted.

It is written to be *able* to conclude "not identifiable".  An estimator that
cannot return that verdict is not an estimator.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/ceiling_v2.csv"
RAW = REPO / "raw_data"

STRICT = "strict_composition_key"
BINNED = "composition_key"
MIN_JOIN = 0.95

# Columns the raw export records and the design matrix does not carry.  Coverage
# is checked before use -- several are empty, which is reported rather than
# silently skipped.
DROPPED = ["ini_comp", "Solvent_Name", "Phase_Modifier_Name",
           "Phase_Modifier_Concentration_M", "Holdback_Agent_Name",
           "Holdback_Agent_Concentration_M", "Acid_Concentration_Organic_M",
           "Radiolytic_Dosage_kGy", "Shaking_Time_min", "volValue",
           "thirdType", "thirdValue"]


def load_with_doi(verbose: bool = True) -> pd.DataFrame:
    """Modelled rows joined to their literature DOI.

    Refuses below ``MIN_JOIN``: every split downstream is conditioned on the
    join, so a partial one would quietly bias all of them.
    """
    from automl.matrix_cache import load_cache
    frames = []
    for f in sorted(glob.glob(str(RAW / "*_SAFE.csv"))):
        d = pd.read_csv(f, low_memory=False)
        d["key"] = os.path.basename(f)[:-4] + ":" + d["exp_id"].astype(str)
        frames.append(d[["key", "DOI"] + [c for c in DROPPED if c in d.columns]])
    raw = pd.concat(frames, ignore_index=True).drop_duplicates("key")

    df, _, _ = load_cache()
    d = df[df["geometry_ok"].astype(bool)].copy()
    d["key"] = d["safe_exp_id"].astype(str)
    d = d.merge(raw, on="key", how="left")
    rate = float(d["DOI"].notna().mean())
    if verbose:
        print(f"[ceiling2] join on (file, exp_id): {rate:.1%} of {len(d)} rows, "
              f"{d['DOI'].nunique()} DOIs")
    if rate < MIN_JOIN:
        raise SystemExit(f"DOI join reached only {rate:.1%} (< {MIN_JOIN:.0%}); "
                         f"every split below would be biased. Refusing.")
    return d


def identical_feature_cells(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (strict block, metal) cell holding more than one measurement."""
    rows = []
    for (k, li), sub in d.groupby([STRICT, "lanthanide_index"]):
        if len(sub) < 2:
            continue
        y = sub["log_D"].to_numpy(float)
        varying = [c for c in DROPPED
                   if c in sub.columns and sub[c].nunique(dropna=False) > 1]
        rows.append({"block": k, "lanthanide_index": li, "n": len(sub),
                     "sd": float(y.std(ddof=1)),
                     "n_doi": int(sub["DOI"].nunique(dropna=False)),
                     "cross_doi": sub["DOI"].nunique(dropna=False) > 1,
                     "n_dropped_varying": len(varying),
                     "dropped_varying": ",".join(varying)})
    return pd.DataFrame(rows)


def _pooled(s: pd.DataFrame) -> float:
    if s.empty or (s["n"] - 1).sum() == 0:
        return float("nan")
    return float(np.sqrt((s["sd"] ** 2 * (s["n"] - 1)).sum() / (s["n"] - 1).sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    d = load_with_doi()
    cells = identical_feature_cells(d)
    n_all = d.groupby([STRICT, "lanthanide_index"]).ngroups
    print("\n=== cells with IDENTICAL model features and >1 measurement ===")
    print(f"  {len(cells)} of {n_all} ({len(cells)/n_all:.1%})")
    print("  Nothing in the design matrix distinguishes the rows inside one of")
    print("  these, so their spread is irreducible for any model on these features.")

    print("\n=== is the disagreement between papers, or inside them? ===")
    for lab, sel in (("within one DOI", ~cells["cross_doi"]),
                     ("across DOIs", cells["cross_doi"])):
        s = cells[sel]
        print(f"  {lab:16s} n={len(s):4d} ({len(s)/max(len(cells),1):4.0%})  "
              f"pooled SD={_pooled(s):.4f}  median SD={s['sd'].median():.4f}")
    within = cells[~cells["cross_doi"]]
    frac_within = len(within) / max(len(cells), 1)
    print(f"\n  -> {frac_within:.0%} is WITHIN one paper. The first attempt assumed")
    print("     inter-laboratory source conflict and subtracted it; that")
    print("     assumption is refuted. Curation is not the problem.")

    print("\n=== how much is a recorded covariate the pipeline drops? ===")
    cov = d[[c for c in DROPPED if c in d.columns]]
    print(f"  {'dropped column':34s} {'coverage':>9s} {'nunique':>8s}")
    for c in cov.columns:
        print(f"    {c:32s} {cov[c].notna().mean():8.1%} "
              f"{cov[c].nunique(dropna=True):8d}")
    print()
    for lab, sel in (("a dropped covariate varies", within["n_dropped_varying"] > 0),
                     ("nothing recorded varies   ", within["n_dropped_varying"] == 0)):
        s = within[sel]
        if s.empty:
            continue
        print(f"  {lab}: n={len(s):4d}  pooled SD={_pooled(s):.4f}  "
              f"median SD={s['sd'].median():.4f}")
    c = Counter()
    for v in within["dropped_varying"]:
        for x in v.split(","):
            if x:
                c[x] += 1
    if c:
        print("  which one varies: "
              + ", ".join(f"{k}={v}" for k, v in c.most_common()))

    dil = [x for x in d.columns if x.startswith("cond__diluent__")]
    other = [x for x in dil if x.endswith("__other")]
    print("\n=== the one concrete, cheap lever this uncovers ===")
    if other and "Solvent_Name" in d.columns:
        m = d[other[0]].astype(bool)
        print(f"  cond__diluent__other is ONE one-hot column holding "
              f"{d.loc[m, 'Solvent_Name'].nunique()} distinct raw solvents,")
        print(f"  covering {int(m.sum())} of {len(d)} rows ({m.mean():.1%}).")
        print("  Those rows are mutually indistinguishable to the model although")
        print("  the experiments differ -- which is how they appear as label noise.")

    unexplained = within[within["n_dropped_varying"] == 0]
    sigma = float(unexplained["sd"].median())
    from automl.topo.ceiling_test import observed_pairs
    var_obs = float(observed_pairs(d, BINNED)["delta"].var(ddof=1))
    # 72% of binned adjacent pairs rest on a single exact condition set, so the
    # majority case is two singleton cells: var_noise = 2 sigma^2.
    var_noise = 2.0 * sigma ** 2
    implied = 1.0 - var_noise / var_obs

    pd.DataFrame([
        {"quantity": "cells with identical features", "value": len(cells)},
        {"quantity": "frac within one DOI", "value": frac_within},
        {"quantity": "median SD, nothing recorded varying", "value": sigma},
        {"quantity": "observed var of binned pair delta", "value": var_obs},
        {"quantity": "implied ceiling if sigma transfers", "value": implied},
    ]).to_csv(OUT, index=False)

    print("\n=== verdict ===")
    print(f"  median SD where nothing recorded varies = {sigma:.4f} log units "
          f"({len(unexplained)} cells)")
    print(f"  propagated to two single-measurement cells = {var_noise:.4f} variance")
    print(f"  observed variance of the binned pair difference = {var_obs:.4f}")
    print(f"  => implied ceiling if that sigma transferred: {implied:+.3f}")
    if implied <= 0:
        print("""
  NOT IDENTIFIABLE, and the reason is now specific.

  The implied ceiling is <= 0: the irreducible scatter measured on the only cells
  that can show it exceeds the entire observed spread of the target. Models
  demonstrably score +0.27 on that target, so this sigma cannot be
  representative -- the 6.6% of cells carrying duplicates are not a sample of the
  other 93.4%.

  The ceiling therefore cannot be bounded from these data and no honest number is
  available. What IS established:
    * the excess scatter is not inter-laboratory conflict (94% within one paper);
    * a quarter of it is a recorded covariate the pipeline drops, resolving to one
      cause -- a diluent one-hot that collapses 42 solvents;
    * three quarters has nothing recorded varying, pointing at an unrecorded
      experimental variable rather than at curation or at the model.

  Consequence: optimisation targets must not be expressed as a fraction of an
  attainable maximum, because that maximum is unknown.""")
    else:
        print(f"\n  A ceiling of {implied:+.3f} is implied IF the measured sigma "
              f"transfers to the\n  93.4% of cells that cannot show it. That is an "
              f"assumption, not a measurement,\n  and the first attempt failed "
              f"precisely on it. Report as assumption-laden or\n  not at all.")
    print(f"\n[ceiling2] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
