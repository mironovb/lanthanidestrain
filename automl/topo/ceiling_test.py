#!/usr/bin/env python3
"""WITHDRAWN 30 July 2026 -- this module's answer was wrong.  See AUDIT_2026-07-30.md.
                                     
Every estimator here fails, and the module now refuses to report a ceiling.

E2, the only one that produced a number in (0, 1], measured the SD of the pair
difference across distinct exact condition sets inside one binned block, and
treated it as irreducible.  It is not irreducible:

* the model holds **64 exact numeric `cond__` columns** and 46% of binned blocks
  (254/552) have at least one of them varying internally, so it can tell those
  rows apart; and
* ``adjacent_pair_arrays`` averages **y and p on the same grouping**, so
  per-condition variation never enters the binned metric at all.

Proof by contradiction: within the 203-pair subset E2 was measured on, the implied
ceiling is **+0.173** -- below the best model's +0.267.  Nothing exceeds a real
ceiling.  The published +0.679 divided that subset's noise variance (0.0235) by
the **full** 905-pair set's variance (0.0733); those populations differ 2.6-fold
in spread.

E1 and E3 fail for a different reason, established by the DOI join in
AUDIT_2026-07-30.md: a cell acquires duplicate rows non-randomly, and 94% of the
disagreement is *within a single paper* rather than between papers.

The module is kept, running and honest rather than deleted, because the three
failed estimators and the reason each fails are the result.

Original docstring follows.
--------------------------------------------------------------------------------

How much of the adjacent-pair metric is predictable *at all*?

Motivation
----------
Every number in this study is an R2 on the adjacent-pair separation, and the
best model reaches +0.267.  Nobody has ever asked what the maximum is.  Against
a ceiling of 1.0 that is a poor model; against a ceiling of 0.35 it is close to
optimal.  The two readings imply opposite next steps, and the paper cannot
choose between them without this number.

Why there is no single answer
-----------------------------
The metric's target is a *difference* of two cell means, and "noise" in a cell
mean is not one thing:

* **row-level noise** -- two rows recording the same metal, the same extractant
  and the same exact conditions, which still disagree (different sources,
  different digitisation, transcription);
* **condition mixing** -- the published metric blocks by ``composition_key``,
  which *bins* conditions, so one cell can average rows taken at genuinely
  different extractant concentrations.

The second is largely **common-mode**: a paper usually reports the whole
lanthanide series at each condition set, so a condition effect shifts both
metals of a pair together and cancels in the difference.  Naive propagation of
within-cell variance ignores that cancellation and produces a nonsensical
negative ceiling -- I ran exactly that and got -18.1 before noticing.

So three estimators are reported, with their assumptions stated, and the honest
answer is the range they span.  None is presented as *the* ceiling.

Estimators
----------
**E1 -- row-level split-half.**  Only cells whose rows share a
``strict_composition_key`` (identical exact conditions), so any disagreement is
pure measurement noise.  Splits each cell's rows in half, forms the pair
difference twice, and uses Var(d1 - d2)/4 as the variance of a full-data
difference.  *Optimistic*: assumes the binned key injects nothing.

**E2 -- condition-level.**  For an adjacent pair measured in the same binned
block at two or more distinct exact condition sets, how much does the
difference itself move?  This is precisely the variance the binned key averages
away and then cannot explain.  *Pessimistic for the strict key*, because part of
this is real condition dependence which a model with the conditions in its
feature vector can predict -- but it is the right number for the binned key,
where those rows collapse into one feature vector.

**E3 -- propagated.**  E1's per-row variance propagated to every pair through
its own cell counts, sigma^2 (1/n_A + 1/n_B).  Covers the 70% of pairs that E1's
subset does not, at the cost of assuming the replicated cells are representative
-- and they are visibly not: a cell acquires duplicates precisely when two
sources disagree.  Reported as a lower bound and labelled as such.

Nothing here uses a model.  These are properties of the measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/ceiling_test.csv"

BINNED = "composition_key"
STRICT = "strict_composition_key"


# ---------------------------------------------------------------------------
def observed_pairs(d: pd.DataFrame, key: str) -> pd.DataFrame:
    """Every adjacent pair, with the row count backing each of its two cells."""
    cell = (d.groupby([key, "lanthanide_index"])["log_D"]
            .agg(["mean", "size"]).reset_index())
    rows = []
    for bk, blk in cell.groupby(key, sort=False):
        if len(blk) < 2:
            continue
        m = blk["lanthanide_index"].to_numpy()
        mu = blk["mean"].to_numpy()
        n = blk["size"].to_numpy()
        i, j = np.triu_indices(len(blk), k=1)
        adj = np.abs(m[i] - m[j]) == 1
        for a, b in zip(i[adj], j[adj]):
            rows.append({"block": bk, "delta": mu[a] - mu[b],
                         "n_a": int(n[a]), "n_b": int(n[b])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def e1_row_split_half(d: pd.DataFrame, n_rep: int = 200, seed: int = 0
                      ) -> dict[str, float]:
    """Var of a full-data pair difference, from splitting genuine replicates.

    Restricted to ``strict_composition_key`` cells, so the rows being split
    really are repeat measurements of one quantity rather than measurements
    under different conditions.

    Algebra: split a cell of n rows into halves of n/2.  A half-based difference
    carries noise variance 2*(2*sigma^2/n) = 4*sigma^2/n; the full-data
    difference carries 2*sigma^2/n.  d1 and d2 use disjoint rows, so
    Var(d1 - d2) = 2 * 4*sigma^2/n = 4 * Var(full).  Hence Var(full) is a
    quarter of the observed split-half scatter, with no need to know sigma.
    """
    groups = {k: v["log_D"].to_numpy()
              for k, v in d.groupby([STRICT, "lanthanide_index"])}
    usable = []
    for bk, blk in d.groupby(STRICT, sort=False):
        mets = sorted(blk["lanthanide_index"].unique())
        for a, b in zip(mets, mets[1:]):
            if b - a != 1:
                continue
            va, vb = groups[(bk, a)], groups[(bk, b)]
            if len(va) >= 2 and len(vb) >= 2:
                usable.append((va, vb))
    if not usable:
        return {"n_pairs": 0}

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_rep):
        for va, vb in usable:
            pa, pb = rng.permutation(va), rng.permutation(vb)
            ha, hb = len(pa) // 2, len(pb) // 2
            d1 = pa[:ha].mean() - pb[:hb].mean()
            d2 = pa[ha:2 * ha].mean() - pb[hb:2 * hb].mean()
            diffs.append(d1 - d2)
    var_full = float(np.var(diffs, ddof=1)) / 4.0
    # per-row variance implied, for E3
    n_eff = np.mean([2.0 / (1.0 / len(va) + 1.0 / len(vb)) for va, vb in usable])
    return {"n_pairs": len(usable), "n_splits": len(diffs),
            "var_noise_delta": var_full,
            "sigma_row": float(np.sqrt(var_full * n_eff / 2.0)),
            "mean_harmonic_n": float(n_eff)}


def e2_condition_level(d: pd.DataFrame) -> dict[str, float]:
    """How much the same adjacent-pair difference moves across exact conditions.

    Within one *binned* block, group by ``strict_composition_key`` and form the
    difference separately for each exact condition set.  Their scatter is what
    the binned key averages into a single cell whose feature vector cannot
    distinguish them.
    """
    sds, ns, means = [], [], []
    for _bk, blk in d.groupby(BINNED, sort=False):
        sub = (blk.groupby([STRICT, "lanthanide_index"])["log_D"]
               .mean().reset_index())
        piv = sub.pivot(index=STRICT, columns="lanthanide_index",
                        values="log_D")
        if len(piv) < 2:
            continue
        cols = sorted(piv.columns)
        for a, b in zip(cols, cols[1:]):
            if b - a != 1:
                continue
            dd = (piv[a] - piv[b]).dropna()
            if len(dd) >= 2:
                sds.append(float(dd.std(ddof=1)))
                ns.append(len(dd))
                means.append(float(dd.mean()))
    if not sds:
        return {"n_pairs": 0}
    sds = np.asarray(sds); ns = np.asarray(ns)
    pooled_var = float((sds ** 2 * (ns - 1)).sum() / (ns - 1).sum())
    return {"n_pairs": len(sds),
            "var_noise_delta": pooled_var,
            "var_between_pairs": float(np.var(means, ddof=1))}


def e3_propagated(pairs: pd.DataFrame, sigma_row: float) -> dict[str, float]:
    """E1's per-row sigma applied to every pair through its own cell counts."""
    v = sigma_row ** 2 * (1.0 / pairs["n_a"] + 1.0 / pairs["n_b"])
    return {"n_pairs": int(len(pairs)), "var_noise_delta": float(v.mean())}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-rep", type=int, default=200,
                    help="random splits per replicated pair, for E1")
    args = ap.parse_args()

    from automl.matrix_cache import load_cache
    df, _, _ = load_cache()
    d = df[df["geometry_ok"].astype(bool)].copy()
    print(f"[ceiling] rows={len(d)} extractants={d['extractant_group'].nunique()}")

    rows = []
    for key in (BINNED, STRICT):
        pairs = observed_pairs(d, key)
        var_obs = float(pairs["delta"].var(ddof=1))
        print(f"\n=== {key} ===")
        print(f"  adjacent pairs = {len(pairs)}   observed Var(delta) = "
              f"{var_obs:.4f}  (SD {np.sqrt(var_obs):.3f})")
        print(f"  cells backing a pair: {(pairs['n_a'] == 1).mean() * 100:.0f}% "
              f"of left cells are a single measurement")

        e1 = e1_row_split_half(d, n_rep=args.n_rep)
        e2 = e2_condition_level(d)
        e3 = (e3_propagated(pairs, e1["sigma_row"])
              if e1.get("n_pairs") else {"n_pairs": 0})

        for name, est, note in (
            ("E1 row-level split-half", e1,
             "optimistic: pure measurement noise only"),
            ("E2 condition-level", e2,
             "right for the binned key; pessimistic for strict"),
            ("E3 propagated", e3,
             "lower bound: replicated cells are not representative"),
        ):
            if not est.get("n_pairs"):
                print(f"  {name:26s}  no usable pairs")
                continue
            vn = est["var_noise_delta"]
            ceil = 1.0 - vn / var_obs
            print(f"  {name:26s} n={est['n_pairs']:5d}  "
                  f"Var(noise)={vn:.4f}  ceiling R2 = {ceil:+.3f}   ({note})")
            rows.append({"key": key, "estimator": name, "note": note,
                         "n_pairs_used": est["n_pairs"],
                         "n_pairs_total": len(pairs),
                         "var_observed": var_obs, "var_noise": vn,
                         "ceiling_r2": ceil})

    out = pd.DataFrame(rows)
    # An estimator that returns a "ceiling" at or below zero has not measured a
    # ceiling; it has shown that its own noise model does not describe this
    # data.  Marking that mechanically, rather than averaging it into a range,
    # is the difference between a result and a number.
    # WITHDRAWN: no estimator here is valid.  E1/E3 fail on a non-representative
    # subset; E2 measures a predictable quantity rather than noise.  Marking them
    # all invalid is the correction, not a bug.
    out["valid"] = False
    out["withdrawn_reason"] = np.where(
        out["estimator"].str.startswith("E2"),
        "measures condition variation, which the model predicts and the metric averages out",
        "non-representative subset: cells acquire duplicates non-randomly")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("\n=== summary: WITHDRAWN, no ceiling is reported ===")
    for _, r in out.iterrows():
        print(f"    {r['estimator']:26s} ({r['key'][:6]}) "
              f"would have given {r['ceiling_r2']:+.3f} -- INVALID")
        print(f"        {r['withdrawn_reason']}")
    print("""
  Why every estimator fails (AUDIT_2026-07-30.md):

  E2 measured how much the pair difference moves across distinct exact condition
  sets inside one binned block, and treated that as irreducible.  It is not.  The
  model holds 64 exact numeric cond__ columns and 46% of binned blocks vary
  internally, so it can tell those rows apart -- and adjacent_pair_arrays averages
  y and p on the SAME grouping, so per-condition variation never reaches the
  metric at all.

  The contradiction that settles it: inside the 203-pair subset E2 was measured
  on, the implied ceiling is +0.173, BELOW the best model's +0.267.  Nothing
  exceeds a real ceiling.  The published +0.679 divided that subset's noise
  (0.0235) by the FULL 905-pair set's variance (0.0733) -- populations 2.6x apart
  in spread.

  E1 and E3 fail differently: a cell acquires duplicate rows non-randomly, and the
  DOI join shows 94% of the disagreement is WITHIN one paper, not between papers,
  so it is not source conflict either.  75% of those cells have nothing recorded
  varying at all.

  => The ceiling is NOT IDENTIFIABLE from this dataset.  That is the result.
     Withdrawn: "+0.679", "39% of attainable", "+0.412 headroom".
     See ceiling_v2.py for the DOI-joined attempt and what it could and could not
     establish.""")
    print(f"\n[ceiling] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
