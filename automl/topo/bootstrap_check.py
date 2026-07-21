#!/usr/bin/env python3
"""Audit of the resampling behind every interval in this study.

**Methodological audit. Changes no reported number; it measures how wide the
reported intervals are relative to a correct cluster bootstrap.**

What was found
--------------
``adjacent_test.paired_adjacent`` resamples whole extractants with replacement
and then scores with ``evaluation.adjacent_pair_metrics``, which begins

    frame.groupby("c")                  # c = composition_key
        .groupby("m").mean()            # one point per (block, metal)

Composition keys are nested inside extractants, so when the bootstrap draws an
extractant twice, its rows carry *the same* composition_key both times, the
groupby merges them, and the per-metal averaging returns exactly the values it
would have returned from one copy.  Verified directly: duplicating a cluster
leaves the statistic bit-identical.

The draw is therefore not a multiset of extractants but the **set** of those
drawn at least once -- on average 1 - 1/e = 63.2 % of them.  That is an
m-out-of-n subsampling bootstrap with random m, not a cluster bootstrap.

Which direction it errs in -- measured, not reasoned
----------------------------------------------------
I predicted the intervals would come out too **wide**: fewer effective clusters
per draw ought to mean more variance per draw, making every published interval
conservative.  **That prediction was wrong.**  Measured on the primary endpoint,
the published intervals are **0.88x** the width of the multiplicity-respecting
ones -- about 12 % too *narrow*, so they mildly **overstate** significance
rather than understating it.

The reason, in hindsight: collapsing gives every present cluster equal weight,
which is a lower-variance statistic than one where a twice-drawn cluster counts
twice.  Both are legitimate estimators; only one is the cluster bootstrap the
methods section claims.

So this has to be reported as a correction, and every "90 % interval excludes
zero" claim has to be re-checked against the corrected resampling rather than
waved through.  That check is what ``main`` below does for the three intervals
the study leans on hardest.

The correction
--------------
Make each drawn copy its own block by suffixing the copy index onto the
composition key, so a twice-drawn extractant contributes two independent blocks
and multiplicity is respected.  Everything else is unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from automl.topo.adjacent_test import adj_r2
from automl.topo.compare_arms import attach_meta, collect
from automl.topo.control_factorial import ensemble, load_cells

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/bootstrap_audit.csv"


def _resample(rows_by_g, pick):
    """Row indices plus a per-copy tag, so duplicates can be kept distinct."""
    idx, tag = [], []
    for copy, g in enumerate(pick):
        r = rows_by_g[g]
        idx.append(r)
        tag.append(np.full(len(r), copy))
    return np.concatenate(idx), np.concatenate(tag)


def compare(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, seed: int = 0) -> dict:
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    y = a["y"].to_numpy(float)
    pa, pb = a["oof"].to_numpy(float), b["oof"].to_numpy(float)
    comp = a["composition_key"].to_numpy().astype(str)
    li = a["lanthanide_index"].to_numpy()
    gcodes, guniq = pd.factorize(a["extractant_group"].to_numpy())
    rows_by_g = [np.flatnonzero(gcodes == i) for i in range(len(guniq))]

    rng = np.random.default_rng(seed)
    as_pub, as_fix, n_eff = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_by_g), len(rows_by_g))
        idx, tag = _resample(rows_by_g, pick)
        n_eff.append(len(np.unique(pick)))
        # as published: shared composition keys, so duplicates collapse
        as_pub.append(adj_r2(y[idx], pb[idx], comp[idx], li[idx])
                      - adj_r2(y[idx], pa[idx], comp[idx], li[idx]))
        # corrected: each drawn copy is its own block
        ck = np.char.add(np.char.add(comp[idx], "#"), tag.astype(str))
        as_fix.append(adj_r2(y[idx], pb[idx], ck, li[idx])
                      - adj_r2(y[idx], pa[idx], ck, li[idx]))

    def summarise(d):
        d = np.asarray([v for v in d if np.isfinite(v)])
        lo, hi = float(np.percentile(d, 5)), float(np.percentile(d, 95))
        return {"delta": float(d.mean()), "lo": lo, "hi": hi,
                "width": hi - lo, "p_better": float((d > 0).mean())}

    pub, fix = summarise(as_pub), summarise(as_fix)
    return {"n_clusters": len(rows_by_g),
            "mean_distinct_clusters_per_draw": float(np.mean(n_eff)),
            "frac_of_clusters": float(np.mean(n_eff)) / len(rows_by_g),
            **{f"pub_{k}": v for k, v in pub.items()},
            **{f"fix_{k}": v for k, v in fix.items()},
            "width_ratio_pub_over_fix": pub["width"] / max(fix["width"], 1e-12)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    members = load_cells(verbose=False)
    ens = {c: ensemble(m) for c, m in members.items()}
    refs = {k: attach_meta(v) for k, v in collect().items()
            if k.startswith("baseline::")}
    ens["FCNN"] = refs.get("baseline::mlp::none")
    ens["CAT"] = refs.get("baseline::catboost::none")

    pairs = [("T0", "S0", "primary endpoint: topology on top of the objective"),
             ("FCNN", "S0", "the published headline"),
             ("CAT", "S0", "SNN ensemble vs CatBoost, as published")]
    rows = []
    for base, arm, label in pairs:
        if ens.get(base) is None or ens.get(arm) is None:
            print(f"  {arm} vs {base}: cell missing -- skipped")
            continue
        r = compare(ens[base], ens[arm], args.n_boot)
        r.update({"base": base, "arm": arm, "label": label})
        rows.append(r)
        print(f"\n{label}")
        print(f"  as published  delta {r['pub_delta']:+.4f}  "
              f"[{r['pub_lo']:+.4f}, {r['pub_hi']:+.4f}]  width {r['pub_width']:.4f}")
        print(f"  multiplicity  delta {r['fix_delta']:+.4f}  "
              f"[{r['fix_lo']:+.4f}, {r['fix_hi']:+.4f}]  width {r['fix_width']:.4f}")
        print(f"  published intervals are {r['width_ratio_pub_over_fix']:.2f}x "
              f"as wide; each draw keeps "
              f"{r['frac_of_clusters']:.1%} of the {r['n_clusters']} extractants")

    if not rows:
        return 1
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n[bootstrap-audit] wrote {OUT}")
    # State the direction from the numbers, not from a prior.  The first version
    # of this line asserted "a ratio above 1 means the intervals are
    # conservative", which was my prediction and was wrong in every case.
    ratios = [r["width_ratio_pub_over_fix"] for r in rows]
    narrow = [r for r in rows if r["width_ratio_pub_over_fix"] < 1]
    still = [r for r in rows if (r["fix_lo"] > 0) == (r["pub_lo"] > 0)]
    print(f"published/corrected width: {min(ratios):.2f}x to {max(ratios):.2f}x "
          f"({len(narrow)}/{len(rows)} too NARROW, i.e. overstating significance)")
    print(f"{len(still)}/{len(rows)} intervals reach the same verdict on "
          f"excluding zero after correction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
