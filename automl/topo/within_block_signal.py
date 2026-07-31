#!/usr/bin/env python3
"""Does a feature block carry signal in the direction the metric measures?

Why this exists
---------------
``sel_adj_logSF_r2`` scores the *difference* between adjacent lanthanides
inside a composition block.  A feature is therefore useful to it only if the
feature's own within-block difference tracks ``dy``.  A feature can be highly
informative about log_D overall and still be worthless -- or harmful -- here.

Sweep2 cell A1 made that concrete: adding 119 well-populated angular and
polyhedral columns to the tabular head moved overall R2 by about -0.10 but the
adjacent-pair metric by -0.32.  This module measures the quantity that
distinguishes those two outcomes, so the explanation is a measurement rather
than a story told after the fact.

Method: reconstruct the metric's own pairs -- group by ``composition_key``,
average per lanthanide, take neighbours -- then for each feature compute the
same within-block difference and correlate it with ``dy``.  The reconstruction
is checked against the metric's published pair count and moments before any
correlation is reported, so a silent mismatch cannot produce a confident number.

Usage
-----
    python3 -m automl.topo.within_block_signal
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"
OUT = REPORTS / "within_block_signal.csv"

N_PAIRS = 905            # the metric's own pair count, from evaluation.py
MEAN_DY = -0.0724        # published; see REANALYSIS_2026-07-29.md part 1


def _pairs(df: pd.DataFrame):
    d = df[["composition_key", "lanthanide_index", "log_D"]].copy()
    d["_r"] = np.arange(len(d))
    out = []
    for _, sub in d.groupby("composition_key"):
        m = {int(k): (v["log_D"].mean(), v["_r"].to_numpy())
             for k, v in sub.groupby("lanthanide_index")}
        ks = sorted(m)
        for i in range(len(ks) - 1):
            if ks[i + 1] - ks[i] == 1:
                out.append((m[ks[i]][1], m[ks[i + 1]][1],
                            m[ks[i]][0] - m[ks[i + 1]][0]))
    return out


def _stats(rows, X, idx, names, dy):
    """Correlations of each feature's within-block difference with dy.

    Three populations have to be kept apart, because conflating them
    miscounts badly: columns dropped because some pair difference is NaN,
    columns that are genuinely CONSTANT within every block, and columns that
    vary.  Only the last can be correlated with dy at all.
    """
    D = np.empty((len(rows), len(idx)))
    for n, (ra, rb, _) in enumerate(rows):
        D[n] = np.nanmean(X[np.ix_(ra, idx)], 0) - np.nanmean(X[np.ix_(rb, idx)], 0)
    finite = np.isfinite(D)
    n_nan = int((~finite.all(0)).sum())
    # block-constant := the within-block adjacent difference is zero wherever
    # it is defined.  NaN entries are not evidence of variation.
    maxabs = np.where(finite, np.abs(D), 0.0).max(axis=0)
    const = maxabs <= 1e-9
    keep = (~const) & finite.all(0)
    Dv, nm = D[:, keep], [n for n, k in zip(names, keep) if k]
    r = np.array([np.corrcoef(Dv[:, j], dy)[0, 1] for j in range(Dv.shape[1])])
    return r, nm, Dv.shape[1], int(const.sum()), n_nan


def main() -> int:
    from automl.topo.train import build_row_table

    df, Xb, cb = build_row_table(preset="baseline_2d", arch="snn")
    _, Xs, cs = build_row_table(preset="baseline_2d_shape", arch="snn")

    rows = _pairs(df)
    dy = np.array([r[2] for r in rows])
    # Refuse to report correlations off a reconstruction that does not match
    # the metric.  A quiet mismatch here would look like a finding.
    if len(rows) != N_PAIRS or abs(dy.mean() - MEAN_DY) > 5e-4:
        raise SystemExit(
            f"pair reconstruction does not match the metric: got {len(rows)} "
            f"pairs (mean dy {dy.mean():+.4f}), expected {N_PAIRS} "
            f"({MEAN_DY:+.4f}). Fix this before trusting any number below.")

    added = [c for c in cs if c not in set(cb)]
    cond = [c for c in cb if c.startswith("cond__")]
    blocks = [("geometry (A1's 119 added columns)", Xs,
               [cs.index(c) for c in added], added),
              ("published cond__", Xb, [cb.index(c) for c in cond], cond)]

    recs = []
    for label, X, idx, names in blocks:
        r, nm, n_live, n_const, n_nan = _stats(rows, X, idx, names, dy)
        a = np.abs(r)
        recs.append(dict(block=label, n_columns=len(names),
                         n_varying_within_block=n_live,
                         n_block_constant=n_const,
                         n_dropped_nan=n_nan,
                         frac_varying=n_live / len(names),
                         median_abs_corr_with_dy=float(np.median(a)),
                         p90_abs_corr=float(np.percentile(a, 90)),
                         max_abs_corr=float(a.max()),
                         best_column=nm[int(np.argmax(a))]))

    out = pd.DataFrame(recs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"pairs {len(rows)}  mean(dy) {dy.mean():+.4f}  SD {dy.std():.4f}  "
          f"(matches the metric)\n")
    print(out.to_string(index=False))

    g, c = recs[0], recs[1]
    print(f"\n{g['n_varying_within_block']} of {g['n_columns']} geometry columns "
          f"({g['frac_varying']:.0%}) differ between adjacent lanthanides inside a "
          f"block, so they are free to be fitted there -- but their median "
          f"correlation with dy is only {g['median_abs_corr_with_dy']:.4f}, against "
          f"{c['median_abs_corr_with_dy']:.4f} for the published cond block.")
    print(f"\nThat is within-block variation nearly orthogonal to what the metric "
          f"scores. It is consistent with A1's asymmetry -- overall R2 barely "
          f"moved while the adjacent-pair metric collapsed -- but it does NOT on "
          f"its own establish that the head fits that variation.")
    print(f"\nOnly {g['n_block_constant']} of the {g['n_columns']} geometry columns "
          f"are block-constant, so an ablation restricted to those is not "
          f"available -- there is nothing there to train on. The falsifiable test "
          f"is instead to replace every added column by its per-block MEAN: same "
          f"columns, same between-block content, within-block variation removed by "
          f"construction. If fitting within-block geometry noise is what breaks "
          f"A1, the block-mean variant does not break it.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
