#!/usr/bin/env python3
"""How much of the adjacent-pair metric is even measurable? Split-half reliability.

Why this exists
---------------
Two previous attempts at a ceiling failed, both the same way.  `ceiling_test`
was withdrawn (AUDIT_2026-07-30 E1) for measuring a quantity the model can
predict and the metric averages away.  `ceiling_v2` returned NOT IDENTIFIABLE
because it estimated a noise SD on the 273 cells that happen to carry duplicates
and assumed it transferred to all 905 pairs -- and cells acquire duplicates
preferentially when sources disagree, so it does not transfer.

Split-half reliability avoids the transfer assumption entirely by measuring the
metric on itself:

    1. keep every (block, metal) cell with >= 2 rows
    2. split each cell's rows at random into halves A and B, average each
    3. build adjacent-lanthanide pairs from A alone and from B alone
    4. r = corr(dy_A, dy_B) is the reliability of a HALF-sized measurement
    5. Spearman-Brown to full size:  r_full = 2r / (1 + r)

Measurement error is independent between the halves by construction, so r_full
bounds the R2 any model can reach against this target.  Nothing has to be
assumed representative.

The number is a bound on a *correlation*; R2 against a noisy target is attenuated
by the same factor, which is why r_full is the quantity to compare +0.2382 to.

Usage
-----
    python3 -m automl.topo.reliability --n-splits 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"
OUT = REPORTS / "reliability.csv"


def _pairs_from(vals: dict[int, float]) -> list[tuple[int, float]]:
    """Adjacent-lanthanide differences from one {metal_index: value} map."""
    ks = sorted(vals)
    return [(ks[i], vals[ks[i]] - vals[ks[i + 1]])
            for i in range(len(ks) - 1) if ks[i + 1] - ks[i] == 1]


def split_half(df: pd.DataFrame, key: str, rng: np.random.Generator):
    """One random split -> (dy_A, dy_B) over pairs where both halves exist."""
    a_vals: dict[tuple, dict[int, float]] = {}
    b_vals: dict[tuple, dict[int, float]] = {}
    for (blk, m), sub in df.groupby([key, "lanthanide_index"], sort=False):
        y = sub["log_D"].to_numpy(float)
        if len(y) < 2:
            continue
        idx = rng.permutation(len(y))
        h = len(y) // 2
        a_vals.setdefault(blk, {})[int(m)] = y[idx[:h]].mean()
        b_vals.setdefault(blk, {})[int(m)] = y[idx[h:]].mean()

    dya, dyb = [], []
    for blk, av in a_vals.items():
        bv = b_vals.get(blk, {})
        pa = dict(_pairs_from(av))
        pb = dict(_pairs_from(bv))
        for k in pa.keys() & pb.keys():
            dya.append(pa[k]); dyb.append(pb[k])
    return np.asarray(dya), np.asarray(dyb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-splits", type=int, default=200)
    args = ap.parse_args()

    from automl.topo.train import build_row_table
    df, _, _ = build_row_table(preset="baseline_2d", arch="snn")

    rows = []
    for key, label in (("composition_key", "binned"),
                       ("strict_composition_key", "strict")):
        cells = df.groupby([key, "lanthanide_index"]).size()
        usable = df[df.set_index([key, "lanthanide_index"]).index.isin(
            cells[cells >= 2].index)]
        rs, ns = [], []
        for s in range(args.n_splits):
            a, b = split_half(usable, key, np.random.default_rng(s))
            if len(a) < 20 or a.std() < 1e-9 or b.std() < 1e-9:
                continue
            rs.append(float(np.corrcoef(a, b)[0, 1]))
            ns.append(len(a))
        if not rs:
            print(f"[{label}] too few replicated pairs to estimate reliability")
            continue
        rs = np.asarray(rs)
        # Spearman-Brown: each half carries ~half the measurements, so the
        # full-data reliability is higher than the half-vs-half correlation.
        full = 2 * rs / (1 + rs)
        rows.append(dict(
            key=key, label=label,
            n_cells_with_replicates=int((cells >= 2).sum()),
            median_pairs_per_split=float(np.median(ns)),
            r_halfhalf_median=float(np.median(rs)),
            r_full_median=float(np.median(full)),
            r_full_lo=float(np.percentile(full, 2.5)),
            r_full_hi=float(np.percentile(full, 97.5)),
            n_splits=len(rs)))

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))

    if not out.empty:
        b = out[out["label"] == "binned"]
        if not b.empty:
            r = float(b["r_full_median"].iloc[0])
            lo, hi = float(b["r_full_lo"].iloc[0]), float(b["r_full_hi"].iloc[0])
            npairs = float(b["median_pairs_per_split"].iloc[0])
            print(f"\nBINNED KEY -- the key the headline metric uses")
            print(f"  reliability of the adjacent-pair difference: "
                  f"r_full = {r:.4f} [{lo:.4f}, {hi:.4f}] "
                  f"over {npairs:.0f} replicated pairs per split")
            print(f"  => no model can exceed about R2 = {r:.4f} against this "
                  f"target on the replicated subset.")
            pub = 0.2382
            if r > 0:
                print(f"  => the published arm reaches {pub:+.4f}, which is "
                      f"{pub / r:.0%} of what is measurable.")
            print(f"\n  Stated carefully: this is measured on the {int(b['n_cells_with_replicates'].iloc[0])} "
                  f"cells that carry replicates, and\n  those cells are not a random "
                  f"sample -- a cell acquires a duplicate when someone\n  measured it "
                  f"twice. The bound is reported for that subset. Unlike ceiling_v2\n"
                  f"  it does not assume the value transfers to the full 905 pairs.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
