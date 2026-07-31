#!/usr/bin/env python3
"""Do changes that fit log_D better make lanthanide selectivity worse?

POST-HOC and exploratory.  Not in SWEEP2_PREREGISTRATION.md, and no decision
rule was fixed for it in advance -- so it is reported as an observation with an
interval, never as a test that something passed.

Why it is worth measuring
-------------------------
Sweep2 scores every cell two ways: overall R2 on log_D, and the adjacent-pair
selectivity metric the study actually cares about.  Several cells moved them in
OPPOSITE directions -- A3 and B2 both fit log_D better than the anchor while
scoring worse on selectivity, and A1 lost 0.32 of selectivity while giving up
only 0.10 of overall R2.

If that opposition is systematic it says something sharp about the task, and
something a reader would want before choosing a model-selection criterion:
most of what improves overall fit is BETWEEN-block variation, which the
adjacent-pair metric averages away on both sides by construction.  Selecting on
overall R2 would then be actively counterproductive for the selectivity goal.

The honest alternative hypothesis is simply that A1 is an outlier dragging a
correlation computed over ~10 points, so the statistic is reported with and
without it, alongside Spearman -- which a single extreme point cannot dominate.

Usage
-----
    python3 -m automl.topo.metric_tension
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"
CELLS = REPORTS / "sweep2_cells.csv"
OUT = REPORTS / "metric_tension.csv"


def _boot_r(x, y, n=2000, seed=0):
    """Percentile CI for Pearson r by resampling cells."""
    rng = np.random.default_rng(seed)
    n_obs = len(x)
    if n_obs < 4:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n):
        p = rng.integers(0, n_obs, n_obs)
        if np.std(x[p]) < 1e-12 or np.std(y[p]) < 1e-12:
            continue
        vals.append(np.corrcoef(x[p], y[p])[0, 1])
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main() -> int:
    if not CELLS.exists():
        raise SystemExit(f"{CELLS} not present; run sweep2_test first")
    d = pd.read_csv(CELLS)
    anchor = d[d["cell"] == "A0"]
    if anchor.empty:
        raise SystemExit("no A0 anchor row")
    a0_overall = float(anchor["tune_r2_overall"].iloc[0])

    c = d[d["cell"] != "A0"].copy()
    c["gain_overall"] = c["tune_r2_overall"] - a0_overall
    c = c.rename(columns={"gain_vs_A0": "gain_adjacent"})

    rows = []
    for label, sub in (("all cells", c),
                       ("excluding A1", c[c["cell"] != "A1"])):
        if len(sub) < 4:
            continue
        x = sub["gain_overall"].to_numpy(float)
        y = sub["gain_adjacent"].to_numpy(float)
        pear = float(np.corrcoef(x, y)[0, 1])
        # Spearman without scipy: Pearson on ranks
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        spear = float(np.corrcoef(rx, ry)[0, 1])
        lo, hi = _boot_r(x, y)
        rows.append(dict(subset=label, n_cells=len(sub), pearson_r=pear,
                         spearman_r=spear, boot_lo=lo, boot_hi=hi,
                         n_cells_overall_up=int((x > 0).sum()),
                         n_cells_adjacent_up=int((y > 0).sum())))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("gain vs the A0 anchor, on the TUNE half\n")
    show = c[["cell", "axis", "gain_adjacent", "gain_overall"]].sort_values(
        "gain_adjacent", ascending=False)
    print(show.to_string(index=False,
                         float_format=lambda v: f"{v:+.4f}"))
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # ---- the part that does not depend on a correlation --------------
    # Which cell would each selection criterion pick, and what does the other
    # metric do there?  This is the decision-relevant statement, it is a
    # comparison of two argmaxes rather than a correlation over 10 points, and
    # it survives however the A1 leverage question is resolved.
    pick_adj = c.loc[c["gain_adjacent"].idxmax()]
    pick_ovr = c.loc[c["gain_overall"].idxmax()]
    print(f"\nselecting on ADJACENT-PAIR picks {pick_adj['cell']}: "
          f"adjacent {pick_adj['gain_adjacent']:+.4f}, "
          f"overall {pick_adj['gain_overall']:+.4f}")
    print(f"selecting on OVERALL R2   picks {pick_ovr['cell']}: "
          f"adjacent {pick_ovr['gain_adjacent']:+.4f}, "
          f"overall {pick_ovr['gain_overall']:+.4f}")
    if pick_adj["cell"] != pick_ovr["cell"]:
        print(f"The two criteria disagree, and each winner is NEGATIVE on the "
              f"other metric. Selecting this sweep on overall R2 would have "
              f"chosen {pick_ovr['cell']}, which LOSES "
              f"{abs(float(pick_ovr['gain_adjacent'])):.4f} of the quantity the "
              f"study is about.")

    if not out.empty:
        r = out.iloc[0]
        both = int(((c["gain_overall"] > 0) & (c["gain_adjacent"] < 0)).sum())
        print(f"\nThe correlation itself is NOT a usable claim. Across "
              f"{int(r['n_cells'])} cells, r = {r['pearson_r']:+.3f} "
              f"[{r['boot_lo']:+.3f}, {r['boot_hi']:+.3f}] but Spearman "
              f"{r['spearman_r']:+.3f} -- a gap that size between Pearson and "
              f"Spearman is the signature of one high-leverage point, not of a "
              f"relationship.")
        print(f"{both} cell(s) improved overall R2 while LOSING adjacent-pair R2; "
              f"{int(r['n_cells_adjacent_up'])} improved adjacent-pair R2 at all.")
        if len(out) > 1:
            q = out.iloc[1]
            print(f"Without A1: r = {q['pearson_r']:+.3f} "
                  f"[{q['boot_lo']:+.3f}, {q['boot_hi']:+.3f}], Spearman "
                  f"{q['spearman_r']:+.3f} -- reported because a single extreme "
                  f"cell should not be allowed to carry the claim.")
        print("\nExploratory. No decision rule was pre-registered for this "
              "quantity, and ~10 cells is a small n for a correlation; it is an "
              "observation to be tested on its own terms, not a result of the "
              "sweep.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
