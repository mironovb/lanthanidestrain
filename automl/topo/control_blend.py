#!/usr/bin/env python3
"""Does the blend's interior maximum survive the control?

**Post-hoc, descriptive. Not a pre-registered endpoint.**  No new runs: this
re-blends out-of-fold predictions that already exist.

``PUBLICATION_ASSESSMENT.md`` calls the blend curve "the strongest single piece
of evidence, and it needs no significance threshold": blending the topological
ensemble with CatBoost peaks at +0.2641 at w = 0.7, *above both endpoints*, and
two models carrying the same information can only interpolate monotonically
between them.  The inference is sound.  What it establishes, though, is that the
**contrast-trained topological ensemble** carries adjacent-pair signal CatBoost
lacks -- and "contrast-trained" and "topological" were confounded in every arm
that existed when the curve was drawn.

The control breaks that confound, so the same curve can now be drawn for a
contrast-trained model with **no topology at all**.  If the tabular blend also
peaks in the interior, the complementarity being demonstrated was the objective's
and not topology's, and the sentence in the assessment needs qualifying rather
than retracting -- the curve is still real, it just attributes elsewhere.

Reported as curves, not as a tuned optimum: reading a maximum off the test
metric and quoting it as a result is exactly what the original analysis was
careful not to do, and this inherits that care.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.adjacent_test import adj_r2
from automl.topo.compare_arms import attach_meta, collect
from automl.topo.control_factorial import ensemble, load_cells

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/control_blend.csv"

GRID = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=["S0", "P0", "T0", "T1"])
    args = ap.parse_args()

    members = load_cells(verbose=False)
    ens = {c: ensemble(m) for c, m in members.items()}
    refs = {k: attach_meta(v) for k, v in collect().items()
            if k.startswith("baseline::")}
    cat = refs.get("baseline::catboost::none")
    if cat is None:
        print("CatBoost baseline not found")
        return 1

    rows = []
    for arm in args.arms:
        e = ens.get(arm)
        if e is None or len(members.get(arm, {})) < 16:
            print(f"  {arm}: skipped (needs all 16 seeds, has "
                  f"{len(members.get(arm, {}))})")
            continue
        idx = e.index.intersection(cat.index)
        b = cat.loc[idx]
        y = b["y"].to_numpy(float)
        comp = b["composition_key"].to_numpy()
        li = b["lanthanide_index"].to_numpy()
        pc = b["oof"].to_numpy(float)
        pa = e.loc[idx, "oof"].to_numpy(float)

        curve = []
        for w in GRID:
            p = (1 - w) * pc + w * pa
            curve.append((float(w), adj_r2(y, p, comp, li), ev._r2(y, p)))
            rows.append({"arm": arm, "w": float(w), "adj_r2": curve[-1][1],
                         "r2_overall": curve[-1][2], "n_rows": len(idx)})
        best = max(curve, key=lambda t: t[1])
        ends = max(curve[0][1], curve[-1][1])
        interior = best[0] not in (0.0, 1.0) and best[1] > ends + 1e-9
        print(f"  {arm:4s} endpoints: CatBoost {curve[0][1]:+.4f}, "
              f"arm {curve[-1][1]:+.4f}  |  peak {best[1]:+.4f} at w={best[0]:.1f}  "
              f"-> {'INTERIOR MAXIMUM' if interior else 'no interior maximum'} "
              f"(+{best[1] - ends:.4f} over the better endpoint)")

    if not rows:
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n[control-blend] wrote {OUT}")
    print("An interior maximum for T0 would mean the complementarity the "
          "assessment attributes to topology is the objective's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
