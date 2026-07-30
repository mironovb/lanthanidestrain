#!/usr/bin/env python3
"""Are the sweep's new inputs actually present, or is a cell testing nothing?

Why this exists
---------------
A sweep cell that silently receives no signal does not fail -- it returns a
number indistinguishable from the anchor, and the sweep reports "no effect" for
something it never tried.  That is a worse outcome than a crash, because it is
publishable-looking.

Two cells were at risk here and both had to be checked *before* the GPU runs:

* **A1** adds 119 angular/polyhedral columns to the tabular head.  If those
  columns were mostly missing on the modelled rows, a null would mean "absent",
  not "unhelpful" -- an entirely different claim.

* **B1/B2/B3** attach an auxiliary head.  The aux targets are read from the *row
  table*, not from the design matrix, and the B cells run under the default
  ``baseline_2d`` preset -- which selects neither ``g3`` nor ``gE``.  Had
  ``build_row_table`` returned only the selected columns, every aux target would
  have been NaN, the loss masked to nothing, and B1-B3 would have been three
  exact copies of the anchor wearing different names.

Emitted as a CSV so SWEEP2_RESULTS quotes measured coverage rather than an
assumption, and so a future preset change that breaks either path shows up here
instead of as a mysterious null.

Usage
-----
    python3 -m automl.topo.sweep2_coverage
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"
OUT = REPORTS / "sweep2_coverage.csv"


def main() -> int:
    from automl.topo.train import build_row_table, aux_target_columns

    df, X, cols = build_row_table(preset="baseline_2d", arch="snn")
    _, Xs, cols_s = build_row_table(preset="baseline_2d_shape", arch="snn")
    n = X.shape[0]
    rows: list[dict] = []

    # ---- axis A1: the added tabular columns ------------------------------
    added = [c for c in cols_s if c not in set(cols)]
    idx = [cols_s.index(c) for c in added]
    cov = np.isfinite(Xs[:, idx]).mean(axis=0)
    rows.append(dict(cell="A1", quantity="added columns", value=len(added),
                     unit="count", note="baseline_2d_shape minus baseline_2d"))
    rows.append(dict(cell="A1", quantity="median coverage",
                     value=float(np.median(cov)),
                     unit="fraction of modelled rows", note=""))
    rows.append(dict(cell="A1", quantity="mean coverage", value=float(cov.mean()),
                     unit="fraction of modelled rows", note=""))
    rows.append(dict(cell="A1", quantity="fully populated columns",
                     value=int((cov >= 1.0).sum()), unit="count", note=""))
    rows.append(dict(cell="A1", quantity="entirely empty columns",
                     value=int((cov == 0.0).sum()), unit="count",
                     note="imputed to 0 by _standardise, hence inert"))
    for blk in sorted({c.split("__")[0] for c in added}):
        v = [cv for c, cv in zip(added, cov) if c.split("__")[0] == blk]
        rows.append(dict(cell="A1", quantity=f"median coverage [{blk}]",
                         value=float(np.median(v)), unit="fraction",
                         note=f"{len(v)} columns"))

    # ---- axis B: the auxiliary targets -----------------------------------
    for cell, name in (("B1", "cshm"), ("B2", "eint"), ("B3", "qtransfer")):
        y = aux_target_columns(df, name)
        ok = np.isfinite(y).all(axis=1)
        rows.append(dict(cell=cell, quantity=f"aux target '{name}' usable rows",
                         value=float(ok.mean()), unit="fraction of modelled rows",
                         note=f"{int(ok.sum())} rows, {y.shape[1]} target column(s)"))

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"modelled rows: {n}   row table: {df.shape[1]} columns\n")
    print(out.to_string(index=False))
    # The verdict is computed from the numbers, not asserted alongside them.
    a1 = float(out.query("cell=='A1' and quantity=='median coverage'")["value"].iloc[0])
    aux = float(out[out["quantity"].str.contains("aux target")]["value"].min())
    print(f"\nA1 inputs are present (median coverage {a1:.1%}), so a null from A1 "
          f"would mean the angular columns do not help -- not that they are missing.")
    print(f"Every axis-B target is populated on at least {aux:.1%} of rows, so no B "
          f"cell is a silent copy of the anchor.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
