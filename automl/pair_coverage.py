#!/usr/bin/env python3
"""How the row imbalance across the series looks at the level the metric scores.

Written to answer a specific question: Eu is 28 % of the modelled rows, so does
the adjacent-pair metric inherit that imbalance?  It does not inherit it in
full, and the reason is structural rather than lucky:

* ``adjacent_pair_arrays`` averages replicate measurements within a
  (composition block, metal) cell first, so a metal measured ten times in one
  block contributes one point, not ten;
* a pair exists only when *both* neighbours were measured in the same block, so
  a metal's pair count is bounded by its scarcer neighbour, not by its own row
  count.

Writes ``automl/reports/pair_coverage.csv``.  Every number quoted anywhere about
per-metal coverage must come from this file.

Usage:  module load anaconda/Python-ML-2025a
        PYTHONPATH=$PWD python3 -m automl.pair_coverage
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[0].parent
MATRIX = REPO / "automl/artifacts/matrix/matrix.parquet"
OUT = REPO / "automl/reports/pair_coverage.csv"

COLS = ["metal_symbol", "lanthanide_index", "has_3d", "geometry_ok",
        "composition_key", "strict_composition_key", "extractant_group"]


def modelled_rows() -> pd.DataFrame:
    """The published subset: the `ok_only` filter every topo run uses."""
    d = pd.read_parquet(MATRIX, columns=COLS)
    return d[d["geometry_ok"].astype(bool) & d["has_3d"]].reset_index(drop=True)


def coverage(df: pd.DataFrame, key: str) -> pd.Series:
    """Adjacent pairs each metal appears in, under one block key.

    Enumerated exactly as ``automl.evaluation.adjacent_pair_arrays`` does:
    unique metals per block, neighbours only (|delta index| == 1).
    """
    app: dict[float, int] = {}
    for _, blk in df.groupby(key):
        idx = np.sort(blk["lanthanide_index"].unique())
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a == 1:
                app[a] = app.get(a, 0) + 1
                app[b] = app.get(b, 0) + 1
    return pd.Series(app).sort_index()


def main() -> int:
    m = modelled_rows()
    lab = m.drop_duplicates("lanthanide_index").set_index("lanthanide_index")["metal_symbol"]
    rows = m.groupby("lanthanide_index").size()

    out = pd.DataFrame({"metal": lab, "rows": rows}).sort_index()
    for key, tag in [("composition_key", "binned"),
                     ("strict_composition_key", "strict")]:
        out[f"pairs_{tag}"] = coverage(m, key).reindex(out.index).fillna(0).astype(int)
    out["row_share"] = out["rows"] / out["rows"].sum()
    out["pair_share_binned"] = out["pairs_binned"] / out["pairs_binned"].sum()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(OUT, index=False)

    n_pairs = int(out["pairs_binned"].sum() // 2)
    print(f"{len(m)} rows · {m['extractant_group'].nunique()} extractants · "
          f"{m.groupby('composition_key').ngroups} binned blocks · "
          f"{n_pairs} adjacent pairs")
    print(f"row imbalance   {out['rows'].max() / out['rows'].min():.1f}x "
          f"({out.loc[out['rows'].idxmax(), 'metal']} "
          f"{int(out['rows'].max())} vs {out.loc[out['rows'].idxmin(), 'metal']} "
          f"{int(out['rows'].min())})")
    print(f"pair imbalance  "
          f"{out['pairs_binned'].max() / out['pairs_binned'].min():.1f}x "
          f"({out.loc[out['pairs_binned'].idxmax(), 'metal']} "
          f"{int(out['pairs_binned'].max())} vs "
          f"{out.loc[out['pairs_binned'].idxmin(), 'metal']} "
          f"{int(out['pairs_binned'].min())})")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
