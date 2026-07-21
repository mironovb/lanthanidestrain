#!/usr/bin/env python3
"""Where is the remaining error, and on which chemistries?

Splits the best model's held-out squared error into
  * a per-extractant *level offset* (the whole family placed too high or low), and
  * *within-extractant scatter* (metal and conditions inside a family),
then lists the extractants that contribute most through a wrong level.

The offset part is addressable by collecting data on the mis-levelled donor
classes; the scatter part is the component the 3D descriptors were meant to
supply and did not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OOF = (REPO / "automl/artifacts/champion"
               / "oof_2_baseline_2D,_CatBoost_+_group_wts.parquet")
MATRIX = REPO / "automl/artifacts/matrix/matrix.parquet"


def breakdown(oof: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    err = (oof["oof"] - oof["y"]).to_numpy(dtype=float)
    codes, uniq = pd.factorize(oof["extractant_group"].to_numpy())
    counts = np.bincount(codes)
    bias = np.bincount(codes, weights=err) / counts
    ss_total = float((err ** 2).sum())
    ss_bias = float((counts * bias ** 2).sum())
    summary = {
        "total_squared_error": ss_total,
        "from_per_extractant_offset": ss_bias,
        "offset_share": ss_bias / ss_total if ss_total else np.nan,
        "from_within_extractant_scatter": ss_total - ss_bias,
        "scatter_share": 1 - ss_bias / ss_total if ss_total else np.nan,
        "n_extractants": int(len(uniq)),
    }
    per = pd.DataFrame({
        "extractant_group": uniq,
        "n": counts,
        "bias": bias,
        "error_share_from_offset": counts * bias ** 2 / ss_total if ss_total else np.nan,
    })
    stats = oof.assign(err=err).groupby("extractant_group").agg(
        sd_logD=("y", "std"), mae=("err", lambda s: s.abs().mean()))
    per = per.merge(stats, on="extractant_group", how="left")
    return summary, per.sort_values("error_share_from_offset", ascending=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", default=str(DEFAULT_OOF))
    ap.add_argument("--out", default=str(REPO / "automl/reports/per_extractant_errors.csv"))
    ap.add_argument("--summary-out",
                    default=str(REPO / "automl/reports/error_breakdown.json"))
    args = ap.parse_args()

    path = Path(args.oof)
    if not path.exists():
        cands = sorted((REPO / "automl/artifacts/champion").glob("oof_*.parquet"))
        if not cands:
            print("no OOF file available yet")
            return 0
        path = cands[0]
    oof = pd.read_parquet(path)
    summary, per = breakdown(oof)
    summary["source_oof"] = path.name

    if MATRIX.exists():
        names = pd.read_parquet(
            MATRIX, columns=["extractant_group", "extractant_name"]
        ).drop_duplicates("extractant_group")
        per = per.merge(names, on="extractant_group", how="left")

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    cols = [c for c in ("extractant_name", "n", "bias", "mae", "sd_logD",
                        "error_share_from_offset") if c in per.columns]
    per[cols + ["extractant_group"]].to_csv(args.out, index=False)

    print(f"total squared error              {summary['total_squared_error']:10.1f}")
    print(f"  per-extractant level offset    {summary['from_per_extractant_offset']:10.1f}"
          f"  ({100 * summary['offset_share']:.1f} %)")
    print(f"  within-extractant scatter      "
          f"{summary['from_within_extractant_scatter']:10.1f}"
          f"  ({100 * summary['scatter_share']:.1f} %)")
    print()
    print("extractants contributing most error through a wrong level:")
    print(per.head(8)[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
