#!/usr/bin/env python3
"""Freeze the fresh confirmation pairs for the post-0.313 campaign.

The campaign (August 2026) deliberately relaxes the ``geometry_ok`` row filter:
the 5,946-row ``has_3d`` population yields ~1,349 adjacent pairs against the
legacy 905.  The ~444 pairs that exist only in the expanded population have
never been part of any model selection, hyper-parameter choice, stack fit or
"look" in the project's history -- every prior decision was made on the
``ok_only`` subset.  They are therefore the one clean confirmation population
left, and this module freezes their identities BEFORE any new model is trained.

A pair identity is ``(composition_key, l_lo, l_hi)`` with ``l_hi == l_lo + 1``,
matching ``automl.evaluation.adjacent_pair_arrays`` exactly (replicates within
a (block, metal) cell average to one point, so identity is at the cell level).

Writes ``automl/artifacts/fresh_eval/fresh_pairs.json`` once and thereafter
refuses to overwrite it (delete the file by hand if you truly mean to re-freeze).

Also provides ``score_on_pairs(oof_df, which=...)`` so any arm's out-of-fold
predictions can be scored on exactly the frozen population.

Usage:  module load anaconda/Python-ML-2025a
        PYTHONPATH=$PWD python3 -m automl.topo.fresh_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MATRIX = REPO / "automl/artifacts/matrix/matrix.parquet"
OUT = REPO / "automl/artifacts/fresh_eval/fresh_pairs.json"

COLS = ["safe_exp_id", "metal_symbol", "lanthanide_index", "has_3d",
        "geometry_ok", "composition_key", "extractant_group", "log_D"]


def _pair_ids(df: pd.DataFrame) -> set[tuple[str, int, int]]:
    """Adjacent-pair identities under the metric's own enumeration."""
    ids: set[tuple[str, int, int]] = set()
    for key, blk in df.groupby("composition_key"):
        idx = np.sort(blk["lanthanide_index"].unique())
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a == 1:
                ids.add((str(key), int(a), int(b)))
    return ids


def freeze() -> dict:
    m = pd.read_parquet(MATRIX, columns=COLS)
    legacy_rows = m[m["geometry_ok"].astype(bool) & m["has_3d"]]
    expanded_rows = m[m["has_3d"].astype(bool)]

    legacy = _pair_ids(legacy_rows)
    expanded = _pair_ids(expanded_rows)
    fresh = sorted(expanded - legacy)

    payload = {
        "frozen_utc": "2026-08-14",
        "definition": ("adjacent pair identity = (composition_key, l_lo, l_hi), "
                       "l_hi = l_lo + 1, enumerated per "
                       "automl.evaluation.adjacent_pair_arrays on the has_3d "
                       "population (geometry_ok relaxed)"),
        "legacy_population": {"rows": int(len(legacy_rows)),
                              "pairs": int(len(legacy))},
        "expanded_population": {"rows": int(len(expanded_rows)),
                                "pairs": int(len(expanded))},
        "n_fresh": len(fresh),
        "fresh_pairs": [[c, a, b] for c, a, b in fresh],
    }
    return payload


def load_fresh() -> set[tuple[str, int, int]]:
    with open(OUT) as fh:
        payload = json.load(fh)
    return {(c, int(a), int(b)) for c, a, b in payload["fresh_pairs"]}


def score_on_pairs(oof: pd.DataFrame, which: str = "fresh") -> dict[str, float]:
    """Score an OOF frame (y, oof, composition_key, lanthanide_index) on the
    frozen population.  ``which``: 'fresh' | 'legacy' | 'all'."""
    fresh = load_fresh() if which != "all" else None
    dy_all, dp_all = [], []
    for key, blk in oof.groupby("composition_key"):
        blk = blk.groupby("lanthanide_index", as_index=False)[["y", "oof"]].mean()
        idx = blk["lanthanide_index"].to_numpy()
        yv, pv = blk["y"].to_numpy(), blk["oof"].to_numpy()
        i, j = np.triu_indices(len(blk), k=1)
        adj = np.abs(idx[i] - idx[j]) == 1
        for a, b in zip(i[adj], j[adj]):
            lo, hi = sorted((int(idx[a]), int(idx[b])))
            pid = (str(key), lo, hi)
            if fresh is not None:
                is_fresh = pid in fresh
                if which == "fresh" and not is_fresh:
                    continue
                if which == "legacy" and is_fresh:
                    continue
            dy_all.append(yv[a] - yv[b])
            dp_all.append(pv[a] - pv[b])
    dy, dp = np.asarray(dy_all), np.asarray(dp_all)
    if not len(dy):
        return {"n_pairs": 0.0}
    ss = float(np.sum((dy - np.mean(dy)) ** 2))
    r2 = 1.0 - float(np.sum((dy - dp) ** 2)) / ss if ss > 0 else float("nan")
    out = {"n_pairs": float(len(dy)), "r2": r2,
           "mae": float(np.mean(np.abs(dy - dp)))}
    if np.std(dy) > 0 and np.std(dp) > 0:
        out["pearson2"] = float(np.corrcoef(dy, dp)[0, 1] ** 2)
    return out


def main() -> int:
    if OUT.exists():
        print(f"{OUT} already exists -- refusing to re-freeze.")
        return 1
    payload = freeze()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"legacy   {payload['legacy_population']['rows']} rows, "
          f"{payload['legacy_population']['pairs']} adjacent pairs")
    print(f"expanded {payload['expanded_population']['rows']} rows, "
          f"{payload['expanded_population']['pairs']} adjacent pairs")
    print(f"FROZEN   {payload['n_fresh']} fresh pairs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
