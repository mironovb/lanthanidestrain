#!/usr/bin/env python3
"""How much does a *single* leave-extractants-out split tell you?  Not much.

Takes the out-of-fold predictions of a fitted configuration and re-scores them
on many random 20 %-of-extractants holdouts.  Because the predictions are held
fixed, the spread that comes out is purely the effect of *which extractants land
in the test set* -- it isolates split luck from model quality.

The answer on this dataset is that the same model reads anywhere from ~0.29 to
~0.70 R^2 across single splits.  Any comparison of two models based on one
holdout is therefore uninterpretable, which is why every claim in the report
uses repeated grouped CV plus a paired bootstrap.
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


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss if ss > 0 else np.nan


def split_spread(oof: pd.DataFrame, test_frac: float = 0.2, n_draws: int = 2000,
                 min_test_rows: int = 50, seed: int = 0) -> np.ndarray:
    groups = oof["extractant_group"].to_numpy()
    y = oof["y"].to_numpy(dtype=float)
    p = oof["oof"].to_numpy(dtype=float)
    uniq = np.unique(groups)
    codes = pd.factorize(groups)[0]
    n_test = max(1, int(round(test_frac * len(uniq))))
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_draws):
        pick = rng.choice(len(uniq), size=n_test, replace=False)
        mask = np.isin(codes, pick)
        if mask.sum() < min_test_rows:
            continue
        scores.append(_r2(y[mask], p[mask]))
    return np.asarray(scores, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", default=str(DEFAULT_OOF))
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--n-draws", type=int, default=2000)
    ap.add_argument("--out", default=str(REPO / "automl/reports/split_variability.json"))
    args = ap.parse_args()

    path = Path(args.oof)
    if not path.exists():
        cands = sorted((REPO / "automl/artifacts/champion").glob("oof_*.parquet"))
        if not cands:
            print("no OOF file available yet")
            return 0
        path = cands[0]
    d = pd.read_parquet(path)
    s = split_spread(d, args.test_frac, args.n_draws)
    pooled = _r2(d["y"].to_numpy(float), d["oof"].to_numpy(float))
    payload = {
        "source_oof": path.name,
        "pooled_repeated_cv_r2": pooled,
        "test_fraction_of_extractants": args.test_frac,
        "n_draws": int(len(s)),
        "percentiles": {str(q): float(np.percentile(s, q))
                        for q in (5, 10, 25, 50, 75, 90, 95)},
        "spread_5_to_95": float(np.percentile(s, 95) - np.percentile(s, 5)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
