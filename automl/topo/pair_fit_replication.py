#!/usr/bin/env python3
"""Is the pair-objective stacking gain robust, or an artefact of one split?

Campaign 5 saw it twice -- tune +0.0171, confirm +0.0559, both keys, both
uncorrected intervals excluding zero -- and reported a null because a 30-look
budget accumulated across four campaigns of unrelated hypotheses swallowed it.

This does not re-litigate that verdict.  It asks a different and more useful
question: **how stable is the effect?**  A finding that survives every way of
cutting the data does not depend on which multiplicity correction one prefers.

Four independent stress tests, none of which is a "confirmatory look":

1. **Full data, nested.**  Both stacks are fitted leave-extractants-out over all
   162 extractants, so every prediction is out-of-fold and no held-out half is
   needed.  Cluster bootstrap over whole extractants.
2. **Split stability.**  Over many random halves of the extractants, in what
   fraction does pair-fitting beat row-fitting?  If it is a split artefact this
   collapses toward 50 %.
3. **Arm-set robustness.**  2, 3, 4 and all available arms.  An effect that only
   appears at one arm count is an overfitting signature.
4. **Both block keys**, throughout.

Reported as an estimate with its stability, not as a hypothesis test.

Usage
-----
    python3 -m automl.topo.pair_fit_replication --n-boot 2000 --n-splits 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
OUT = REPORTS / "pair_fit_replication.csv"

ARM_SETS = {
    "2 arms":  ("G0", "repaired"),
    "3 arms":  ("G0", "repaired", "CatBoost"),
    "4 arms":  ("G0", "repaired", "CatBoost", "D0"),
    "all arms": None,          # everything available
}
KEYS = ("composition_key", "strict_composition_key")


def _r2(dy, dp):
    tot = float(((dy - dy.mean()) ** 2).sum())
    return 1.0 - float(((dy - dp) ** 2).sum()) / tot if tot > 0 else float("nan")


def both_stacks(frames: dict, names: list[str], key: str):
    """Nested row-fitted and pair-fitted predictions over the SAME pairs."""
    from automl.topo.pair_stack_probe import pair_vectors, _nnls_weights
    ref = frames[names[0]]
    y = ref["y"].to_numpy(float)
    g = ref["extractant_group"].to_numpy()
    R = np.column_stack([frames[n]["oof"].to_numpy(float) for n in names])

    row_pred = np.zeros_like(y)
    for gt in pd.unique(g):
        te = g == gt; tr = ~te
        row_pred[te] = (R[te] @ _nnls_weights(R[tr], y[tr])) if tr.sum() >= 20 \
            else (y[tr].mean() if tr.sum() else 0.0)
    bl = ref.copy(); bl["oof"] = row_pred
    dy, dp_row, gp = pair_vectors(bl, key=key)

    DP = {n: pair_vectors(frames[n], key=key)[1] for n in names}
    A = np.column_stack([DP[n] for n in names])
    dp_pair = np.zeros_like(dy)
    for gt in pd.unique(gp):
        te = gp == gt; tr = ~te
        dp_pair[te] = (A[te] @ _nnls_weights(A[tr], dy[tr])) if tr.sum() >= 20 \
            else (dy[tr].mean() if tr.sum() else 0.0)
    return dy, dp_row, dp_pair, gp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-splits", type=int, default=200)
    args = ap.parse_args()

    from automl.topo.full_stack import collect_arms
    arms = collect_arms(verbose=False)
    available = sorted(arms)
    print(f"[repl] {len(available)} arms available\n")

    rows = []
    for label, want in ARM_SETS.items():
        names = list(want) if want else available
        names = [n for n in names if n in arms]
        if len(names) < 2:
            continue
        idx = None
        for n in names:
            idx = arms[n].index if idx is None else idx.intersection(arms[n].index)
        F = {n: arms[n].loc[idx] for n in names}
        for key in KEYS:
            tag = "binned" if key == "composition_key" else "strict"
            dy, dp_r, dp_p, gp = both_stacks(F, names, key)
            r_row, r_pair = _r2(dy, dp_r), _r2(dy, dp_p)
            obs = r_pair - r_row

            # 1. cluster bootstrap over whole extractants, full data
            rng = np.random.default_rng(0)
            uniq = pd.unique(gp)
            by = {g: np.flatnonzero(gp == g) for g in uniq}
            draws = []
            for _ in range(args.n_boot):
                pick = rng.integers(0, len(uniq), len(uniq))
                sel = np.concatenate([by[uniq[i]] for i in pick])
                d = dy[sel]
                if d.std() < 1e-12:
                    continue
                draws.append(_r2(d, dp_p[sel]) - _r2(d, dp_r[sel]))
            draws = np.asarray(draws)
            lo, hi = np.percentile(draws, [2.5, 97.5])
            frac_pos = float((draws > 0).mean())

            # 2. split stability: random halves of the extractants
            rs = np.random.default_rng(1)
            wins = 0; tried = 0
            for _ in range(args.n_splits):
                half = set(rs.choice(uniq, size=max(2, len(uniq)//2),
                                     replace=False).tolist())
                m = np.array([g in half for g in gp])
                if m.sum() < 30 or dy[m].std() < 1e-12:
                    continue
                tried += 1
                wins += int(_r2(dy[m], dp_p[m]) > _r2(dy[m], dp_r[m]))
            share = wins / tried if tried else float("nan")

            print(f"  {label:9s} [{tag:6s}] n={len(dy):4d}  row {r_row:+.4f}  "
                  f"pair {r_pair:+.4f}  delta {obs:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]")
            print(f"                     bootstrap draws positive: {frac_pos:.1%}   "
                  f"random halves won: {share:.1%} of {tried}")
            rows.append(dict(arm_set=label, key=key, n_pairs=len(dy),
                             n_arms=len(names), row_level=r_row,
                             pair_level=r_pair, delta=obs, lo=lo, hi=hi,
                             frac_bootstrap_positive=frac_pos,
                             frac_random_halves_won=share, n_halves=tried))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)

    print(f"\n=== summary over {len(d)} configurations ===")
    print(f"  delta positive in {(d['delta'] > 0).sum()}/{len(d)} configurations")
    print(f"  delta range {d['delta'].min():+.4f} to {d['delta'].max():+.4f}, "
          f"median {d['delta'].median():+.4f}")
    print(f"  bootstrap fraction positive: min {d['frac_bootstrap_positive'].min():.1%}, "
          f"median {d['frac_bootstrap_positive'].median():.1%}")
    print(f"  random halves won:           min {d['frac_random_halves_won'].min():.1%}, "
          f"median {d['frac_random_halves_won'].median():.1%}")
    print("\n  This is an estimate with its stability, not a hypothesis test. It "
          "does not\n  revise campaign 5's pre-registered verdict; it measures "
          "how much that verdict\n  depended on the correction rather than on "
          "the data.")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
