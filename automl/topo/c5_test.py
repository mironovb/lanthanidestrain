#!/usr/bin/env python3
"""Campaign 5: does fitting a stack on the pair objective beat fitting it on levels?

Pre-registered in ``CAMPAIGN5_PREREGISTRATION.md``, committed before this ran.

ONE confirmatory look, on the 78 confirm extractants, both block keys, cluster
bootstrap over whole extractants, corrected for 30 looks.

The two sides differ in exactly one respect: the objective the meta-learner is
fitted against.  Same four arms, same non-negative least squares, same
leave-extractants-out folds, same pair set.  Anything else varying would make
the contrast uninterpretable -- the probe compared against a weight-grid stack
once and had to be corrected before any number was believed.

Usage
-----
    python3 -m automl.topo.c5_test --n-boot 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
OUT = REPORTS / "c5_test.csv"

ARMS = ("G0", "repaired", "CatBoost", "D0")     # fixed in advance
N_LOOKS = 30
KEYS = ("composition_key", "strict_composition_key")


def _r2(dy, dp):
    tot = float(((dy - dy.mean()) ** 2).sum())
    return 1.0 - float(((dy - dp) ** 2).sum()) / tot if tot > 0 else float("nan")


def fit_both(frames: dict, names: list[str], key: str):
    """OOF predictions from a row-fitted and a pair-fitted stack.

    Returns (dy, dp_row, dp_pair, group-per-pair).
    """
    from automl.topo.pair_stack_probe import pair_vectors, _nnls_weights
    ref = frames[names[0]]
    y = ref["y"].to_numpy(float)
    g = ref["extractant_group"].to_numpy()
    R = np.column_stack([frames[n]["oof"].to_numpy(float) for n in names])

    # --- row-fitted -------------------------------------------------------
    row_pred = np.zeros_like(y)
    for gt in pd.unique(g):
        te = g == gt; tr = ~te
        row_pred[te] = (R[te] @ _nnls_weights(R[tr], y[tr])) if tr.sum() >= 20 \
            else (y[tr].mean() if tr.sum() else 0.0)
    bl = ref.copy(); bl["oof"] = row_pred
    dy, dp_row, gp = pair_vectors(bl, key=key)

    # --- pair-fitted ------------------------------------------------------
    DP = {}
    for n in names:
        a, b, _ = pair_vectors(frames[n], key=key)
        DP[n] = b
    A = np.column_stack([DP[n] for n in names])
    dp_pair = np.zeros_like(dy)
    for gt in pd.unique(gp):
        te = gp == gt; tr = ~te
        dp_pair[te] = (A[te] @ _nnls_weights(A[tr], dy[tr])) if tr.sum() >= 20 \
            else (dy[tr].mean() if tr.sum() else 0.0)
    return dy, dp_row, dp_pair, gp


def cluster_bootstrap(dy, dp_a, dp_b, groups, n_boot: int, seed: int = 0):
    """Paired cluster bootstrap over whole extractants.

    Resampling WITH replacement and keeping repeats is the correction: an
    extractant drawn twice must count twice, which is what makes the interval
    respect the fact that rows on one ligand are not independent.
    """
    rng = np.random.default_rng(seed)
    uniq = pd.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    obs = _r2(dy, dp_b) - _r2(dy, dp_a)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by[uniq[i]] for i in pick])
        d = dy[sel]
        if d.std() < 1e-12:
            continue
        draws.append(_r2(d, dp_b[sel]) - _r2(d, dp_a[sel]))
    lo, hi = np.percentile(draws, [5.0, 95.0])
    return obs, float(lo), float(hi), len(draws)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    from automl.topo.full_stack import collect_arms
    from automl.topo.objective_test import load_split, restrict
    from automl.topo.stack_test import _corrected

    tune, conf = load_split()
    arms = collect_arms(verbose=False)
    missing = [a for a in ARMS if a not in arms]
    if missing:
        raise SystemExit(f"arms absent: {missing}")
    idx = None
    for a in ARMS:
        idx = arms[a].index if idx is None else idx.intersection(arms[a].index)
    C = {a: restrict(arms[a].loc[idx], conf) for a in ARMS}
    print(f"[c5] CONFIRMATORY, one look, {len(conf)} confirm extractants, "
          f"arms {list(ARMS)}\n")

    rows = []
    for key in KEYS:
        tag = "binned" if key == "composition_key" else "STRICT"
        dy, dp_row, dp_pair, gp = fit_both(C, list(ARMS), key)
        r_row, r_pair = _r2(dy, dp_row), _r2(dy, dp_pair)
        obs, lo, hi, nb = cluster_bootstrap(dy, dp_row, dp_pair, gp, args.n_boot)
        clo, chi = _corrected(obs, lo, hi, N_LOOKS)
        v = "adds" if lo > 0 else ("hurts" if hi < 0 else "not distinguishable")
        cv = "adds" if clo > 0 else ("hurts" if chi < 0 else "not distinguishable")
        print(f"  [{tag:6s}] {len(dy)} pairs   row {r_row:+.4f}   pair {r_pair:+.4f}")
        print(f"           delta={obs:+.4f} [{lo:+.4f}, {hi:+.4f}] {v} | "
              f"{N_LOOKS}-look [{clo:+.4f}, {chi:+.4f}] {cv}")
        rows.append(dict(key=key, n_pairs=len(dy), row_level=r_row,
                         pair_level=r_pair, delta=obs, lo=lo, hi=hi,
                         lo_30look=clo, hi_30look=chi, verdict=v,
                         verdict_corrected=cv, n_boot=nb))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    adds = bool((d[d.key == "composition_key"]["verdict_corrected"] == "adds").any())
    print("\n  ==> " + (
        "FITTING A STACK ON THE PAIR OBJECTIVE IS A REAL IMPROVEMENT. It "
        "replicates on the\n      held-out half after correction for all 30 "
        "looks -- the study's first positive\n      methodological result on the "
        "headline metric."
        if adds else
        "The tune-half gain did NOT replicate on the held-out half after\n"
        "      correction. Report the null."))
    print(f"\n[c5] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
