#!/usr/bin/env python3
"""Does stacking at the PAIR level beat stacking at the row level?

EXPLORATORY.  Tune half only, no decision rule, no confirmatory claim.  Its job
is to say whether a campaign is worth pre-registering.

The idea
--------
Every stack in this study blends per-ROW predictions and lets the metric
difference the block averages afterwards.  But the metric scores a DIFFERENCE,
and the weights that best predict levels are not the weights that best predict
differences: an arm can be mediocre at levels while tracking the slope across
the series well, or the reverse.  Row-level blending optimises the wrong
objective and has no way to notice.

Pair-level stacking forms each arm's predicted separation dp for every adjacent
pair, then fits weights on dp -> dy directly.  That is exactly the quantity
`sel_adj_logSF_r2` scores, so the meta-learner and the metric agree by
construction.

Weights are fitted with leave-extractants-out inner folds, because a pair
belongs to a block which belongs to an extractant, and rows on one ligand are
not independent.

Usage
-----
    python3 -m automl.topo.pair_stack_probe
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
OUT = REPORTS / "pair_stack_probe.csv"


def pair_vectors(fr: pd.DataFrame, key: str = "composition_key"):
    """(dy, dp, extractant per pair) using the metric's own averaging."""
    d = fr[["y", "oof", key, "lanthanide_index", "extractant_group"]].copy()
    d.columns = ["y", "p", "c", "m", "g"]
    dy, dp, gg = [], [], []
    for c, blk in d.groupby("c"):
        g0 = blk["g"].iloc[0]
        b = blk.groupby("m", as_index=False)[["y", "p"]].mean()
        if len(b) < 2:
            continue
        m = b["m"].to_numpy(); yv = b["y"].to_numpy(); pv = b["p"].to_numpy()
        i, j = np.triu_indices(len(b), k=1)
        adj = np.abs(m[i] - m[j]) == 1
        if not adj.any():
            continue
        dy.extend((yv[i][adj] - yv[j][adj]).tolist())
        dp.extend((pv[i][adj] - pv[j][adj]).tolist())
        gg.extend([g0] * int(adj.sum()))
    return np.asarray(dy), np.asarray(dp), np.asarray(gg, dtype=object)


def _r2(dy, dp):
    ss = float(((dy - dp) ** 2).sum())
    tot = float(((dy - dy.mean()) ** 2).sum())
    return 1.0 - ss / tot if tot > 0 else float("nan")


def _nnls_weights(A: np.ndarray, b: np.ndarray, ridge: float = 1e-6):
    """Non-negative least squares by projected gradient. No scipy dependency.

    Non-negative because a stack is a combination of predictors, and a negative
    weight means the meta-learner is exploiting a sign flip it cannot justify
    chemically -- which is exactly how a stack overfits a small pair set.
    """
    n = A.shape[1]
    w = np.full(n, 1.0 / n)
    G = A.T @ A + ridge * np.eye(n)
    c = A.T @ b
    lr = 1.0 / (np.linalg.eigvalsh(G).max() + 1e-12)
    for _ in range(5000):
        w = np.maximum(0.0, w - lr * (G @ w - c))
    s = w.sum()
    return w / s if s > 1e-9 else np.full(n, 1.0 / n)


def main() -> int:
    from automl.topo.full_stack import collect_arms
    from automl.topo.objective_test import load_split, restrict

    tune, conf = load_split()
    arms = collect_arms(verbose=False)
    print(f"[probe] {len(arms)} arms available: {sorted(arms)}\n")

    # shared rows across every arm, restricted to the tune half
    idx = None
    for f in arms.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    names = sorted(arms)
    T = {n: restrict(arms[n].loc[idx], tune) for n in names}

    # per-arm pair vectors, all on the identical pair set
    dy = None; DP = {}
    for n in names:
        a, b, g = pair_vectors(T[n])
        if dy is None:
            dy, groups = a, g
        DP[n] = b
    A = np.column_stack([DP[n] for n in names])
    print(f"[probe] {len(dy)} tune pairs over {len(set(groups))} extractants\n")

    rows = []
    for n in names:
        rows.append(dict(kind="single arm", arm=n, adj_r2=_r2(dy, DP[n])))
    single = pd.DataFrame(rows).sort_values("adj_r2", ascending=False)
    print("single arms, adjacent-pair R2 on the tune half:")
    print(single.head(8).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # ---- pair-level stack, leave-extractants-out ------------------------
    uniq = pd.unique(groups)
    pred = np.zeros_like(dy)
    for gtest in uniq:
        te = groups == gtest
        tr = ~te
        if tr.sum() < 20:
            pred[te] = dy[tr].mean() if tr.sum() else 0.0
            continue
        w = _nnls_weights(A[tr], dy[tr])
        pred[te] = A[te] @ w
    r_pair = _r2(dy, pred)

    # ---- the row-level comparator ---------------------------------------
    # Fitted with the IDENTICAL procedure -- same NNLS, same leave-extractants-
    # out folds -- so the only thing that differs is the level the weights are
    # fitted at.  Comparing against best_stack.nested_stack instead would
    # confound "pair versus row" with "NNLS versus its weight grid", which is a
    # different question and not the one being asked.
    best_single = float(single["adj_r2"].max())
    ref = T[names[0]]
    y_row = ref["y"].to_numpy(float)
    g_row = ref["extractant_group"].to_numpy()
    R = np.column_stack([T[n]["oof"].to_numpy(float) for n in names])
    row_pred = np.zeros_like(y_row)
    for gtest in pd.unique(g_row):
        te = g_row == gtest; tr = ~te
        if tr.sum() < 20:
            row_pred[te] = y_row[tr].mean() if tr.sum() else 0.0
            continue
        wr = _nnls_weights(R[tr], y_row[tr])
        row_pred[te] = R[te] @ wr
    blended = ref.copy(); blended["oof"] = row_pred
    dy_r, dp_r, _ = pair_vectors(blended)
    r_row = _r2(dy_r, dp_r)

    print(f"\n  best single arm            {best_single:+.4f}")
    print(f"  row-level nested stack     {r_row:+.4f}")
    print(f"  PAIR-level nested stack    {r_pair:+.4f}")
    if np.isfinite(r_row):
        print(f"\n  pair-level minus row-level = {r_pair - r_row:+.4f}")
    print(f"  pair-level minus best single = {r_pair - best_single:+.4f}")

    # ---- the FAIR single-arm comparator ---------------------------------
    # "best single arm" above is the max over 15 arms chosen after seeing their
    # tune scores -- an in-sample selection compared against a cross-validated
    # stack.  Selecting the arm INSIDE each fold makes the two honest.
    nested_single = np.zeros_like(dy)
    picked = []
    for gtest in uniq:
        te = groups == gtest; tr = ~te
        if tr.sum() < 20:
            continue
        best_n = max(names, key=lambda n: _r2(dy[tr], DP[n][tr]))
        picked.append(best_n)
        nested_single[te] = DP[best_n][te]
    r_single_nested = _r2(dy, nested_single)
    from collections import Counter
    print(f"\n  best single arm, selected IN-SAMPLE   {best_single:+.4f}"
          f"   <- optimistic, not comparable")
    print(f"  best single arm, selected PER FOLD    {r_single_nested:+.4f}"
          f"   <- the fair comparator")
    print(f"    arms chosen across folds: "
          f"{dict(Counter(picked).most_common(4))}")
    print(f"\n  PAIR-level stack minus fair single arm = "
          f"{r_pair - r_single_nested:+.4f}")
    rows.append(dict(kind="fair single arm (per fold)", arm="nested",
                     adj_r2=r_single_nested))

    # 15 weights fitted over 41 extractant groups is a real overfitting risk,
    # so repeat with the four arms the published stack actually uses.
    small = [n for n in ("G0", "repaired", "CatBoost", "D0") if n in names]
    if len(small) >= 2:
        # own intersection: requiring all 15 arms discards rows these four cover
        si = None
        for n in small:
            si = arms[n].index if si is None else si.intersection(arms[n].index)
        Ts = {n: restrict(arms[n].loc[si], tune) for n in small}
        dy_s = None; DPs = {}
        for n in small:
            a_, b_, g_ = pair_vectors(Ts[n])
            if dy_s is None:
                dy_s, groups_s = a_, g_
            DPs[n] = b_
        y_row_s = Ts[small[0]]["y"].to_numpy(float)
        g_row_s = Ts[small[0]]["extractant_group"].to_numpy()
        print(f"\n  4-arm intersection: {len(dy_s)} pairs over "
              f"{len(set(groups_s))} extractants "
              f"(vs {len(dy)} / {len(set(groups))} for all 15)")
        As = np.column_stack([DPs[n] for n in small])
        Rs = np.column_stack([Ts[n]["oof"].to_numpy(float) for n in small])
        dy, groups, y_row, g_row, ref = dy_s, groups_s, y_row_s, g_row_s, Ts[small[0]]
        uniq = pd.unique(groups)
        pp = np.zeros_like(dy); rp = np.zeros_like(y_row)
        for gtest in uniq:
            te = groups == gtest; tr = ~te
            if tr.sum() >= 20:
                pp[te] = As[te] @ _nnls_weights(As[tr], dy[tr])
        for gtest in pd.unique(g_row):
            te = g_row == gtest; tr = ~te
            if tr.sum() >= 20:
                rp[te] = Rs[te] @ _nnls_weights(Rs[tr], y_row[tr])
        bl = ref.copy(); bl["oof"] = rp
        dyq, dpq, _ = pair_vectors(bl)
        print(f"\n  published 4 arms {small}:")
        print(f"    row-level  {_r2(dyq, dpq):+.4f}")
        print(f"    PAIR-level {_r2(dy, pp):+.4f}")
        rows.append(dict(kind="4-arm row-level", arm="|".join(small),
                         adj_r2=_r2(dyq, dpq)))
        rows.append(dict(kind="4-arm pair-level", arm="|".join(small),
                         adj_r2=_r2(dy, pp)))

    w = _nnls_weights(A, dy)
    print("\n  in-sample pair-level weights (what the metric would prefer):")
    for n, wi in sorted(zip(names, w), key=lambda t: -t[1])[:6]:
        if wi > 0.01:
            print(f"    {n:10s} {wi:.3f}")

    out = pd.DataFrame([dict(quantity="best single arm", value=best_single),
                        dict(quantity="row-level nested stack", value=r_row),
                        dict(quantity="pair-level nested stack", value=r_pair),
                        dict(quantity="n_tune_pairs", value=float(len(dy))),
                        dict(quantity="n_arms", value=float(len(names)))])
    out.to_csv(OUT, index=False)
    print(f"\n  EXPLORATORY -- tune half, no decision rule, no confirmatory claim.")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
