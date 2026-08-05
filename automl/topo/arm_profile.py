#!/usr/bin/env python3
"""Which arms are strong on LEVELS but weak on DIFFERENCES?

The replication showed row-level stack fitting COLLAPSING from +0.2743 to
+0.2060 when a third arm is added, while pair-level fitting holds at ~+0.265.
The proposed mechanism is that level-fitting rewards an arm for predicting log D
well and the metric never scores that -- so an arm strong on levels and weak on
differences gets weight it does not deserve.

That is a testable claim, not a story: it predicts a specific arm profile, and
it predicts which arm the two fits disagree about.  This measures both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/arm_profile.csv"


def main() -> int:
    from automl.topo.full_stack import collect_arms
    from automl.topo.pair_stack_probe import pair_vectors, _nnls_weights
    from automl.evaluation import _r2 as r2_level

    arms = collect_arms(verbose=False)
    names = sorted(arms)
    idx = None
    for n in names:
        idx = arms[n].index if idx is None else idx.intersection(arms[n].index)
    F = {n: arms[n].loc[idx] for n in names}
    y = F[names[0]]["y"].to_numpy(float)

    rows = []
    for n in names:
        p = F[n]["oof"].to_numpy(float)
        dy, dp, _ = pair_vectors(F[n])
        tot = float(((dy - dy.mean()) ** 2).sum())
        adj = 1.0 - float(((dy - dp) ** 2).sum()) / tot
        rows.append(dict(arm=n, level_r2=float(r2_level(y, p)), pair_r2=adj))
    d = pd.DataFrame(rows)
    d["level_minus_pair"] = d["level_r2"] - d["pair_r2"]
    d = d.sort_values("level_minus_pair", ascending=False)
    print("arms ranked by how much better they look on LEVELS than on PAIRS:\n")
    print(d.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # what weight does each fit give? the mechanism predicts they disagree most
    # about the arm with the largest level_minus_pair
    four = [a for a in ("G0", "repaired", "CatBoost", "D0") if a in names]
    R = np.column_stack([F[n]["oof"].to_numpy(float) for n in four])
    w_row = _nnls_weights(R, y)
    dy = pair_vectors(F[four[0]])[0]
    A = np.column_stack([pair_vectors(F[n])[1] for n in four])
    w_pair = _nnls_weights(A, dy)
    print(f"\nin-sample weights over {four}:")
    print(f"  {'arm':10s} {'row-fit':>9s} {'pair-fit':>9s} {'difference':>11s}")
    for n, a, b in zip(four, w_row, w_pair):
        print(f"  {n:10s} {a:9.3f} {b:9.3f} {b-a:+11.3f}")
    # The comparison must be restricted to arms actually IN the stack.  My first
    # version ranked over all 15 and reported a mismatch, because S1 ties
    # CatBoost on level_minus_pair and sorts first -- but S1 is not weighted by
    # this stack at all, so it cannot be what the two fits disagree about.  The
    # test was mis-constructed, not the hypothesis.
    dis = four[int(np.argmax(np.abs(w_pair - w_row)))]
    sub = d[d["arm"].isin(four)].sort_values("level_minus_pair", ascending=False)
    worst = sub.iloc[0]["arm"]
    print(f"\n  among the {len(four)} arms in the stack:")
    print(sub.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\n  the two fits disagree most about : {dis}")
    print(f"  the most level-flattered arm is  : {worst}")
    print(f"  mechanism predicts these match   : {dis == worst}")
    # and the sharper prediction: row-fitting should favour the arm ranked by
    # LEVEL accuracy, pair-fitting the arm ranked by PAIR accuracy
    top_level = sub.sort_values("level_r2", ascending=False).iloc[0]["arm"]
    top_pair = sub.sort_values("pair_r2", ascending=False).iloc[0]["arm"]
    row_pick = four[int(np.argmax(w_row))]
    pair_pick = four[int(np.argmax(w_pair))]
    print(f"\n  heaviest row-fit weight  : {row_pick:10s} "
          f"(best on LEVELS is {top_level})   match: {row_pick == top_level}")
    print(f"  heaviest pair-fit weight : {pair_pick:10s} "
          f"(best on PAIRS  is {top_pair})   match: {pair_pick == top_pair}")
    d.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
