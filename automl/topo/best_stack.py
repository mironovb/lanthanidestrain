#!/usr/bin/env python3
"""The best deployable model, and what topology contributes to it.

**DESCRIPTIVE, not a new confirmatory test.**  The pre-registered claim lives in
``STACK_PREREGISTRATION.md`` / ``STACK_RESULTS.md`` and is already decided.  This
answers a practical question that follows from it -- *what is the best model we
can actually build, and does topology earn a place in it* -- and every interval
here is reported with the multiplicity correction, because a leave-one-out
ablation of the stack is yet another look at "does topology add".

The three components are strong at different things:

    CatBoost          overall R2 +0.4987, adjacent +0.1422   (accuracy)
    repaired FCNN     overall R2 +0.3218, adjacent +0.2206   (selectivity)
    S0 simplicial     overall R2 +0.3678, adjacent +0.2382   (selectivity, 3D)

Weights are fitted per held-out extractant on the others only (nested), on a
simplex grid, so no row influences the weights it is scored under.  The grid is
coarse deliberately: with 149 extractants a fine grid would fit the selection
set, and the earlier nested-blend work showed the chosen weight is stable to a
zero-width IQR at this resolution.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.adjacent_test import adj_r2
from automl.topo.compare_arms import attach_meta, collect
from automl.topo.control_factorial import (ensemble, load_cells,
                                           paired_adjacent_fast)

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
OUT = REPORTS / "best_stack.csv"
STEP = 0.1


def _simplex(n: int, step: float = STEP):
    """Weight vectors on the n-simplex at the given resolution."""
    k = int(round(1.0 / step))
    for combo in itertools.product(range(k + 1), repeat=n - 1):
        if sum(combo) <= k:
            w = [c * step for c in combo]
            w.append(1.0 - sum(w))
            yield np.asarray(w)


def nested_stack(frames: dict[str, pd.DataFrame], names: list[str],
                 metric: str = "adjacent",
                 key_col: str = "composition_key"):
    """Nested per-extractant weights over ``names``.

    Like ``stack_test.nested_blend`` this exploits linearity: the stacked
    prediction is a convex combination, and ``adjacent_pair_arrays`` is linear,
    so each extractant's pair vectors are computed once per component and every
    candidate weight is then vector arithmetic.

    ``key_col`` selects the blocking column; the weights are *fitted* under the
    same key they are scored under, which is the only self-consistent choice --
    a stack tuned on one definition of "identical conditions" and scored on
    another is testing two things at once.
    """
    idx = None
    for n in names:
        idx = frames[n].index if idx is None else idx.intersection(frames[n].index)
    ref = frames[names[0]].loc[idx]
    y = ref["y"].to_numpy(float)
    comp = ref[key_col].to_numpy()
    li = ref["lanthanide_index"].to_numpy()
    g = ref["extractant_group"].to_numpy()
    P = {n: frames[n].loc[idx, "oof"].to_numpy(float) for n in names}

    groups = pd.unique(g)
    per = {}
    for grp in groups:
        m = g == grp
        dy, _ = ev.adjacent_pair_arrays(y[m], P[names[0]][m], comp[m], li[m])
        dps = {}
        for n in names:
            _, dp = ev.adjacent_pair_arrays(y[m], P[n][m], comp[m], li[m])
            dps[n] = dp
        per[grp] = (dy, dps)

    out = np.empty(len(ref), dtype=float)
    chosen = []
    grid = list(_simplex(len(names)))
    for grp in groups:
        others = [o for o in groups if o != grp and len(per[o][0])]
        if not others:
            w = np.full(len(names), 1.0 / len(names))
        else:
            dy = np.concatenate([per[o][0] for o in others])
            D = {n: np.concatenate([per[o][1][n] for o in others]) for n in names}
            ss = float(np.sum((dy - dy.mean()) ** 2))
            best, best_v = None, -np.inf
            for cand in grid:
                dp = sum(cand[i] * D[n] for i, n in enumerate(names))
                v = 1.0 - float(np.sum((dy - dp) ** 2)) / ss if ss > 0 else np.nan
                if np.isfinite(v) and v > best_v:
                    best_v, best = v, cand
            w = best
        te = g == grp
        out[te] = sum(w[i] * P[n][te] for i, n in enumerate(names))
        chosen.append(w)
    frame = ref.copy()
    frame["oof"] = out
    return frame, np.asarray(chosen)


def _score(d, key_col: str = "composition_key"):
    y = d["y"].to_numpy(float); p = d["oof"].to_numpy(float)
    return (adj_r2(y, p, d[key_col].to_numpy(),
                   d["lanthanide_index"].to_numpy()), ev._r2(y, p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    cells = load_cells(verbose=False)
    ens = {c: ensemble(m) for c, m in cells.items()}
    frames = {
        "CatBoost": attach_meta(collect()["baseline::catboost::none"]),
        "repaired": attach_meta(
            pd.read_parquet(REPORTS / "oof_fcnn_std_scaler_ens16.parquet")
            .drop_duplicates("safe_exp_id").set_index("safe_exp_id")),
        "S0": ens["S0"],
        "T0w": ens["T0w"],
    }

    print("=== components ===")
    for k in ("CatBoost", "repaired", "S0", "T0w"):
        a, r = _score(frames[k])
        print(f"  {k:10s} adjR2={a:+.4f}  R2={r:+.4f}")

    print("\n=== nested stacks (weights fitted per held-out extractant) ===")
    combos = {
        "full (CatBoost+repaired+S0)": ["CatBoost", "repaired", "S0"],
        "no topology (CatBoost+repaired)": ["CatBoost", "repaired"],
        "topology swapped for control": ["CatBoost", "repaired", "T0w"],
    }
    built = {}
    for label, names in combos.items():
        fr, ws = nested_stack(frames, names)
        built[label] = fr
        a, r = _score(fr)
        wtxt = ", ".join(f"{n}={np.median(ws[:, i]):.2f}"
                         for i, n in enumerate(names))
        print(f"  {label:34s} adjR2={a:+.4f}  R2={r:+.4f}   median w: {wtxt}")

    print("\n=== what topology contributes (DESCRIPTIVE; corrected shown) ===")
    from automl.topo.stack_test import _corrected
    rows = []
    pairs = [("no topology (CatBoost+repaired)", "full (CatBoost+repaired+S0)",
              "drop-in: does adding S0 to the best no-topology stack help?"),
             ("topology swapped for control", "full (CatBoost+repaired+S0)",
              "swap: S0 vs the matched tabular control in the same slot")]
    for a, b, q in pairs:
        r = paired_adjacent_fast(built[a], built[b], args.n_boot, seed=0)
        if r is None:
            continue
        clo, chi = _corrected(r["delta"], r["lo"], r["hi"], 5)
        v = ("adds" if r["lo"] > 0 else "worse" if r["hi"] < 0
             else "not distinguishable")
        cv = ("adds" if clo > 0 else "worse" if chi < 0
              else "not distinguishable")
        print(f"  {b}\n    minus {a}")
        print(f"    delta={r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
              f"P={r['p_better']:.2f}  {v}")
        print(f"    5-test corrected [{clo:+.4f}, {chi:+.4f}]  {cv}")
        print(f"    | {q}")
        rows.append({"base": a, "arm": b, "question": q, **r,
                     "lo_5test": clo, "hi_5test": chi,
                     "verdict": v, "verdict_5test": cv})
    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"\n[best-stack] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
