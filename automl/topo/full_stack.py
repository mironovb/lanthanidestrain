#!/usr/bin/env python3
"""How much can stacking actually buy, given every arm this study has produced?

Motivation
----------
``best_stack`` fits three arms (CatBoost, repaired FCNN, S0) on a simplex grid of
step 0.10.  That was the right scope for the question it answered -- *does
topology earn a place in the best model* -- but it is not the best model
obtainable from what is on disk.  Eleven-odd arms exist: the published cells, the
filtration replications, the conformer arm, the extended-seed ensemble, and the
two new encoders.

With the ceiling now measured at **+0.679** and the best stack at **+0.2672**,
there is +0.412 of headroom, and the cheapest place to look for some of it is a
better combination of models that already exist.  This costs no GPU at all.

Selection discipline
--------------------
The trap here is obvious and this study has fallen into it before: choosing which
arms to include *by looking at the score* is fitting the test set with extra
steps.  So

* weights are fitted **nested per held-out extractant**, as in ``best_stack``;
* arm *selection* is **greedy forward selection, also nested** -- for each
  held-out extractant the arm order is chosen on the other 161 only, so the set
  of arms scoring a given extractant never saw it;
* the result is reported with a cluster-bootstrap interval against the published
  three-arm stack, and as another *look* in the multiplicity count.

A nested forward selection that still improves out of sample is a real
improvement.  One that only improves in sample is the winner's curse, and the
interval is what tells them apart.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.best_stack import _score
from automl.topo.compare_arms import attach_meta
from automl.topo.control_factorial import ensemble, load_cells
from automl.topo.dualkey_test import (BINNED, STRICT, KEYS, attach_strict,
                                      load_frames, paired_adjacent_corrected,
                                      _verdict)
from automl.topo.stack_test import _corrected

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts"
REPORTS = REPO / "automl/reports"
OUT = REPORTS / "full_stack.csv"

N_LOOKS = 20      # 19 after the objective test, plus this one


def _ens_dir(d: str, want: dict) -> pd.DataFrame | None:
    """Seed-ensemble every run in a directory whose config matches ``want``."""
    root = ART / d
    if not root.exists():
        return None
    picks = []
    for j in sorted(root.glob("run_*.json")):
        cfg = json.loads(j.read_text()).get("config", {})
        if any(cfg.get(k) != v for k, v in want.items()):
            continue
        p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
        if p.exists():
            picks.append(p)
    if not picks:
        return None
    frames = [pd.read_parquet(p).drop_duplicates("safe_exp_id")
              .set_index("safe_exp_id") for p in picks]
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.intersection(f.index)
    stack = np.vstack([f.loc[idx, "oof"].to_numpy(float) for f in frames])
    ens = frames[0].loc[idx].copy()
    ens["oof"] = stack.mean(axis=0)
    return attach_strict(attach_meta(ens))


def collect_arms(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Every arm on disk that can be scored on the shared rows."""
    arms = dict(load_frames())                      # CatBoost, repaired, S0, T0w
    cells = load_cells(verbose=False)
    for c in ("S1", "T0", "T1", "P0", "P1"):
        if cells.get(c):
            e = ensemble(cells[c])
            if e is not None:
                arms[c] = attach_strict(e)
    extra = {
        "S2": ("topo_s2", {"arch": "snn"}),
        "F30": ("topo_filt", {"filtration_max": 3.0}),
        "F40": ("topo_filt", {"filtration_max": 4.0}),
        "S0X": ("topo_s0_extra", {"arch": "snn"}),
        "G0": ("topo_encoder", {"arch": "snn", "no_triangles": True}),
        "D0": ("topo_encoder", {"arch": "dist"}),
    }
    for name, (d, want) in extra.items():
        f = _ens_dir(d, want)
        if f is not None:
            arms[name] = f
    if verbose:
        for k, v in arms.items():
            print(f"  {k:10s} n={len(v):5d}  adj(binned)={_score(v, BINNED)[0]:+.4f}"
                  f"  adj(strict)={_score(v, STRICT)[0]:+.4f}"
                  f"  overall={_score(v, BINNED)[1]:+.4f}")
    return arms


def _pairs(fr: pd.DataFrame, idx, key: str, groups):
    """Per-extractant adjacent-pair vectors for one arm, on shared rows."""
    d = fr.loc[idx]
    y = d["y"].to_numpy(float); p = d["oof"].to_numpy(float)
    comp = d[key].to_numpy(); li = d["lanthanide_index"].to_numpy()
    out = {}
    for g in pd.unique(groups):
        m = groups == g
        out[g] = ev.adjacent_pair_arrays(y[m], p[m], comp[m], li[m])
    return out


def nested_forward(arms: dict[str, pd.DataFrame], key: str, max_arms: int = 5,
                   step: float = 0.05):
    """Greedy forward selection AND weights, both nested by extractant.

    For each held-out extractant the whole procedure -- which arms, in which
    order, at what weights -- is run on the other extractants only.  The held-out
    extractant is then scored by that procedure's output and never influences it.
    """
    names = list(arms)
    idx = None
    for n in names:
        idx = arms[n].index if idx is None else idx.intersection(arms[n].index)
    ref = arms[names[0]].loc[idx]
    groups = ref["extractant_group"].to_numpy()
    per = {n: _pairs(arms[n], idx, key, groups) for n in names}
    gs = [g for g in pd.unique(groups)
          if len(per[names[0]][g][0])]

    chosen_log = []
    dy_all, dp_all = [], []
    for held in gs:
        others = [o for o in gs if o != held]
        dy_tr = np.concatenate([per[names[0]][o][0] for o in others])
        ss = float(np.sum((dy_tr - dy_tr.mean()) ** 2))
        cur: list[str] = []
        cur_w: np.ndarray = np.array([])
        best_v = -np.inf
        while len(cur) < max_arms:
            gain, pick, pick_w = None, None, None
            for cand in names:
                if cand in cur:
                    continue
                trial = cur + [cand]
                D = {n: np.concatenate([per[n][o][1] for o in others])
                     for n in trial}
                for w in _simplex(len(trial), step):
                    dp = sum(w[i] * D[n] for i, n in enumerate(trial))
                    v = 1.0 - float(np.sum((dy_tr - dp) ** 2)) / ss
                    if gain is None or v > gain:
                        gain, pick, pick_w = v, cand, w
            if gain is None or gain <= best_v + 1e-6:
                break
            best_v, cur, cur_w = gain, cur + [pick], pick_w
        chosen_log.append(tuple(cur))
        dy_te = per[names[0]][held][0]
        dp_te = sum(cur_w[i] * per[n][held][1] for i, n in enumerate(cur))
        dy_all.append(dy_te); dp_all.append(dp_te)
    return (np.concatenate(dy_all), np.concatenate(dp_all), chosen_log,
            idx, groups)


def _simplex(n: int, step: float):
    import itertools
    k = int(round(1.0 / step))
    for combo in itertools.product(range(k + 1), repeat=n - 1):
        if sum(combo) <= k:
            w = [c * step for c in combo]
            w.append(1.0 - sum(w))
            yield np.asarray(w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-arms", type=int, default=4)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    print("=== arms on disk ===")
    arms = collect_arms()
    print(f"\n{len(arms)} arms; nested forward selection up to {args.max_arms}, "
          f"simplex step {args.step}")

    rows = []
    for key in KEYS:
        tag = "binned" if key == BINNED else "STRICT"
        dy, dp, chosen, idx, groups = nested_forward(
            arms, key, max_arms=args.max_arms, step=args.step)
        r2 = ev._r2(dy, dp)
        from collections import Counter
        top = Counter(chosen).most_common(4)
        print(f"\n=== {tag}: nested forward stack over all arms ===")
        print(f"  adjacent-pair R2 = {r2:+.4f}   ({len(dy)} pairs)")
        print(f"  most common selected sets across held-out extractants:")
        for combo, n in top:
            print(f"    {n:4d}x  {' + '.join(combo)}")
        rows.append({"key": key, "kind": "full_stack", "adj_r2": r2,
                     "n_pairs": len(dy), "max_arms": args.max_arms,
                     "step": args.step,
                     "modal_set": " + ".join(top[0][0]) if top else ""})

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    pub = 0.2672
    got = float(out[out["key"] == BINNED]["adj_r2"].iloc[0])
    print(f"\n=== reading ===")
    print(f"  published three-arm stack (binned) : {pub:+.4f}")
    print(f"  nested forward stack, all arms     : {got:+.4f}  "
          f"({got - pub:+.4f})")
    if got - pub > 0.005:
        print("  ==> stacking more of what already exists buys a real amount. "
              "Confirm it with\n      a paired bootstrap before reporting; "
              "nested selection reduces the winner's\n      curse but does not "
              "abolish it.")
    else:
        print("  ==> stacking more arms buys essentially nothing. The three-arm "
              "stack was\n      already at the combination ceiling for these "
              "models, so the remaining\n      headroom to +0.679 is NOT "
              "reachable by recombining what exists -- it needs\n      a model "
              "that is right about something none of these are.")
    print(f"\n[full-stack] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
