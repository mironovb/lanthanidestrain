#!/usr/bin/env python3
"""Does splitting the loss into level and contrast terms help?

Pre-registered in ``automl/reports/OBJECTIVE_PREREGISTRATION.md``, committed
before the first run of the decomposed objective existed.

The defect being attacked, measured on this dataset: the composition-block mean
carries Var 1.82 (binned) / 2.41 (strict) and the within-block contrast carries
0.84 / 0.25.  So the published Huber objective spends **68-91% of its gradient**
on a quantity the adjacent-pair metric never reads, and which CatBoost already
predicts better than any network here.  ``--pair-loss-weight`` could only ever
*add* a contrast term on top of the full MSE; no setting of it removes the level
term.  ``--level-weight`` replaces the Huber-on-raw-target with a per-block level
term so the two can be weighted independently.

Analysis discipline, fixed in advance
--------------------------------------
* **Main effects, not cell ranking.**  At 8 seeds the per-cell standard error is
  ~0.017 and cells cannot be ranked; averaging 16 runs per level gives ~0.012.
  That is the lesson ``PI_SWEEP_PRECISION.md`` paid 25 runs to learn, applied
  before the fact this time.
* **Selection on the tune half only.**  Every run trains on all 162 extractants
  and the frozen 84/78 split is applied at *scoring* time -- ``--restrict-groups``
  was tried once, removed 57% of the training rows and collapsed the arm.
* **One confirmatory look** on the 78 confirm extractants, for the winning cell
  only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl.topo.best_stack import nested_stack, _score
from automl.topo.compare_arms import attach_meta
from automl.topo.dualkey_test import (BINNED, STRICT, KEYS, attach_strict,
                                      load_frames, paired_adjacent_corrected,
                                      _verdict)
from automl.topo.stack_test import _corrected

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/topo_objective"
REPORTS = REPO / "automl/reports"
SPLIT = REPO / "automl/artifacts/pi_sweep/split.json"
OUT_CELLS = REPORTS / "objective_cells.csv"
OUT_TEST = REPORTS / "objective_test.csv"

SEEDS = [7, 11, 23, 37, 42, 51, 67, 83]
LEVELS = [0.1, 0.3, 1.0]
BLOCK_KEYS = ["composition_key", "strict_composition_key"]

# Pre-registered: three new contrasts take the look count from 16 to 19.
N_LOOKS = 19


def load_split() -> tuple[set[str], set[str]]:
    """The frozen tune/confirm extractant split.

    Reused rather than re-drawn: one split for the whole study, and its hash is
    checked by ``pi_split --verify``.  Drawing a fresh split per experiment would
    make each one's confirm half a different thing.
    """
    if not SPLIT.exists():
        raise SystemExit(f"{SPLIT} missing; run automl.topo.pi_split --freeze")
    d = json.loads(SPLIT.read_text())
    tune = set(d.get("tune") or d.get("tune_extractants") or [])
    conf = set(d.get("confirm") or d.get("confirm_extractants") or [])
    if not tune or not conf:
        raise SystemExit(f"could not read tune/confirm lists from {SPLIT}: "
                         f"keys present are {sorted(d)}")
    return tune, conf


def load_cells(verbose: bool = True) -> dict[tuple[float, str], pd.DataFrame]:
    """Seed-ensembled out-of-fold predictions, one per (level_weight, block_key).

    Membership comes from the recorded configuration, never from the tag -- the
    rule ``control_factorial`` follows, because a tag is a label someone typed
    and a config is what the run actually did.
    """
    found: dict[tuple[float, str], dict[int, Path]] = {}
    if not ART.exists():
        return {}
    for j in sorted(ART.glob("run_*.json")):
        cfg = json.loads(j.read_text()).get("config", {})
        lw, bk = cfg.get("level_weight"), cfg.get("block_key")
        if lw is None or bk is None:
            continue
        seed = int(cfg.get("seed", -1))
        if seed not in SEEDS:
            continue
        p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
        if not p.exists():
            continue
        cell = (float(lw), str(bk))
        if seed in found.setdefault(cell, {}):
            raise RuntimeError(f"cell {cell} seed {seed} matched twice")
        found[cell][seed] = p

    out: dict[tuple[float, str], pd.DataFrame] = {}
    for cell, per in sorted(found.items()):
        frames = {s: pd.read_parquet(p).drop_duplicates("safe_exp_id")
                  .set_index("safe_exp_id") for s, p in sorted(per.items())}
        idx = None
        for f in frames.values():
            idx = f.index if idx is None else idx.intersection(f.index)
        stack = np.vstack([frames[s].loc[idx, "oof"].to_numpy(float)
                           for s in sorted(frames)])
        ens = frames[sorted(frames)[0]].loc[idx].copy()
        ens["oof"] = stack.mean(axis=0)      # every seed, never a subset
        out[cell] = attach_strict(attach_meta(ens))
        if verbose:
            missing = sorted(set(SEEDS) - set(per))
            print(f"  level_weight={cell[0]:<4} block_key={cell[1]:24s} "
                  f"seeds={len(per)}/8" + (f"  MISSING {missing}" if missing else ""))
    return out


def restrict(d: pd.DataFrame, keep: set[str]) -> pd.DataFrame:
    return d[d["extractant_group"].astype(str).isin(keep)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    tune, conf = load_split()
    print(f"[objective] frozen split: {len(tune)} tune / {len(conf)} confirm "
          f"extractants")

    print("\n=== cells ===")
    cells = load_cells()
    if not cells:
        print("\nNo decomposed-objective runs on disk yet.")
        return 1
    complete = [c for c in cells if len(cells[c]) > 0]
    n_expected = len(LEVELS) * len(BLOCK_KEYS)
    if len(complete) < n_expected and not args.allow_partial:
        print(f"\n[objective] {len(complete)}/{n_expected} cells present. The "
              f"pre-registered analysis needs all of them; rerun with "
              f"--allow-partial for an interim read that is NOT the endpoint.")
        return 1
    interim = len(complete) < n_expected

    frames = load_frames()
    rows = []
    print(f"\n=== cell scores (TUNE half only -- selection sees nothing else) ===")
    print(f"  {'level_w':>8s} {'block_key':>24s} {'tune binned':>12s} "
          f"{'tune strict':>12s} {'full binned':>12s}")
    for cell, fr in sorted(cells.items()):
        lw, bk = cell
        t = restrict(fr, tune)
        tb, _ = _score(t, BINNED)
        ts, _ = _score(t, STRICT)
        fb, fr2 = _score(fr, BINNED)
        print(f"  {lw:8.1f} {bk:>24s} {tb:+12.4f} {ts:+12.4f} {fb:+12.4f}")
        rows.append({"level_weight": lw, "block_key": bk,
                     "n_seeds": len(cells[cell]),
                     "tune_adj_binned": tb, "tune_adj_strict": ts,
                     "full_adj_binned": fb, "full_r2_overall": fr2})

    cf = pd.DataFrame(rows)
    OUT_CELLS.parent.mkdir(parents=True, exist_ok=True)
    cf.to_csv(OUT_CELLS, index=False)

    # ---- main effects, which is what this design can actually resolve -------
    print("\n=== main effects on the tune half "
          "(the design's resolvable unit, not cell ranking) ===")
    for axis, levels in (("level_weight", LEVELS), ("block_key", BLOCK_KEYS)):
        print(f"  {axis}:")
        for lv in levels:
            sub = cf[cf[axis] == lv]
            if sub.empty:
                continue
            print(f"    {str(lv):24s} mean tune adj (binned) = "
                  f"{sub['tune_adj_binned'].mean():+.4f}  "
                  f"(n={int(sub['n_seeds'].sum())} runs)")

    # ---- selection, then ONE confirmatory look -----------------------------
    best = cf.loc[cf["tune_adj_binned"].idxmax()]
    best_cell = (float(best["level_weight"]), str(best["block_key"]))
    print(f"\n=== selected on the tune half: level_weight="
          f"{best_cell[0]}, block_key={best_cell[1]} ===")

    frames["OBJ"] = cells[best_cell]
    combos = {
        "with S0": ["CatBoost", "repaired", "S0"],
        "with OBJ": ["CatBoost", "repaired", "OBJ"],
        "no topology": ["CatBoost", "repaired"],
    }
    test_rows = []
    for key in KEYS:
        tag = "binned" if key == BINNED else "STRICT"
        built = {}
        for name, names in combos.items():
            fr, _ = nested_stack(frames, names, key_col=key)
            built[name] = fr
        # confirm half only -- the endpoint
        for base, arm, q in (("with S0", "with OBJ",
                              "does the decomposed objective beat S0 in the "
                              "same stack slot?"),
                             ("no topology", "with OBJ",
                              "does it add to the no-topology stack at all?")):
            a = restrict(built[base], conf)
            b = restrict(built[arm], conf)
            r = paired_adjacent_corrected(a, b, args.n_boot, seed=0,
                                          key_col=key)
            if r is None:
                continue
            clo, chi = _corrected(r["delta"], r["lo"], r["hi"], N_LOOKS)
            v, cv = _verdict(r["lo"], r["hi"]), _verdict(clo, chi)
            print(f"  [{tag:6s} CONFIRM] {arm} minus {base}: "
                  f"delta={r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
                  f"{v} | {N_LOOKS}-look [{clo:+.4f}, {chi:+.4f}] {cv}")
            test_rows.append({"key": key, "half": "confirm", "base": base,
                              "arm": arm, "question": q,
                              "level_weight": best_cell[0],
                              "block_key": best_cell[1], **r,
                              f"lo_{N_LOOKS}look": clo,
                              f"hi_{N_LOOKS}look": chi,
                              "verdict": v, "verdict_corrected": cv,
                              "interim": interim})

    tf = pd.DataFrame(test_rows)
    tf.to_csv(OUT_TEST, index=False)

    print("\n=== pre-registered reading "
          "(OBJECTIVE_PREREGISTRATION.md sec 6) ===")
    if interim:
        print("  INTERIM -- not all cells present; this is NOT the endpoint.")
    beats_s0 = bool(len(tf) and (
        (tf["arm"] == "with OBJ") & (tf["base"] == "with S0")
        & (tf["key"] == BINNED) & (tf["verdict_corrected"] == "adds")).any())
    adds_at_all = bool(len(tf) and (
        (tf["base"] == "no topology") & (tf["key"] == BINNED)
        & (tf["verdict_corrected"] == "adds")).any())
    print(f"  beats S0 in the same slot (confirm half) : {beats_s0}")
    print(f"  adds to the no-topology stack            : {adds_at_all}")
    if beats_s0:
        print("\n  ==> THE OBJECTIVE WAS THE BINDING CONSTRAINT, not the "
              "representation.\n      The published +0.04 topology effect was "
              "measured through a loss that was\n      mostly looking "
              "elsewhere. This is the study's first genuine improvement\n"
              "      to the headline metric rather than another control.")
    elif adds_at_all:
        print("\n  ==> The decomposed arm earns a stack slot but does not beat "
              "S0 in it.\n      Report the level/contrast split as a better way "
              "to train this model,\n      not as a larger effect.")
    else:
        print("\n  ==> The level term was NOT the binding constraint. An "
              "objective spending\n      ~91% of its gradient on nuisance is "
              "not what limits this problem,\n      which points at the "
              "representation or the data rather than the loss.\n      That is "
              "worth stating plainly -- it closes a live hypothesis.")
    print(f"\n[objective] wrote {OUT_CELLS}\n[objective] wrote {OUT_TEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
