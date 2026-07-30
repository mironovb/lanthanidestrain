#!/usr/bin/env python3
"""Does angular information, or an auxiliary target, extract more from the complexes?

Pre-registered in ``automl/reports/SWEEP2_PREREGISTRATION.md``, committed before
the first run of any cell existed.

The gap, established by inventory: across all 662 topological runs on disk
``preset`` is ``baseline_2d`` in 662/662, node inputs are five scalars and edge
inputs are one distance, so **no angular, directional or three-body quantity has
ever reached a neural encoder in this study** -- while 119 angular/polyhedral
columns sit in the tabular blocks losing to trees.  A coordination polyhedron is
an angular object.

Analysis discipline, fixed in advance
------------------------------------
* **Screening is selection, not inference.**  Every cell is scored on the 84 tune
  extractants only, against the A0 anchor.  No multiplicity penalty is claimed
  for the screen and no confirmatory language is used about it.
* **One confirmatory look**, on the 78 confirm extractants, for the winner only,
  at 16 seeds with A0 also at 16 -- both sides replicated.
* **If no cell beats A0 by more than 0.005 on tune, the confirmatory run is not
  made.**  Looking twice at nothing is how a winner gets manufactured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl.topo.best_stack import _score
from automl.topo.compare_arms import attach_meta
from automl.topo.dualkey_test import (BINNED, STRICT, KEYS, attach_strict,
                                      load_frames, paired_adjacent_corrected,
                                      _verdict)
from automl.topo.objective_test import load_split, restrict
from automl.topo.stack_test import _corrected

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/topo_sweep2"
REPORTS = REPO / "automl/reports"
OUT_CELLS = REPORTS / "sweep2_cells.csv"
OUT_TEST = REPORTS / "sweep2_test.csv"

SEEDS = [7, 11, 23, 37]
MIN_GAIN = 0.005          # pre-registered screening threshold
N_LOOKS = 21              # >= 20 before this, plus one confirmatory look

# cell -> the config fields that identify it.  Membership comes from the
# RECORDED config, never from the tag: a tag is a label someone typed.
CELLS: dict[str, dict] = {
    "A0": {},
    "A1": {"preset": "baseline_2d_shape"},
    "A2": {"node_angular": True},
    "A3": {"angular_readout": True},
    "B1": {"aux_target": "cshm"},
    "B2": {"aux_target": "eint"},
    "B3": {"aux_target": "qtransfer"},
    "C1": {"radial_bins": 64, "radial_max": 10.0},
    "C2": {"attn_pool": True},
    "C3": {"lr": 5e-4},
    "C4": {"weight_decay": 1e-3},
}
AXIS = {"A": "angular information", "B": "auxiliary target",
        "C": "readout / optimisation"}

# Fields that must match the anchor unless the cell varies them, so a run can
# never be swept into the wrong cell.
DEFAULTS = {"preset": "baseline_2d", "node_angular": False,
            "angular_readout": False, "attn_pool": False,
            "aux_target": None, "radial_bins": None, "radial_max": None,
            "lr": 2e-3, "weight_decay": 1e-4}


def _matches(cfg: dict, want: dict) -> bool:
    if cfg.get("arch") != "snn" or not cfg.get("no_triangles"):
        return False
    if float(cfg.get("pair_loss_weight") or 0.0) != 2.0:
        return False
    if (cfg.get("select_on") or "mse") != "adjacent":
        return False
    if cfg.get("level_weight") is not None:
        return False
    for k, default in DEFAULTS.items():
        target = want.get(k, default)
        got = cfg.get(k, default)
        if isinstance(target, float) or isinstance(got, float):
            if abs(float(got or 0) - float(target or 0)) > 1e-12:
                return False
        elif got != target:
            return False
    return True


def load_cells(verbose: bool = True):
    """Seed-ensembled out-of-fold predictions per cell, plus the seed counts."""
    out: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    if not ART.exists():
        return out, counts
    runs = []
    for j in sorted(ART.glob("run_*.json")):
        cfg = json.loads(j.read_text()).get("config", {})
        p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
        if p.exists():
            runs.append((cfg, p))
    for name, want in CELLS.items():
        found: dict[int, Path] = {}
        for cfg, p in runs:
            if not _matches(cfg, want):
                continue
            s = int(cfg.get("seed", -1))
            if s not in SEEDS:
                continue
            if s in found:
                raise RuntimeError(f"cell {name} seed {s} matched twice: "
                                   f"{found[s].name} and {p.name}")
            found[s] = p
        if verbose:
            missing = sorted(set(SEEDS) - set(found))
            print(f"  {name:3s} {str(want) or '(anchor)':44s} "
                  f"seeds={len(found)}/{len(SEEDS)}"
                  + (f"  MISSING {missing}" if missing else ""))
        if not found:
            continue
        frames = {s: pd.read_parquet(p).drop_duplicates("safe_exp_id")
                  .set_index("safe_exp_id") for s, p in sorted(found.items())}
        idx = None
        for f in frames.values():
            idx = f.index if idx is None else idx.intersection(f.index)
        stack = np.vstack([frames[s].loc[idx, "oof"].to_numpy(float)
                           for s in sorted(frames)])
        # A run whose predictions contain NaN must stop the analysis, not be
        # quietly averaged into a cell.  The cell smoke reported A1 as "OK"
        # while it returned R2 = nan, because it checked the exit code and
        # nothing else -- an all-NaN feature column had poisoned every
        # prediction.  Averaging such a run would turn one broken seed into a
        # broken cell and a NaN contrast, which reads as "no effect".
        bad = ~np.isfinite(stack)
        if bad.any():
            rows = bad.any(axis=0).sum()
            seeds = [sorted(frames)[i] for i in np.where(bad.any(axis=1))[0]]
            raise SystemExit(
                f"[sweep2] cell {name}: {int(bad.sum())} non-finite predictions "
                f"over {rows} rows in seed(s) {seeds}. Refusing to ensemble. "
                f"Re-run those seeds; do not analyse a partly-NaN cell.")
        ens = frames[sorted(frames)[0]].loc[idx].copy()
        ens["oof"] = stack.mean(axis=0)      # every seed present, never a subset
        out[name] = attach_strict(attach_meta(ens))
        counts[name] = len(found)
    return out, counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    tune, conf = load_split()
    print(f"[sweep2] frozen split: {len(tune)} tune / {len(conf)} confirm")
    print("\n=== cells ===")
    cells, counts = load_cells()
    if "A0" not in cells:
        print("\nThe A0 anchor has no runs; nothing can be screened against it.")
        return 1
    incomplete = [c for c, n in counts.items() if n < len(SEEDS)]
    if incomplete and not args.allow_partial:
        print(f"\n[sweep2] incomplete cells {incomplete}. The screen needs all "
              f"{len(SEEDS)} seeds; rerun with --allow-partial for an interim "
              f"read that is NOT the pre-registered screen.")
        return 1

    rows = []
    a0_tune = _score(restrict(cells["A0"], tune), BINNED)[0]
    print(f"\n=== screening on the TUNE half only (anchor A0 = {a0_tune:+.4f}) ===")
    print(f"  {'cell':4s} {'axis':22s} {'tune binned':>12s} {'vs A0':>9s} "
          f"{'tune strict':>12s} {'overall':>9s}")
    for name in CELLS:
        if name not in cells:
            continue
        t = restrict(cells[name], tune)
        tb, r2 = _score(t, BINNED)
        ts, _ = _score(t, STRICT)
        gain = tb - a0_tune
        mark = "  <-- beats threshold" if gain > MIN_GAIN else ""
        print(f"  {name:4s} {AXIS.get(name[0], ''):22s} {tb:+12.4f} "
              f"{gain:+9.4f} {ts:+12.4f} {r2:+9.4f}{mark}")
        rows.append({"cell": name, "axis": AXIS.get(name[0], ""),
                     "n_seeds": counts.get(name, 0),
                     "tune_adj_binned": tb, "gain_vs_A0": gain,
                     "tune_adj_strict": ts, "tune_r2_overall": r2})

    cf = pd.DataFrame(rows)
    OUT_CELLS.parent.mkdir(parents=True, exist_ok=True)
    cf.to_csv(OUT_CELLS, index=False)

    print("\n=== main effects per axis (tune half, vs A0) ===")
    for a, label in AXIS.items():
        sub = cf[cf["cell"].str.startswith(a) & (cf["cell"] != "A0")]
        if len(sub) >= 2:
            print(f"  {label:24s} mean gain {sub['gain_vs_A0'].mean():+.4f} "
                  f"over {len(sub)} cells, best {sub['gain_vs_A0'].max():+.4f} "
                  f"({sub.loc[sub['gain_vs_A0'].idxmax(), 'cell']})")

    # ---- the pre-registered gate -----------------------------------------
    cand = cf[cf["cell"] != "A0"]
    best = cand.loc[cand["gain_vs_A0"].idxmax()] if len(cand) else None
    print("\n=== pre-registered decision (SWEEP2_PREREGISTRATION.md sec 6) ===")
    if best is None or float(best["gain_vs_A0"]) <= MIN_GAIN:
        top = f"{best['cell']} at {float(best['gain_vs_A0']):+.4f}" if best is not None else "none"
        print(f"  best cell: {top}, threshold +{MIN_GAIN:.3f}")
        print(f"""
  NULL. No cell clears the screening threshold, so the confirmatory run is NOT
  made -- looking twice at nothing is how a winner gets manufactured, and that
  was fixed in advance.

  This is a substantive statement, not an absence of one: the encoder is not
  limited by its blindness to angles. Every encoder in this study sees distances
  and scalars only, and giving it the coordination polyhedron -- as tabular
  columns, as node features, and as a readout -- does not help. For a quantity
  as angular as a coordination polyhedron that is surprising, and it points away
  from the representation and towards the data.""")
        pd.DataFrame([]).to_csv(OUT_TEST, index=False)
        print(f"\n[sweep2] wrote {OUT_CELLS}")
        return 0

    name = str(best["cell"])
    print(f"  winner on tune: {name} at {float(best['gain_vs_A0']):+.4f} "
          f"(> +{MIN_GAIN:.3f})")
    print(f"  -> confirmatory contrast on the {len(conf)} CONFIRM extractants, "
          f"both keys, {N_LOOKS}-look Bonferroni")

    test_rows = []
    for key in KEYS:
        tag = "binned" if key == BINNED else "STRICT"
        a = restrict(cells["A0"], conf)
        b = restrict(cells[name], conf)
        r = paired_adjacent_corrected(a, b, args.n_boot, seed=0, key_col=key)
        if r is None:
            continue
        clo, chi = _corrected(r["delta"], r["lo"], r["hi"], N_LOOKS)
        v, cv = _verdict(r["lo"], r["hi"]), _verdict(clo, chi)
        print(f"  [{tag:6s}] {name} minus A0: delta={r['delta']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {v} | "
              f"{N_LOOKS}-look [{clo:+.4f}, {chi:+.4f}] {cv}")
        test_rows.append({"key": key, "cell": name, "base": "A0", **r,
                          f"lo_{N_LOOKS}look": clo, f"hi_{N_LOOKS}look": chi,
                          "verdict": v, "verdict_corrected": cv})

    tf = pd.DataFrame(test_rows)
    tf.to_csv(OUT_TEST, index=False)
    adds = bool(len(tf) and (tf[(tf["key"] == BINNED)]["verdict_corrected"]
                             == "adds").any())
    axis_label = AXIS.get(name[0], "")
    if adds:
        print(f"""
  ==> {axis_label.upper()} IS A REAL ADDITION. {name} beats the anchor on the
      held-out half after correction for all {N_LOOKS} looks. This is the study's
      first genuine improvement to the headline metric rather than another
      control, and it says every encoder to date was blind to something it
      needed.""")
    else:
        print(f"""
  ==> SCREENING NOISE. {name} won on the 84 tune extractants and did not
      replicate on the 78 confirm extractants after correction. Report the null.
      This is exactly the failure the two-stage design exists to catch, and it is
      the fourth time in this study that a screen-selected winner has not
      survived its own confirmation.""")
    print(f"\n[sweep2] wrote {OUT_CELLS}\n[sweep2] wrote {OUT_TEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
