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
# The confirmatory stage adds twelve seeds to the winner and to A0, giving
# sixteen a side.  The four screening seeds are reused rather than re-run:
# selection happened on the 84 tune extractants alone, so scoring an existing
# run on the 78 confirm extractants is still a first look at those rows, and
# the runs are deterministic -- re-running would return identical predictions
# for eight GPU runs of nothing.
CONFIRM_SEEDS = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157]
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

# POST-HOC cells.  Deliberately NOT in CELLS: they were designed after seeing
# the screen, so letting one compete for the pre-registered gate would be
# selection on a look.  Scored only under --posthoc, against A0 and A1.
POSTHOC: dict[str, dict] = {
    "A1BM": {"preset": "baseline_2d_shape", "extra_block_mean": True},
    # C1 changed the radial basis TWICE over: 32->64 bins and 8.0->10.0 A.
    # If C1 confirms, the study's one improvement would rest on a cell whose
    # cause is unidentified, so these separate the two.  Physically the cutoff
    # is the suspect: 24.1% of atoms lie beyond 8.0 A, so the old basis was
    # saturating for a quarter of every ligand, and 10.0 A exposes an 18.4%
    # shell that had been collapsed onto the boundary.
    "C1BINS": {"radial_bins": 64, "radial_max": 8.0},
    "C1MAX": {"radial_bins": 32, "radial_max": 10.0},
}
# Each post-hoc cell is a decomposition of one screen cell, so it is read
# against that cell, not against a common baseline.  Comparing C1BINS to A1
# would be meaningless.
POSTHOC_REF = {"A1BM": "A1", "C1BINS": "C1", "C1MAX": "C1"}

# Fields that must match the anchor unless the cell varies them, so a run can
# never be swept into the wrong cell.
# extra_block_mean MUST be here: without it a cell whose `want` omits the key
# leaves it unchecked, and the A1BM runs -- same preset as A1 -- would be swept
# into cell A1 and averaged into a published contrast.
DEFAULTS = {"preset": "baseline_2d", "extra_block_mean": False,
            "node_angular": False,
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


def load_cells(verbose: bool = True, seeds: list[int] | None = None,
               include_posthoc: bool = False):
    """Seed-ensembled out-of-fold predictions per cell, plus the seed counts.

    ``seeds`` selects which runs are admitted.  The screen uses the four
    screening seeds; the confirmatory stage passes SEEDS + CONFIRM_SEEDS so the
    winner and the anchor are each ensembled over sixteen.  It is an explicit
    argument rather than a module constant so a confirmatory analysis cannot
    silently pick up screening-only runs, or vice versa.
    """
    seeds = list(SEEDS) if seeds is None else list(seeds)
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
    for name, want in {**CELLS, **(POSTHOC if include_posthoc else {})}.items():
        found: dict[int, Path] = {}
        for cfg, p in runs:
            if not _matches(cfg, want):
                continue
            s = int(cfg.get("seed", -1))
            if s not in seeds:
                continue
            if s in found:
                raise RuntimeError(f"cell {name} seed {s} matched twice: "
                                   f"{found[s].name} and {p.name}")
            # The tag names the cell the job intended; the config records the
            # flags it actually ran with.  If they disagree, a run has been
            # assigned to the wrong cell -- which would corrupt a contrast
            # silently, since every number downstream would still look normal.
            # It is the failure mode a recording bug produces, and nothing else
            # in the pipeline would notice it.
            tag = str(cfg.get("tag") or "")
            if tag.startswith("sw2_"):
                intended = tag[4:].rsplit("_s", 1)[0]
                if intended != name:
                    raise RuntimeError(
                        f"run {p.name} is tagged for cell {intended!r} but its "
                        f"config matches cell {name!r}. One of the two is wrong; "
                        f"refusing to score either.")
            found[s] = p
        if verbose:
            missing = sorted(set(seeds) - set(found))
            print(f"  {name:3s} {str(want) or '(anchor)':44s} "
                  f"seeds={len(found)}/{len(seeds)}"
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
    ap.add_argument("--posthoc", action="store_true",
                    help="score the POST-HOC cells (not pre-registered, no "
                         "confirmatory status) against A0 and A1")
    ap.add_argument("--confirm", metavar="CELL", default=None,
                    help="run the CONFIRMATORY stage for CELL at 16 seeds a "
                         "side. Only legitimate for the cell the screen "
                         "selected, and only once.")
    args = ap.parse_args()

    tune, conf = load_split()
    print(f"[sweep2] frozen split: {len(tune)} tune / {len(conf)} confirm")
    print("\n=== cells ===")
    seed_set = (SEEDS + CONFIRM_SEEDS) if args.confirm else SEEDS
    cells, counts = load_cells(seeds=seed_set, include_posthoc=args.posthoc)
    if args.posthoc:
        return posthoc(cells, counts, tune)
    if "A0" not in cells:
        print("\nThe A0 anchor has no runs; nothing can be screened against it.")
        return 1
    if args.confirm:
        # Only the anchor and the named cell need the full sixteen; every other
        # cell is irrelevant to this stage and is simply not consulted.
        need = ("A0", args.confirm)
        short = [c for c in need if counts.get(c, 0) < len(seed_set)]
        if short and not args.allow_partial:
            print(f"\n[sweep2] {short} have fewer than {len(seed_set)} seeds. "
                  f"The confirmatory contrast is pre-registered at 16 a side, "
                  f"BOTH sides replicated -- that rule cost 25 runs to learn "
                  f"(PI_SWEEP_PRECISION.md). Submit:\n"
                  f"  automl/slurm/campaign_driver.sh automl/slurm/sweep2.sh "
                  f"24 8 34 MODE=confirm CELL={args.confirm}")
            return 1
        return confirmatory(args.confirm, cells, conf, args.n_boot,
                            len(seed_set))

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
        axis = "(anchor)" if name == "A0" else AXIS.get(name[0], "")
        print(f"  {name:4s} {axis:22s} {tb:+12.4f} "
              f"{gain:+9.4f} {ts:+12.4f} {r2:+9.4f}{mark}")
        rows.append({"cell": name, "axis": axis,
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
    print(f"""
  The screen selected a winner, so the confirmatory stage is authorised. It is
  NOT run from these predictions: the pre-registration specifies 16 seeds a
  side, both replicated, and the screen holds only {len(SEEDS)}. Confirming at
  {len(SEEDS)} would compare an under-replicated winner against an
  under-replicated anchor on the one look that is supposed to be decisive.

  Submit, then re-run this module with --confirm {name}:
    automl/slurm/campaign_driver.sh automl/slurm/sweep2.sh 24 8 34 \\
        MODE=confirm CELL={name}""")
    print(f"\n[sweep2] wrote {OUT_CELLS}")
    return 0


def posthoc(cells: dict, counts: dict, tune) -> int:
    """Score the post-hoc cells. Explanatory only -- no gate, no correction."""
    if "A0" not in cells:
        print("[sweep2] no A0 anchor")
        return 1
    a0 = _score(restrict(cells["A0"], tune), BINNED)[0]
    print(f"\n=== POST-HOC (not pre-registered) -- tune half, "
          f"anchor A0 = {a0:+.4f} ===")
    print(f"  {'cell':7s} {'ref':4s} {'seeds':>5s} {'tune binned':>12s} "
          f"{'vs A0':>9s} {'vs ref':>9s}")
    val: dict[str, float] = {}
    for name in list(POSTHOC) + sorted(set(POSTHOC_REF.values())):
        if name in val or name not in cells:
            if name not in cells:
                print(f"  {name:7s} {'':4s} -- no runs")
            continue
        val[name] = _score(restrict(cells[name], tune), BINNED)[0]
    for name in list(POSTHOC) + sorted(set(POSTHOC_REF.values())):
        if name not in val:
            continue
        ref = POSTHOC_REF.get(name, "")
        dref = (f"{val[name] - val[ref]:+9.4f}"
                if ref in val and name in POSTHOC else " " * 9)
        print(f"  {name:7s} {ref:4s} {counts.get(name,0):5d} "
              f"{val[name]:+12.4f} {val[name] - a0:+9.4f} {dref}")

    if "A1BM" in val and "A1" in val:
        bm, a1 = val["A1BM"], val["A1"]
        near = abs(bm - a0) < abs(bm - a1)
        print(f"\n  A1BM keeps A1's 119 columns and its between-block content and "
              f"removes only\n  the within-block variation. It lands "
              f"{'NEAR THE ANCHOR' if near else 'NEAR A1'} "
              f"({bm:+.4f}; A0 {a0:+.4f}, A1 {a1:+.4f}).")
        print("  => " + ("consistent with the head fitting within-block geometry "
                         "variation the\n     metric cannot use: remove that "
                         "variation and the damage goes with it."
                         if near else
                         "NOT the within-block mechanism -- the damage survives "
                         "removal of\n     within-block variation, so something "
                         "else about these columns is responsible."))
    if "C1BINS" in val and "C1MAX" in val and "C1" in val:
        gb, gm, gc = (val["C1BINS"] - a0, val["C1MAX"] - a0, val["C1"] - a0)
        driver = ("the CUTOFF (radial_max 8->10 A)" if gm > gb
                  else "the RESOLUTION (radial_bins 32->64)")
        print(f"\n  C1 moved two things at once. Split: bins-only {gb:+.4f}, "
              f"cutoff-only {gm:+.4f},\n  both {gc:+.4f} -- so C1's gain is "
              f"carried by {driver}.")
        print(f"  Physical check already in hand: 24.1% of atoms lie beyond "
              f"8.0 A, so the\n  published basis saturated for a quarter of "
              f"every ligand.")
    print("\n  Explanatory, post-hoc, no pre-registered decision rule.")
    return 0


def confirmatory(name: str, cells: dict, conf, n_boot: int, n_seeds: int) -> int:
    """The single pre-registered confirmatory look, at 16 seeds a side."""
    if name not in cells:
        print(f"[sweep2] cell {name} has no runs")
        return 1
    print(f"\n=== CONFIRMATORY: {name} vs A0, {n_seeds} seeds a side ===")
    print(f"  {len(conf)} confirm extractants, both keys, "
          f"{N_LOOKS}-look Bonferroni")

    test_rows = []
    for key in KEYS:
        tag = "binned" if key == BINNED else "STRICT"
        a = restrict(cells["A0"], conf)
        b = restrict(cells[name], conf)
        r = paired_adjacent_corrected(a, b, n_boot, seed=0, key_col=key)
        if r is None:
            continue
        clo, chi = _corrected(r["delta"], r["lo"], r["hi"], N_LOOKS)
        v, cv = _verdict(r["lo"], r["hi"]), _verdict(clo, chi)
        print(f"  [{tag:6s}] {name} minus A0: delta={r['delta']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {v} | "
              f"{N_LOOKS}-look [{clo:+.4f}, {chi:+.4f}] {cv}")
        test_rows.append({"key": key, "cell": name, "base": "A0",
                          "n_seeds": n_seeds, **r,
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
    print(f"\n[sweep2] wrote {OUT_TEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
