#!/usr/bin/env python3
"""Campaign 3: does predicting the difference, or conditioning on the medium, help?

Pre-registered in ``CAMPAIGN3_PREREGISTRATION.md``, committed before the code.

Structure follows ``sweep2_test``: screen on the 84 tune extractants against the
D0 anchor, and only if a cell clears the +0.005 gate spend ONE confirmatory look
at 16 seeds a side on the 78 confirm extractants, both block keys, cluster
bootstrap, Bonferroni.

The look count is **26**, not 21.  The confirm extractants were already spent
once on sweep2's C1 contrast, so every claim here is corrected more strictly
than sweep2's was.  Carrying that forward is the whole point of counting looks.

Usage
-----
    python3 -m automl.topo.c3_test --n-boot 400
    python3 -m automl.topo.c3_test --confirm T2X --n-boot 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl.topo.dualkey_test import (BINNED, STRICT, KEYS, attach_strict,
                                      paired_adjacent_corrected)
from automl.topo.objective_test import load_split, restrict
from automl.topo.stack_test import _corrected
from automl.topo.best_stack import _score
from automl.topo.compare_arms import attach_meta

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/topo_c3"
REPORTS = REPO / "automl/reports"
OUT_CELLS = REPORTS / "c3_cells.csv"
OUT_TEST = REPORTS / "c3_test.csv"

SEEDS = [7, 11, 23, 37]
CONFIRM_SEEDS = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157]
MIN_GAIN = 0.005
N_LOOKS = 26           # 21 from sweep2 + one per cell here

CELLS: dict[str, dict] = {
    "D0":  {"pair_loss_weight": 2.0},
    "T2":  {"pair_loss_weight": 2.0, "pair_head": True, "pair_head_weight": 1.0},
    "T2W": {"pair_loss_weight": 2.0, "pair_head": True, "pair_head_weight": 3.0},
    "T2X": {"pair_loss_weight": 0.0, "pair_head": True, "pair_head_weight": 2.0},
    "T3":  {"pair_loss_weight": 2.0, "film": True},
    "T23": {"pair_loss_weight": 2.0, "pair_head": True, "pair_head_weight": 1.0,
            "film": True},
}
# POST-HOC.  Deliberately not in CELLS: designed after the screen, so letting
# either compete for the pre-registered gate would be selection on a look.
POSTHOC: dict[str, dict] = {
    "T2REC":  {"pair_loss_weight": 2.0, "pair_head": True,
               "pair_head_weight": 1.0, "pair_reconcile": True},
    "T2XREC": {"pair_loss_weight": 0.0, "pair_head": True,
               "pair_head_weight": 2.0, "pair_reconcile": True},
}
POSTHOC_REF = {"T2REC": "T2", "T2XREC": "T2X"}

AXIS = {"D0": "(anchor)", "T2": "pairwise head", "T2W": "pairwise head",
        "T2X": "pairwise head", "T3": "condition FiLM", "T23": "both"}

DEFAULTS = {"pair_head": False, "pair_head_weight": 1.0, "film": False,
            "pair_reconcile": False,
            "preset": "baseline_2d", "node_angular": False,
            "angular_readout": False, "attn_pool": False, "aux_target": None,
            "extra_block_mean": False, "radial_bins": None, "radial_max": None,
            "lr": 2e-3, "weight_decay": 1e-4, "pair_loss_weight": 2.0}


def _matches(cfg: dict, want: dict) -> bool:
    if cfg.get("arch") != "snn" or not cfg.get("no_triangles"):
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
    # pair_head_weight is irrelevant when the head is off; do not let it split
    # the anchor into phantom cells.
    return True


def load_cells(verbose: bool = True, seeds: list[int] | None = None,
               include_posthoc: bool = False):
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
            tag = str(cfg.get("tag") or "")
            if tag.startswith("c3_"):
                intended = tag[3:].rsplit("_s", 1)[0]
                if intended != name:
                    continue          # the tag is authoritative for cell identity
            if s in found:
                raise RuntimeError(f"cell {name} seed {s} matched twice: "
                                   f"{found[s].name} and {p.name}")
            found[s] = p
        if verbose:
            missing = sorted(set(seeds) - set(found))
            print(f"  {name:4s} {str(want):62s} seeds={len(found)}/{len(seeds)}"
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
        if not np.isfinite(stack).all():
            raise SystemExit(
                f"[c3] cell {name}: {int((~np.isfinite(stack)).sum())} "
                f"non-finite predictions. Refusing to ensemble.")
        ens = frames[sorted(frames)[0]].loc[idx].copy()
        ens["oof"] = stack.mean(axis=0)
        out[name] = attach_strict(attach_meta(ens))
        counts[name] = len(found)
    return out, counts


def _verdict(lo, hi):
    return "adds" if lo > 0 else ("hurts" if hi < 0 else "not distinguishable")


def confirmatory(name, cells, conf, n_boot, n_seeds) -> int:
    print(f"\n=== CONFIRMATORY: {name} vs D0, {n_seeds} seeds a side ===")
    print(f"  {len(conf)} confirm extractants, both keys, "
          f"{N_LOOKS}-look Bonferroni")
    rows = []
    for key in KEYS:
        tag = "binned" if key == BINNED else "STRICT"
        r = paired_adjacent_corrected(restrict(cells["D0"], conf),
                                      restrict(cells[name], conf),
                                      n_boot, seed=0, key_col=key)
        if r is None:
            continue
        clo, chi = _corrected(r["delta"], r["lo"], r["hi"], N_LOOKS)
        v, cv = _verdict(r["lo"], r["hi"]), _verdict(clo, chi)
        print(f"  [{tag:6s}] {name} minus D0: delta={r['delta']:+.4f} "
              f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {v} | "
              f"{N_LOOKS}-look [{clo:+.4f}, {chi:+.4f}] {cv}")
        rows.append({"key": key, "cell": name, "base": "D0",
                     "n_seeds": n_seeds, **r,
                     f"lo_{N_LOOKS}look": clo, f"hi_{N_LOOKS}look": chi,
                     "verdict": v, "verdict_corrected": cv})
    tf = pd.DataFrame(rows)
    tf.to_csv(OUT_TEST, index=False)
    adds = bool(len(tf) and (tf[tf["key"] == BINNED]["verdict_corrected"]
                             == "adds").any())
    print("\n  ==> " + (
        f"{AXIS.get(name,'')} IS A REAL ADDITION. {name} beats the anchor on "
        f"the held-out half after correction for all {N_LOOKS} looks -- the "
        f"study's first genuine improvement to the headline metric."
        if adds else
        f"SCREENING NOISE. {name} won on the 84 tune extractants and did not "
        f"replicate on the 78 confirm extractants after correction."))
    print(f"\n[c3] wrote {OUT_TEST}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--posthoc", action="store_true")
    ap.add_argument("--confirm", metavar="CELL", default=None)
    args = ap.parse_args()

    tune, conf = load_split()
    print(f"[c3] frozen split: {len(tune)} tune / {len(conf)} confirm")
    print("\n=== cells ===")
    seed_set = (SEEDS + CONFIRM_SEEDS) if args.confirm else SEEDS
    cells, counts = load_cells(seeds=seed_set, include_posthoc=args.posthoc)
    if args.posthoc:
        d0 = _score(restrict(cells["D0"], tune), BINNED)[0]
        print(f"\n=== POST-HOC (not pre-registered) -- tune half, "
              f"D0 = {d0:+.4f} ===")
        print(f"  {'cell':7s} {'ref':5s} {'tune binned':>12s} {'vs D0':>9s} "
              f"{'vs ref':>9s}")
        for nm in list(POSTHOC) + sorted(set(POSTHOC_REF.values())):
            if nm not in cells:
                print(f"  {nm:7s} -- no runs"); continue
            v = _score(restrict(cells[nm], tune), BINNED)[0]
            ref = POSTHOC_REF.get(nm, "")
            dv = ""
            if ref in cells:
                dv = f"{v - _score(restrict(cells[ref], tune), BINNED)[0]:+9.4f}"
            print(f"  {nm:7s} {ref:5s} {v:+12.4f} {v - d0:+9.4f} {dv:>9s}")
        print("\n  Routing the pair head into the metric either recovers what "
              "the\n  separate pathway was discarding, or shows there was "
              "nothing to route.")
        return 0
    if "D0" not in cells:
        print("\nno D0 anchor; nothing can be screened against it.")
        return 1

    if args.confirm:
        short = [c for c in ("D0", args.confirm)
                 if counts.get(c, 0) < len(seed_set)]
        if short and not args.allow_partial:
            print(f"\n[c3] {short} have fewer than {len(seed_set)} seeds. The "
                  f"confirmatory contrast is pre-registered at 16 a side, BOTH "
                  f"sides replicated. Submit:\n  automl/slurm/campaign_driver.sh "
                  f"automl/slurm/campaign3.sh 24 8 30 MODE=confirm "
                  f"CELL={args.confirm}")
            return 1
        return confirmatory(args.confirm, cells, conf, args.n_boot,
                            len(seed_set))

    incomplete = [c for c, n in counts.items() if n < len(SEEDS)]
    if incomplete and not args.allow_partial:
        print(f"\n[c3] incomplete cells {incomplete}; the screen needs all "
              f"{len(SEEDS)} seeds (--allow-partial for an interim read that "
              f"is NOT the pre-registered screen).")
        return 1

    d0 = _score(restrict(cells["D0"], tune), BINNED)[0]
    print(f"\n=== screening on the TUNE half only (anchor D0 = {d0:+.4f}) ===")
    print(f"  {'cell':5s} {'axis':16s} {'tune binned':>12s} {'vs D0':>9s} "
          f"{'tune strict':>12s} {'overall':>9s}")
    rows = []
    for name in CELLS:
        if name not in cells:
            continue
        t = restrict(cells[name], tune)
        tb, r2 = _score(t, BINNED)
        ts, _ = _score(t, STRICT)
        gain = tb - d0
        mark = "  <-- beats threshold" if gain > MIN_GAIN else ""
        print(f"  {name:5s} {AXIS.get(name,''):16s} {tb:+12.4f} {gain:+9.4f} "
              f"{ts:+12.4f} {r2:+9.4f}{mark}")
        rows.append({"cell": name, "axis": AXIS.get(name, ""),
                     "n_seeds": counts.get(name, 0), "tune_adj_binned": tb,
                     "gain_vs_D0": gain, "tune_adj_strict": ts,
                     "tune_r2_overall": r2})
    cf = pd.DataFrame(rows)
    cf.to_csv(OUT_CELLS, index=False)

    cand = cf[cf["cell"] != "D0"]
    best = cand.loc[cand["gain_vs_D0"].idxmax()] if len(cand) else None
    print("\n=== pre-registered decision "
          "(CAMPAIGN3_PREREGISTRATION.md sec 4) ===")
    if best is None or float(best["gain_vs_D0"]) <= MIN_GAIN:
        top = (f"{best['cell']} at {float(best['gain_vs_D0']):+.4f}"
               if best is not None else "none")
        print(f"  best cell: {top}, threshold +{MIN_GAIN:.3f}")
        print("""
  NULL. No cell clears the screening threshold, so the confirmatory run is NOT
  made. Putting the scored quantity directly into the loss -- the best-motivated
  architecture change available -- does not move the metric, and neither does
  making the structural representation depend on the medium.""")
        print(f"\n[c3] wrote {OUT_CELLS}")
        return 0

    name = str(best["cell"])
    print(f"  winner on tune: {name} at {float(best['gain_vs_D0']):+.4f} "
          f"(> +{MIN_GAIN:.3f})")
    print(f"""
  Confirmatory stage authorised, NOT run from these predictions: 16 seeds a
  side is pre-registered and the screen holds only {len(SEEDS)}.
    automl/slurm/campaign_driver.sh automl/slurm/campaign3.sh 24 8 30 \\
        MODE=confirm CELL={name}
  then: python3 -m automl.topo.c3_test --confirm {name} --n-boot 400""")
    print(f"\n[c3] wrote {OUT_CELLS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
