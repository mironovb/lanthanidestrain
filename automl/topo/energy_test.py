#!/usr/bin/env python3
"""Do reference xTB energetics carry adjacent-pair selectivity?

Pre-registered in ``automl/reports/ENERGY_PREREGISTRATION.md`` (commit 6abaf35),
committed before any model had seen these features.

The gap: `binding_energy_eV`, `strain_energy_eV` and the frontier-orbital columns
were queued for 957 complexes and never computed, and **there is not one
energetic descriptor in the entire design matrix** -- though a separation factor
*is* a difference of complexation free energies.

The features are now on disk (``automl/qc/reference_xtb.py``,
``automl/qc/energy_features.py``).  This runs the pre-registered contrast:
the same CatBoost, the same folds, the same seed, the same rows -- one block
added.

A guard runs first, and it is not a formality
----------------------------------------------
Adding columns to the matrix cache must not change what ``baseline_2d`` selects,
or the "control" arm would silently be a different model and the whole A/B would
be meaningless.  The block list is compared column-for-column against the cache
as it stood before, and the run aborts on any difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.experiment import ExperimentSpec, apply_row_filter, run_cv
from automl.topo.best_stack import _score
from automl.topo.dualkey_test import (BINNED, STRICT, KEYS, attach_strict,
                                      paired_adjacent_corrected, _verdict)
from automl.topo.stack_test import _corrected

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
ART = REPO / "automl/artifacts/xtb_reference"
OUT = REPORTS / "energy_test.csv"

# Pre-registered: three new contrasts, taking the look count from 13 to 16.
N_LOOKS = 16

# The published CatBoost baseline, exactly as topo_baselines ran it:
# adj +0.1422, overall +0.4987 (control_cells.csv).
BASE_SPEC = dict(model="catboost", weight_scheme="none", row_filter="ok_only",
                 n_splits=5, repeats=3, seed=42)

PRESETS = {
    "baseline": "baseline_2d",
    "energy (all)": "baseline_2d_energy",
    "energy (absolute only)": "baseline_2d_energy_abs",
    "energy (family-relative only)": "baseline_2d_energy_rel",
}


def guard_baseline_unchanged(blocks, cached_blocks_path: Path) -> None:
    """``baseline_2d`` must select exactly the columns it selected before.

    If it does not, the control arm of this A/B is a different model and the
    comparison measures two things at once.
    """
    if not cached_blocks_path.exists():
        print("[energy] no previous blocks.json to compare against -- skipping "
              "the guard and saying so, rather than passing it silently.")
        return
    prev = json.loads(cached_blocks_path.read_text())["blocks"]
    from automl.dataset import BLOCK_PRESETS
    want = BLOCK_PRESETS["baseline_2d"]
    before, after = [], []
    for n in want:
        before.extend(prev.get(n, []))
        after.extend(blocks.mapping.get(n, []))
    if before != after:
        only_b = set(before) - set(after)
        only_a = set(after) - set(before)
        raise SystemExit(
            f"baseline_2d changed when the energy block was added: "
            f"{len(before)} -> {len(after)} columns, "
            f"{len(only_b)} removed {sorted(only_b)[:3]}, "
            f"{len(only_a)} added {sorted(only_a)[:3]}. "
            f"The A/B would not be an A/B; refusing to run.")
    print(f"[energy] guard OK: baseline_2d still selects {len(after)} columns, "
          f"identical list")


def as_frame(sub: pd.DataFrame, oof: np.ndarray) -> pd.DataFrame:
    f = pd.DataFrame({
        "safe_exp_id": sub["safe_exp_id"].to_numpy(),
        "y": sub["log_D"].to_numpy(dtype=float), "oof": oof,
        "extractant_group": sub["extractant_group"].to_numpy(),
        "composition_key": sub["composition_key"].to_numpy(),
        STRICT: sub[STRICT].to_numpy(),
        "metal": sub["metal"].to_numpy(),
        "lanthanide_index": sub["lanthanide_index"].to_numpy(),
    }).drop_duplicates("safe_exp_id").set_index("safe_exp_id")
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    from automl.matrix_cache import BLOCKS_PATH, build_cache
    prev_blocks = BLOCKS_PATH
    prev_copy = None
    if prev_blocks.exists():
        prev_copy = ART / "blocks_before_energy.json"
        prev_copy.parent.mkdir(parents=True, exist_ok=True)
        prev_copy.write_text(prev_blocks.read_text())

    print("[energy] rebuilding the matrix cache with the gE block")
    df, blocks, info = build_cache()
    guard_baseline_unchanged(blocks, prev_copy) if prev_copy else None
    for b in ("gE", "gE_abs", "gE_rel"):
        print(f"  block {b:8s}: {len(blocks.mapping.get(b, []))} columns")
    if not blocks.mapping.get("gE"):
        raise SystemExit("the gE block is empty -- energy_features.parquet is "
                         "missing or did not merge")

    frames, rows = {}, []
    print("\n=== CatBoost, one block added, everything else identical ===")
    print(f"  {'preset':32s} {'adj (binned)':>13s} {'adj (strict)':>13s} "
          f"{'overall R2':>11s} {'n feat':>7s}")
    for label, preset in PRESETS.items():
        spec = ExperimentSpec(preset=preset, **BASE_SPEC)
        res = run_cv(df, blocks, spec, n_jobs=args.n_jobs)
        sub = apply_row_filter(df, spec.row_filter)
        fr = as_frame(sub, res.oof)
        frames[label] = fr
        a_b, r2 = _score(fr, BINNED)
        a_s, _ = _score(fr, STRICT)
        print(f"  {label:32s} {a_b:+13.4f} {a_s:+13.4f} {r2:+11.4f} "
              f"{res.metrics.get('n_features', 0):7.0f}")
        fr.reset_index().to_parquet(
            REPORTS / f"oof_catboost_{preset}.parquet", index=False)
        rows.append({"kind": "arm", "label": label, "preset": preset,
                     "adj_r2_binned": a_b, "adj_r2_strict": a_s,
                     "r2_overall": r2,
                     "n_features": res.metrics.get("n_features")})

    print("\n=== pre-registered contrasts "
          f"(multiplicity-respecting bootstrap, {N_LOOKS}-look Bonferroni) ===")
    for key in KEYS:
        tag = "binned" if key == BINNED else "STRICT"
        for label in list(PRESETS)[1:]:
            r = paired_adjacent_corrected(frames["baseline"], frames[label],
                                          args.n_boot, seed=0, key_col=key)
            if r is None:
                continue
            clo, chi = _corrected(r["delta"], r["lo"], r["hi"], N_LOOKS)
            v, cv = _verdict(r["lo"], r["hi"]), _verdict(clo, chi)
            print(f"  [{tag:6s}] {label:32s} delta={r['delta']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {v:20s} "
                  f"| {N_LOOKS}-look [{clo:+.4f}, {chi:+.4f}] {cv}")
            rows.append({"kind": "contrast", "key": key, "base": "baseline",
                         "arm": label, **r,
                         f"lo_{N_LOOKS}look": clo, f"hi_{N_LOOKS}look": chi,
                         "verdict": v, "verdict_corrected": cv})

    # Overall log D matters independently: the study's standing weakness is that
    # the deployed stack (+0.4369) is worse on overall R2 than plain CatBoost
    # (+0.4987).  A block that fixes only that is still worth having, and must
    # not be inflated into a selectivity claim.
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)

    print("\n=== pre-registered reading (ENERGY_PREREGISTRATION.md sec 5) ===")
    arms = frame[frame["kind"] == "arm"].set_index("label")
    base_adj = float(arms.loc["baseline", "adj_r2_binned"])
    base_r2 = float(arms.loc["baseline", "r2_overall"])
    con = frame[(frame["kind"] == "contrast") & (frame["key"] == BINNED)]
    any_adds = bool(len(con) and (con["verdict_corrected"] == "adds").any())
    best_r2 = float(arms["r2_overall"].max())
    best_r2_label = str(arms["r2_overall"].idxmax())
    print(f"  baseline CatBoost: adj {base_adj:+.4f}, overall {base_r2:+.4f}")
    print(f"  best overall R2:   {best_r2:+.4f} ({best_r2_label})")
    if any_adds:
        best = con.loc[con["delta"].idxmax()]
        print(f"  ==> ENERGETICS CARRY ADJACENT-PAIR INFORMATION. Best variant "
              f"'{best['arm']}' adds {best['delta']:+.4f} "
              f"[{best['lo']:+.4f}, {best['hi']:+.4f}]. This is the first "
              f"non-geometric feature class in the study.")
    else:
        print(f"  ==> GFN2 ENERGETICS DO NOT carry recoverable adjacent-pair "
              f"information, despite resolving adjacent metals at 17x the "
              f"scale that matters (metal_probe.csv). The barrier is the "
              f"ACCURACY of the semi-empirical energies, not their resolution "
              f"-- which points at DFT, not at more features.")
    if best_r2 > base_r2 + 1e-4:
        print(f"  Overall log D improves by {best_r2 - base_r2:+.4f} "
              f"({base_r2:+.4f} -> {best_r2:+.4f}); reported separately from "
              f"the selectivity claim.")
    print(f"\n[energy] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
