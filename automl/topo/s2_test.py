#!/usr/bin/env python3
"""The S2 confirmatory test.

Pre-registered in ``automl/reports/S2_PREREGISTRATION.md``, committed as
``33324ea`` before the first S2 job was submitted.  This module only executes
what was written there; it was itself written before the runs finished, so the
analysis could not be shaped by the numbers it would produce.

**Primary endpoint.** S2 (32-seed ensemble) minus the repaired FCNN baseline --
the published pipeline with ``QuantileTransformer`` swapped for
``StandardScaler``, 16 seeds, +0.2206.  Paired cluster bootstrap over
extractants, 400 draws, seed 0, via ``control_factorial.paired_adjacent_fast``,
which is verified to reproduce the published headline exactly.

**Multiplicity is reported, not buried.**  S0 was already tested against this
same baseline (+0.0261, spans zero).  S2 is therefore the *second* confirmatory
test of "topology beats the repaired baseline", and the two-test corrected
interval is printed beside the headline every time -- a reader should not have
to reconstruct it.

Secondary contrasts (S2 - T0w, S2 - S0) are descriptive.  They say whether the
levers did anything; they are not the claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.adjacent_test import adj_r2
from automl.topo.compare_arms import attach_meta, collect
from automl.topo.control_factorial import (SEEDS as CONTROL_SEEDS, ensemble,
                                           load_cells, paired_adjacent_fast)

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
S2_DIR = REPO / "automl/artifacts/topo_s2"
OUT = REPORTS / "s2_test.csv"

# The 32 pre-registered seeds; the first 16 are the control factorial's matched
# set so S2 - S0 stays seed-paired.
S2_SEEDS = tuple(CONTROL_SEEDS) + (307, 311, 313, 317, 331, 337, 347, 349,
                                   353, 359, 367, 373, 379, 383, 389, 397)


def load_s2(require: int = 32) -> dict[int, pd.DataFrame]:
    """S2 runs by seed, identified from the recorded config rather than the tag."""
    out: dict[int, pd.DataFrame] = {}
    if not S2_DIR.exists():
        return out
    for j in sorted(S2_DIR.glob("run_*.json")):
        cfg = json.loads(j.read_text()).get("config", {})
        if cfg.get("arch") != "snn":
            continue
        if not cfg.get("block_centre") or int(cfg.get("conformers", 1)) < 2:
            continue
        seed = int(cfg.get("seed", -1))
        if seed not in S2_SEEDS:
            continue
        p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
        if p.exists():
            out[seed] = (pd.read_parquet(p).drop_duplicates("safe_exp_id")
                         .set_index("safe_exp_id"))
    return out


def _score(d: pd.DataFrame) -> tuple[float, float]:
    y = d["y"].to_numpy(float)
    p = d["oof"].to_numpy(float)
    return (adj_r2(y, p, d["composition_key"].to_numpy(),
                   d["lanthanide_index"].to_numpy()), ev._r2(y, p))


def _corrected(dd_lo_hi, n_tests: int) -> tuple[float, float]:
    """Bonferroni-style two-sided interval from the stored bootstrap quantiles.

    Recomputing quantiles needs the draws, which paired_adjacent_fast does not
    return, so the corrected bound is derived from the reported interval under a
    normal approximation.  Stated as an approximation because it is one: with
    90 % -> 95 % one-sided the z ratio is 1.960/1.645, and the interval here is
    mildly asymmetric.
    """
    delta, lo, hi = dd_lo_hi
    se = (hi - lo) / (2 * 1.645)
    return delta - 1.960 * se, delta + 1.960 * se


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--allow-partial", action="store_true",
                    help="score fewer than 32 seeds -- progress check only, "
                         "never a result")
    args = ap.parse_args()

    members = load_s2()
    print(f"=== S2 membership (by recorded config) ===")
    missing = sorted(set(S2_SEEDS) - set(members))
    print(f"  seeds {len(members)}/32" + (f"  MISSING {missing}" if missing else ""))
    if len(members) < len(S2_SEEDS) and not args.allow_partial:
        print("\nRefusing to report: the pre-registration fixes 32 seeds, and an "
              "ensemble over a different seed set is not the arm that was "
              "registered. Use --allow-partial for a progress check only.")
        return 1
    if missing:
        print(f"\n*** PARTIAL: {len(members)}/32 seeds -- NOT a result ***")

    s2 = ensemble(members)
    if s2 is None:
        print("no S2 runs yet")
        return 1

    fixed_path = REPORTS / "oof_fcnn_std_scaler_ens16.parquet"
    if not fixed_path.exists():
        print(f"missing {fixed_path.name}")
        return 1
    fixed = attach_meta(pd.read_parquet(fixed_path)
                        .drop_duplicates("safe_exp_id").set_index("safe_exp_id"))

    cells = load_cells(verbose=False)
    ens = {c: ensemble(m) for c, m in cells.items()}
    ens["FCNN"] = attach_meta(collect().get("baseline::mlp::none"))

    print("\n=== arms (adjacent-pair R2 | overall R2) ===")
    singles = [_score(attach_meta(d))[0] for d in members.values()]
    a, r = _score(s2)
    for name, d in (("repaired FCNN baseline", fixed),
                    ("S0  SNN + contrast (16 seeds)", ens.get("S0")),
                    ("T0w tabular control (16 seeds)", ens.get("T0w"))):
        if d is not None:
            x, o = _score(d)
            print(f"  {name:34s} {x:+.4f}   {o:+.4f}")
    print(f"  {'S2  variance-reduced SNN':34s} {a:+.4f}   {r:+.4f}   "
          f"single {np.mean(singles):+.3f}+/-{np.std(singles):.3f} "
          f"(n={len(singles)})")
    if ens.get("S0") is not None:
        s0_sd = float(pd.read_csv(REPORTS / "control_cells.csv")
                      .set_index("cell").loc["S0", "single_sd"])
        print(f"\n  per-seed SD: S2 {np.std(singles):.4f} vs S0 {s0_sd:.4f}  "
              f"({'lower -- the variance lever worked' if np.std(singles) < s0_sd else 'NOT lower -- the variance diagnosis did not hold'})")

    print("\n=== pre-registered PRIMARY endpoint ===")
    rows = []
    pr = paired_adjacent_fast(fixed, s2, args.n_boot, seed=0)
    if pr is None:
        print("  could not pair S2 against the repaired baseline")
        return 1
    clo, chi = _corrected((pr["delta"], pr["lo"], pr["hi"]), 2)
    verdict = ("SIGNIFICANT" if pr["lo"] > 0 else
               "worse" if pr["hi"] < 0 else "not distinguishable")
    cverdict = ("SIGNIFICANT" if clo > 0 else
                "worse" if chi < 0 else "not distinguishable")
    print(f"  S2 - repaired baseline   delta = {pr['delta']:+.4f}")
    print(f"    90% interval (pre-registered headline) [{pr['lo']:+.4f}, "
          f"{pr['hi']:+.4f}]  P = {pr['p_better']:.2f}   {verdict}")
    print(f"    corrected for 2 tests (~95% one-sided) [{clo:+.4f}, "
          f"{chi:+.4f}]   {cverdict}")
    print(f"    (S0 was the first test of this claim: +0.0261 "
          f"[-0.0049, +0.0762], not distinguishable)")
    rows.append({"kind": "primary", "arm": "S2", "base": "repaired_fcnn",
                 **pr, "lo_2test": clo, "hi_2test": chi,
                 "verdict": verdict, "verdict_2test": cverdict})

    print("\n=== secondary contrasts (descriptive, not confirmatory) ===")
    for base_name, base in (("T0w", ens.get("T0w")), ("S0", ens.get("S0")),
                            ("FCNN", ens.get("FCNN"))):
        if base is None:
            continue
        r2 = paired_adjacent_fast(base, s2, args.n_boot, seed=0)
        if r2 is None:
            continue
        v = ("arm better" if r2["lo"] > 0 else
             "arm worse" if r2["hi"] < 0 else "not distinguishable")
        print(f"  S2 - {base_name:5s} delta = {r2['delta']:+.4f} "
              f"[{r2['lo']:+.4f}, {r2['hi']:+.4f}]  P = {r2['p_better']:.2f}  {v}")
        rows.append({"kind": "secondary", "arm": "S2", "base": base_name,
                     **r2, "verdict": v})

    # A partial run writes to a different path.  s2_test.csv is read by
    # figures_topo to draw the S2 bar and its headline, so a progress check
    # landing there would render a 4-seed ensemble as though it were the
    # 32-seed result -- which nearly happened.
    dest = OUT if not missing else OUT.with_name("s2_test_PARTIAL.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\n[s2] wrote {dest.name}"
          + ("  (PARTIAL -- not the registered arm, not read by the figures)"
             if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
