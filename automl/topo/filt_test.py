#!/usr/bin/env python3
"""Is the positive result about the complex, or about one filtration radius?

Pre-registered in ``automl/reports/FILT_PREREGISTRATION.md``; this module was
written before any filtration run finished and executes only what was fixed
there.

Same architecture, objective, folds and seeds as S0 -- only ``--filtration-max``
differs (3.0 A and 4.0 A against S0's 3.5 A), so the comparison isolates the
complex rather than the model.

For each radius the primary endpoint is whether it adds to the best no-topology
stack, exactly as S0 was tested.  The secondary reports the two quantities the
mechanism says must *both* be favourable -- the arm's own adjacent-pair R2 and
its error correlation with the repaired baseline -- so a failure can be
attributed rather than left unexplained.  That is how the PI-CNN failure was
diagnosed, and it is the check that makes the mechanism predictive rather than a
story fitted after the fact.
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
from automl.topo.best_stack import nested_stack, _score
from automl.topo.control_factorial import (ensemble, load_cells,
                                           paired_adjacent_fast)
from automl.topo.stack_test import _corrected

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
FILT = REPO / "automl/artifacts/topo_filt"
OUT = REPORTS / "filt_test.csv"
N_LOOKS = 7          # S0, S2, stack primary, stack decisive, S0X, F30, F40


def load_filt(radius: float) -> dict[int, pd.DataFrame]:
    """Runs at one radius, identified by recorded config rather than tag."""
    out: dict[int, pd.DataFrame] = {}
    if not FILT.exists():
        return out
    for j in sorted(FILT.glob("run_*.json")):
        cfg = json.loads(j.read_text()).get("config", {})
        if cfg.get("arch") != "snn":
            continue
        if abs(float(cfg.get("filtration_max", -1)) - radius) > 1e-6:
            continue
        if (float(cfg.get("pair_loss_weight") or 0) != 2.0
                or cfg.get("select_on") != "adjacent"):
            continue
        p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
        if p.exists():
            out[int(cfg["seed"])] = (pd.read_parquet(p)
                                     .drop_duplicates("safe_exp_id")
                                     .set_index("safe_exp_id"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--min-seeds", type=int, default=8)
    args = ap.parse_args()

    cells = load_cells(verbose=False)
    s0 = ensemble(cells["S0"])
    a0, _ = _score(s0)
    print(f"harness check: published S0 re-ensembles to {a0:+.4f} "
          f"(must be +0.2382)")
    if abs(a0 - 0.2382) > 5e-4:
        raise SystemExit("published S0 drifted; refusing to report")

    base = {"CatBoost": attach_meta(collect()["baseline::catboost::none"]),
            "repaired": attach_meta(
                pd.read_parquet(REPORTS / "oof_fcnn_std_scaler_ens16.parquet")
                .drop_duplicates("safe_exp_id").set_index("safe_exp_id"))}
    noto, _ = nested_stack(base, ["CatBoost", "repaired"])
    an, _ = _score(noto)
    print(f"no-topology stack: adjR2 = {an:+.4f}\n")

    arms = {"S0 (3.5 A, published)": s0}
    for r in (3.0, 4.0):
        m = load_filt(r)
        print(f"filtration {r} A: {len(m)} seeds"
              + ("" if len(m) >= args.min_seeds else "  -- INCOMPLETE"))
        if len(m) >= args.min_seeds:
            arms[f"F{int(r*10)} ({r} A)"] = ensemble(m)

    # Mechanism numbers: strong on the metric AND decorrelated?
    print("\n=== mechanism (both must be favourable) ===")
    ref = base["repaired"]
    rows = []
    for label, arm in arms.items():
        idx = arm.index.intersection(ref.index)
        A, B = arm.loc[idx], ref.loc[idx]
        y = B["y"].to_numpy(float)
        comp = B["composition_key"].to_numpy(); li = B["lanthanide_index"].to_numpy()
        dy, dpa = ev.adjacent_pair_arrays(y, A["oof"].to_numpy(float), comp, li)
        _, dpb = ev.adjacent_pair_arrays(y, B["oof"].to_numpy(float), comp, li)
        corr = float(np.corrcoef(dy - dpa, dy - dpb)[0, 1])
        a, _r = _score(arm)
        print(f"  {label:24s} adjR2={a:+.4f}   corr with repaired err={corr:+.3f}")
        rows.append({"arm": label, "adj_r2": a, "err_corr": corr})

    print("\n=== PRIMARY: does each radius add to the best no-topology stack? ===")
    out = []
    for label, arm in arms.items():
        st, _ = nested_stack({**base, "T": arm}, ["CatBoost", "repaired", "T"])
        a, r2 = _score(st)
        res = paired_adjacent_fast(noto, st, args.n_boot, seed=0)
        if res is None:
            continue
        clo, chi = _corrected(res["delta"], res["lo"], res["hi"], N_LOOKS)
        v = ("ADDS" if res["lo"] > 0 else "worse" if res["hi"] < 0
             else "not distinguishable")
        cv = ("ADDS" if clo > 0 else "worse" if chi < 0
              else "not distinguishable")
        print(f"  {label:24s} stack adjR2={a:+.4f}  "
              f"delta={res['delta']:+.4f} [{res['lo']:+.4f}, {res['hi']:+.4f}]  {v}")
        print(f"     {N_LOOKS}-look corrected [{clo:+.4f}, {chi:+.4f}]  {cv}")
        out.append({"arm": label, "stack_adj_r2": a, "stack_r2": r2, **res,
                    f"lo_{N_LOOKS}look": clo, f"hi_{N_LOOKS}look": chi,
                    "verdict": v, "verdict_corrected": cv})

    if out:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out).to_csv(OUT, index=False)
        pd.DataFrame(rows).to_csv(OUT.with_name("filt_mechanism.csv"), index=False)
        print(f"\n[filt] wrote {OUT}")
        adds = [o for o in out if o["arm"].startswith("F") and o["lo"] > 0]
        print(f"\n{len(adds)} of 2 alternative radii add nominally. Both adding "
              f"=> the finding is about the COMPLEX; neither => it is about the "
              f"3.5 A radius specifically, which is close to a tuning artefact "
              f"and must be reported as one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
