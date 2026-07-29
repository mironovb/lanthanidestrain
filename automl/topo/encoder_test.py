#!/usr/bin/env python3
"""Is it *simplicial*, or merely *3D message passing*?

Pre-registered in ``automl/reports/ENCODER_PREREGISTRATION.md`` (commit 6abaf35,
plus amendment 1), committed before either arm had been run.

Two new arms, both over the *same* Vietoris-Rips edges as the published S0:

* **G0** ``--no-triangles`` -- the same ``SimplicialNet`` with the 2-simplex
  level removed, so message passing happens over the **graph** of the same
  complex.  Isolates the triangles and nothing else.
* **D0** ``--arch dist`` -- SchNet-style continuous filters over interatomic
  distance, no boundary maps and no filtration anywhere.  Isolates the
  simplicial construction from 3D message passing in general.

Both use the published S0 configuration and the same 16 seeds, so the
inner-validation splits and batch order match and the paired bootstrap compares
arms rather than splits.

Nothing here is trained; every out-of-fold vector already exists on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.adjacent_test import adj_r2
from automl.topo.best_stack import nested_stack, _score
from automl.topo.compare_arms import attach_meta
from automl.topo.dualkey_test import (BINNED, STRICT, KEYS, attach_strict,
                                      load_frames, paired_adjacent_corrected,
                                      _verdict)
from automl.topo.stack_test import _corrected

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/topo_encoder"
REPORTS = REPO / "automl/reports"
OUT = REPORTS / "encoder_test.csv"
OUT_ARMS = REPORTS / "encoder_arms.csv"

SEEDS = [7, 11, 23, 37, 42, 51, 67, 83, 211, 223, 233, 241, 251, 263, 271, 281]

# Cell membership comes from the recorded configuration, never from the tag --
# the same rule control_factorial uses, and for the same reason: a tag is a
# label someone typed, a config is what the run actually did.
ARMS = {
    "G0": {"arch": "snn", "no_triangles": True,
           "label": "SNN, triangles removed (graph over the same complex)"},
    "D0": {"arch": "dist", "no_triangles": False,
           "label": "continuous-filter distance net (no simplices at all)"},
}

# Pre-registered: three new contrasts take the topology claim's look count from
# 10 (after the dual-key re-analysis) to 13.
N_LOOKS = 13


def load_arm(arm: str, verbose: bool = True) -> pd.DataFrame | None:
    """Ensemble every seed of one arm, matched on configuration."""
    spec = ARMS[arm]
    found: dict[int, Path] = {}
    if not ART.exists():
        return None
    for j in sorted(ART.glob("run_*.json")):
        cfg = json.loads(j.read_text()).get("config", {})
        if cfg.get("arch") != spec["arch"]:
            continue
        if bool(cfg.get("no_triangles", False)) != spec["no_triangles"]:
            continue
        if cfg.get("preset") != "baseline_2d":
            continue
        if float(cfg.get("pair_loss_weight") or 0.0) != 2.0:
            continue
        if (cfg.get("select_on") or "mse") != "adjacent":
            continue
        if cfg.get("level_weight") is not None:
            continue
        seed = int(cfg.get("seed", -1))
        if seed not in SEEDS:
            continue
        p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
        if not p.exists():
            continue
        if seed in found:
            raise RuntimeError(f"arm {arm} seed {seed} matched twice: "
                               f"{found[seed].name} and {p.name}")
        found[seed] = p
    if verbose:
        missing = sorted(set(SEEDS) - set(found))
        print(f"  {arm:3s} {spec['label']:52s} seeds={len(found):2d}/16"
              + (f"  MISSING {missing}" if missing else ""))
    if not found:
        return None
    frames = {s: pd.read_parquet(p).drop_duplicates("safe_exp_id")
              .set_index("safe_exp_id") for s, p in sorted(found.items())}
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    stack = np.vstack([frames[s].loc[idx, "oof"].to_numpy(float)
                       for s in sorted(frames)])
    ens = frames[sorted(frames)[0]].loc[idx].copy()
    # Mean over EVERY seed present, never a subset: choosing which replicates to
    # average on the scored metric would manufacture the result.
    ens["oof"] = stack.mean(axis=0)
    return attach_strict(attach_meta(ens))


def _err_corr(a: pd.DataFrame, b: pd.DataFrame, key: str) -> float:
    """Correlation of the two arms' adjacent-pair errors.

    One of the two axes of the mechanism rule: an arm earns a stack slot only if
    it is both accurate on the scored metric and making *different* errors.
    """
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    y = a["y"].to_numpy(float)
    comp, li = a[key].to_numpy(), a["lanthanide_index"].to_numpy()
    dy, dpa = ev.adjacent_pair_arrays(y, a["oof"].to_numpy(float), comp, li)
    _, dpb = ev.adjacent_pair_arrays(y, b["oof"].to_numpy(float), comp, li)
    ea, eb = dy - dpa, dy - dpb
    if ea.std() < 1e-12 or eb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ea, eb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--allow-partial", action="store_true",
                    help="report with fewer than 16 seeds per arm; the result "
                         "is then NOT the pre-registered endpoint and is "
                         "labelled as an interim read")
    args = ap.parse_args()

    print("=== arms ===")
    frames = load_frames()
    new = {a: load_arm(a) for a in ARMS}
    have = {a: f for a, f in new.items() if f is not None}
    if not have:
        print("\nNo encoder runs on disk yet. Nothing to report.")
        return 1

    # Refuse to report the pre-registered endpoint on a partial ensemble, the
    # same standing rule control_factorial applies.
    partial = [a for a, f in have.items()
               if len(sorted(ART.glob(f"run_*.json"))) and len(f) and
               _n_seeds(a) < len(SEEDS)]
    if partial and not args.allow_partial:
        print(f"\n[encoder] arms {partial} are incomplete. The pre-registered "
              f"endpoint needs all 16 seeds; rerun with --allow-partial for an "
              f"interim read that is NOT the endpoint.")
        return 1
    interim = bool(partial)

    frames.update(have)
    s0_pub, _ = _score(frames["S0"], BINNED)
    print(f"\n  standing precondition: S0 re-ensembles to {s0_pub:+.4f} "
          f"(published +0.2382) -> "
          f"{'OK' if abs(s0_pub - 0.2382) < 5e-4 else 'DRIFT'}")
    if abs(s0_pub - 0.2382) >= 5e-4:
        raise SystemExit("S0 has drifted; that is a bug, not a result.")

    arm_rows = []
    print("\n=== single arms (adjacent-pair R2, and error correlation with the "
          "repaired baseline) ===")
    for name in ["CatBoost", "repaired", "S0", "T0w"] + list(have):
        a_b, r = _score(frames[name], BINNED)
        a_s, _ = _score(frames[name], STRICT)
        ec = _err_corr(frames[name], frames["repaired"], BINNED)
        print(f"  {name:10s} binned={a_b:+.4f}  strict={a_s:+.4f}  "
              f"overall={r:+.4f}  err-corr={ec:.3f}")
        arm_rows.append({"arm": name, "adj_r2_binned": a_b,
                         "adj_r2_strict": a_s, "r2_overall": r,
                         "err_corr_with_repaired": ec})

    contrasts = []
    combos = {
        "no topology": ["CatBoost", "repaired"],
        "with S0": ["CatBoost", "repaired", "S0"],
    }
    for a in have:
        combos[f"with {a}"] = ["CatBoost", "repaired", a]
    pairs = [("no topology", f"with {a}",
              f"does {ARMS[a]['label']} add to the best no-topology stack?")
             for a in have]
    if "D0" in have:
        pairs.append(("with D0", "with S0",
                      "is the simplicial arm better than a plain 3D net "
                      "in the same slot?"))

    rows = []
    for key in KEYS:
        label = "binned (published)" if key == BINNED else "STRICT"
        print(f"\n=== {label} ===")
        built = {}
        for name, names in combos.items():
            fr, ws = nested_stack(frames, names, key_col=key)
            built[name] = fr
            adj, r2 = _score(fr, key)
            wtxt = ", ".join(f"{n}={np.median(ws[:, i]):.2f}"
                             for i, n in enumerate(names))
            print(f"  {name:16s} adjR2={adj:+.4f}  R2={r2:+.4f}   w: {wtxt}")
        for base, arm, q in pairs:
            r = paired_adjacent_corrected(built[base], built[arm], args.n_boot,
                                          seed=0, key_col=key)
            if r is None:
                continue
            clo, chi = _corrected(r["delta"], r["lo"], r["hi"], N_LOOKS)
            v, cv = _verdict(r["lo"], r["hi"]), _verdict(clo, chi)
            print(f"    {arm} minus {base}: delta={r['delta']:+.4f} "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}] {v} | "
                  f"{N_LOOKS}-look [{clo:+.4f}, {chi:+.4f}] {cv}")
            rows.append({"key": key, "base": base, "arm": arm, "question": q,
                         **r, f"lo_{N_LOOKS}look": clo,
                         f"hi_{N_LOOKS}look": chi,
                         "verdict": v, "verdict_corrected": cv,
                         "interim": interim})

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    pd.DataFrame(arm_rows).to_csv(OUT_ARMS, index=False)

    print("\n=== pre-registered reading (ENCODER_PREREGISTRATION.md sec 4) ===")
    if interim:
        print("  INTERIM -- not all 16 seeds present; this is NOT the endpoint.")

    def adds(base, arm, key=BINNED):
        m = frame[(frame["key"] == key) & (frame["base"] == base)
                  & (frame["arm"] == arm)]
        return bool(len(m) and (m["verdict_corrected"] == "adds").all())

    d0 = adds("no topology", "with D0") if "D0" in have else None
    g0 = adds("no topology", "with G0") if "G0" in have else None
    s0_vs_d0 = adds("with D0", "with S0") if "D0" in have else None
    print(f"  G0 (graph, no triangles) adds : {g0}")
    print(f"  D0 (no simplices at all) adds : {d0}")
    print(f"  S0 beats D0 in the same slot  : {s0_vs_d0}")
    if d0 and not s0_vs_d0:
        print("\n  ==> THE CLAIM BROADENS. A non-simplicial 3D encoder adds as "
              "much, so the effect is 'a learned 3D representation', not 'a "
              "simplicial complex'. The contribution is the RULE for what a "
              "candidate representation must satisfy; VR is one instance.")
    elif not d0:
        print("\n  ==> THE CLAIM IS BOUNDED to simplicial message passing. A "
              "plain 3D encoder over the same edges does not add, so the "
              "persistence-image null is no longer the only evidence for "
              "specificity.")
    elif s0_vs_d0:
        print("\n  ==> Both help, and the simplicial arm is better in the slot. "
              "The claim survives in its present form and gains a second "
              "supporting representation.")
    print(f"\n[encoder] wrote {OUT}\n[encoder] wrote {OUT_ARMS}")
    return 0


def _n_seeds(arm: str) -> int:
    spec = ARMS[arm]
    n = 0
    for j in sorted(ART.glob("run_*.json")):
        cfg = json.loads(j.read_text()).get("config", {})
        if (cfg.get("arch") == spec["arch"]
                and bool(cfg.get("no_triangles", False)) == spec["no_triangles"]
                and int(cfg.get("seed", -1)) in SEEDS):
            p = j.with_name(j.name.replace("run_", "oof_")).with_suffix(".parquet")
            if p.exists():
                n += 1
    return n


if __name__ == "__main__":
    raise SystemExit(main())
