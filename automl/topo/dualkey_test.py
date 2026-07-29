#!/usr/bin/env python3
"""Does the topology result survive a stricter definition of "identical conditions"?

Pre-registered in ``automl/reports/DUALKEY_PREREGISTRATION.md`` (commit 836ad30),
committed before any of the contrasts below existed.

The defect
----------
The headline metric blocks by ``composition_key``, which is built from *binned*
condition columns.  ``dataset.py`` builds a second key eight lines later and says
why the first is not good enough:

    Strict variant: every numeric condition included, so the *only* thing that
    varies inside a block is the lanthanide.  Delta learning needs this -- with
    the binned key two rows can share a block while differing in extractant
    concentration, which turns a real log D difference into label noise.

``strict_composition_key`` has existed for the whole study.  The headline metric
has never been computed with it.  That matters because the claim is specifically
about differences between two lanthanides *measured under identical conditions*;
if "identical" means "in the same bin", part of the predicted quantity is
condition effect rather than selectivity.

What this computes
------------------
Both published confirmatory contrasts, under both keys, with the
**multiplicity-respecting** cluster bootstrap (each drawn copy of an extractant
tagged so a twice-drawn cluster counts twice -- the correction measured in
``bootstrap_check.py``, which found the published intervals 12-29% too narrow).

Nothing is trained.  Every out-of-fold vector already exists on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.topo.adjacent_test import adj_r2
from automl.topo.best_stack import nested_stack, _score
from automl.topo.compare_arms import attach_meta, collect
from automl.topo.control_factorial import (ensemble, load_cells,
                                           _assert_fast_matches)
from automl.topo.stack_test import _corrected

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
OUT_CONTRASTS = REPORTS / "dualkey_test.csv"
OUT_ARMS = REPORTS / "dualkey_arms.csv"

BINNED = "composition_key"
STRICT = "strict_composition_key"
KEYS = (BINNED, STRICT)

# Pre-registered in DUALKEY_PREREGISTRATION.md sec 3: this re-analysis is a new
# look at the same question, taking the topology claim's look count from 8 to 10.
N_LOOKS = 10


# ---------------------------------------------------------------------------
def attach_strict(d: pd.DataFrame) -> pd.DataFrame:
    """Join ``strict_composition_key`` on ``safe_exp_id``.

    Kept separate from ``compare_arms.attach_meta`` on purpose: that function
    defines the metadata every published arm carries, and widening it would make
    every existing arm's provenance depend on this re-analysis.
    """
    if STRICT in d.columns:
        return d
    from automl.matrix_cache import load_cache
    src, _, _ = load_cache()
    add = (src[["safe_exp_id", STRICT]].drop_duplicates("safe_exp_id")
           .set_index("safe_exp_id"))
    out = d.join(add, how="left")
    if out[STRICT].isna().any():
        raise RuntimeError(f"{int(out[STRICT].isna().sum())} rows have no "
                           f"{STRICT}; the join key is wrong")
    return out


def assert_nested(d: pd.DataFrame, key_col: str) -> None:
    """A block must never span two extractants, or the cluster bootstrap is wrong.

    The fast path and the whole resampling argument rest on this.  It is true by
    construction for both keys -- both are prefixed with the extractant -- but
    'true by construction' is what was said about the collapsing bootstrap too,
    so it is checked.
    """
    n = d.groupby(key_col)["extractant_group"].nunique()
    bad = int((n > 1).sum())
    if bad:
        raise RuntimeError(f"{bad} {key_col} blocks span more than one "
                           f"extractant; the cluster bootstrap is invalid")


# ---------------------------------------------------------------------------
def paired_adjacent_corrected(a: pd.DataFrame, b: pd.DataFrame, n_boot: int,
                              seed: int = 0, key_col: str = BINNED
                              ) -> dict | None:
    """Paired cluster bootstrap, multiplicity respected, on either block key.

    Differs from ``control_factorial.paired_adjacent_fast`` in exactly one way:
    a drawn copy of an extractant gets its own block, by suffixing the copy index
    onto the block key.  Without that a twice-drawn cluster contributes
    *identically* to a once-drawn one -- the published draw was an m-out-of-n
    subsample, not a cluster bootstrap, and its intervals came out 12-29% narrow
    (``bootstrap_audit.csv``).

    The fast precomputed path cannot be reused here: it keys on the set of drawn
    clusters, which is precisely the information the correction restores.
    """
    common = a.index.intersection(b.index)
    if len(common) < 0.5 * min(len(a), len(b)):
        return None
    a, b = a.loc[common], b.loc[common]
    y = a["y"].to_numpy(float)
    pa, pb = a["oof"].to_numpy(float), b["oof"].to_numpy(float)
    comp = a[key_col].to_numpy().astype(str)
    li = a["lanthanide_index"].to_numpy()
    gcodes, guniq = pd.factorize(a["extractant_group"].to_numpy())
    rows_by_g = [np.flatnonzero(gcodes == i) for i in range(len(guniq))]

    obs_a = adj_r2(y, pa, comp, li)
    obs_b = adj_r2(y, pb, comp, li)

    rng = np.random.default_rng(seed)
    da, db, dd = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows_by_g), len(rows_by_g))
        idx = np.concatenate([rows_by_g[g] for g in pick])
        tag = np.concatenate([np.full(len(rows_by_g[g]), c)
                              for c, g in enumerate(pick)])
        ck = np.char.add(np.char.add(comp[idx], "#"), tag.astype(str))
        va = adj_r2(y[idx], pa[idx], ck, li[idx])
        vb = adj_r2(y[idx], pb[idx], ck, li[idx])
        if np.isfinite(va) and np.isfinite(vb):
            da.append(va); db.append(vb); dd.append(vb - va)
    if len(dd) < 30:
        return None
    dd = np.asarray(dd)
    return {"baseline_obs": obs_a, "arm_obs": obs_b,
            "arm_lo": float(np.percentile(db, 5)),
            "arm_hi": float(np.percentile(db, 95)),
            "delta": float(dd.mean()),
            "lo": float(np.percentile(dd, 5)),
            "hi": float(np.percentile(dd, 95)),
            "p_better": float((dd > 0).mean()),
            "n_boot": len(dd)}


def _verdict(lo: float, hi: float) -> str:
    return ("adds" if lo > 0 else "worse" if hi < 0 else "not distinguishable")


# ---------------------------------------------------------------------------
COMBOS = {
    "full (CatBoost+repaired+S0)": ["CatBoost", "repaired", "S0"],
    "no topology (CatBoost+repaired)": ["CatBoost", "repaired"],
    "topology swapped for control": ["CatBoost", "repaired", "T0w"],
}
CONTRASTS = [
    ("no topology (CatBoost+repaired)", "full (CatBoost+repaired+S0)",
     "drop-in: does adding S0 to the best no-topology stack help?"),
    ("topology swapped for control", "full (CatBoost+repaired+S0)",
     "swap: S0 vs the matched tabular control in the same slot"),
]


def load_frames() -> dict[str, pd.DataFrame]:
    """The four components, exactly as ``best_stack.main`` assembles them."""
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
    return {k: attach_strict(v) for k, v in frames.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    frames = load_frames()

    # --- guards, before any number is reported ------------------------------
    print("=== guards ===")
    for key in KEYS:
        assert_nested(frames["S0"], key)
        g = frames["S0"]["extractant_group"].to_numpy()
        _assert_fast_matches(frames["S0"], g, n_checks=10, key_col=key)
        nb = frames["S0"].groupby(key).ngroups
        print(f"  {key:26s} blocks={nb:5d}  nested in extractant OK  "
              f"fast path == adjacent_pair_metrics OK")

    # standing precondition: the published S0 ensemble must not have moved
    s0_pub, _ = _score(frames["S0"], BINNED)
    print(f"  published S0 re-ensembles to {s0_pub:+.4f} "
          f"(must be +0.2382): {'OK' if abs(s0_pub - 0.2382) < 5e-4 else 'DRIFT'}")
    if abs(s0_pub - 0.2382) >= 5e-4:
        raise SystemExit("S0 has drifted from its published value -- that is a "
                         "bug, not a result; refusing to report.")

    # --- single arms, both keys --------------------------------------------
    arm_rows = []
    print("\n=== single arms ===")
    print(f"  {'arm':12s} {'binned':>10s} {'strict':>10s} {'overall R2':>11s}")
    for name in ("CatBoost", "repaired", "S0", "T0w"):
        a_b, r = _score(frames[name], BINNED)
        a_s, _ = _score(frames[name], STRICT)
        print(f"  {name:12s} {a_b:+10.4f} {a_s:+10.4f} {r:+11.4f}")
        arm_rows.append({"arm": name, "adj_r2_binned": a_b,
                         "adj_r2_strict": a_s, "r2_overall": r})

    # --- stacks and contrasts, both keys -----------------------------------
    rows = []
    for key in KEYS:
        label = "binned (published)" if key == BINNED else "STRICT"
        print(f"\n=== {label}: nested stacks "
              f"(weights fitted AND scored under {key}) ===")
        built = {}
        for name, names in COMBOS.items():
            fr, ws = nested_stack(frames, names, key_col=key)
            built[name] = fr
            a, r = _score(fr, key)
            wtxt = ", ".join(f"{n}={np.median(ws[:, i]):.2f}"
                             for i, n in enumerate(names))
            print(f"  {name:34s} adjR2={a:+.4f}  R2={r:+.4f}   median w: {wtxt}")

        print(f"\n--- {label}: pre-registered contrasts "
              f"(multiplicity-respecting bootstrap, {N_LOOKS}-look Bonferroni) ---")
        for base, arm, question in CONTRASTS:
            r = paired_adjacent_corrected(built[base], built[arm], args.n_boot,
                                          seed=0, key_col=key)
            if r is None:
                print(f"  {question}: insufficient overlap"); continue
            clo, chi = _corrected(r["delta"], r["lo"], r["hi"], N_LOOKS)
            v, cv = _verdict(r["lo"], r["hi"]), _verdict(clo, chi)
            print(f"  {arm}\n    minus {base}")
            print(f"    delta={r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
                  f"P={r['p_better']:.2f}   {v}")
            print(f"    {N_LOOKS}-look corrected [{clo:+.4f}, {chi:+.4f}]   {cv}")
            rows.append({"key": key, "base": base, "arm": arm,
                         "question": question, **r,
                         f"lo_{N_LOOKS}look": clo, f"hi_{N_LOOKS}look": chi,
                         "verdict": v, "verdict_corrected": cv})

    # --- the pre-registered decision ---------------------------------------
    frame = pd.DataFrame(rows)
    OUT_CONTRASTS.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CONTRASTS, index=False)
    pd.DataFrame(arm_rows).to_csv(OUT_ARMS, index=False)

    print("\n=== pre-registered decision (DUALKEY_PREREGISTRATION.md sec 4) ===")
    passed = {}
    for key in KEYS:
        sub = frame[frame["key"] == key]
        passed[key] = bool(len(sub) == len(CONTRASTS)
                           and (sub["verdict_corrected"] == "adds").all())
        print(f"  {key:26s} both contrasts add after {N_LOOKS}-look "
              f"Bonferroni: {passed[key]}")
    if passed[BINNED] and passed[STRICT]:
        verdict = ("CLAIM STANDS AND IS STRENGTHENED -- the effect is not an "
                   "artefact of the binning. Report both columns from here on; "
                   "the strict key becomes primary for all new work.")
    elif passed[BINNED]:
        verdict = ("CLAIM DOWNGRADED -- survives only under the binned key. The "
                   "published effect is partly an artefact of averaging "
                   "measurements taken under different conditions into one "
                   "cell. Rewrite the headline; state the strict-key null in "
                   "the abstract.")
    elif passed[STRICT]:
        verdict = ("BINNED KEY IS THE DEFECTIVE ONE, as dataset.py:387 argues. "
                   "Re-baseline the study on the strict key; published numbers "
                   "are superseded, not supplemented.")
    else:
        verdict = ("PUBLISHED RESULT DOES NOT REPRODUCE under the corrected "
                   "multiplicity-respecting bootstrap under either key. That is "
                   "a larger problem than the key and is reported first.")
    print(f"\n  ==> {verdict}")
    print(f"\n[dualkey] wrote {OUT_CONTRASTS}")
    print(f"[dualkey] wrote {OUT_ARMS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
