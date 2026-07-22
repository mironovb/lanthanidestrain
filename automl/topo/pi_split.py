#!/usr/bin/env python3
"""The frozen tune/confirm extractant split that protects the PI-sweep endpoint.

Why this exists
---------------
A hyperparameter sweep picks a winner, so the winner's score is optimistically
biased.  The protection used here is a split of *extractants*: the whole sweep
sees only the tune half, and the single winning configuration is scored once on
the confirm half, which no sweep run ever touched.  Because selection happens on
disjoint extractants, the confirm endpoint carries **no multiplicity penalty for
the number of configurations tried** -- only one extra look.

Why 50/50 rather than the more usual 2/3-1/3
--------------------------------------------
The ratio was calibrated from data, using the *already published* S0 simplicial
arm -- a different model from the one under test -- before any sweep run
existed.  Replaying S0's known +0.0381 stack gain through candidate splits:

    2/3-1/3   confirm 18 extractants /  302 pairs -> +0.0398 [-0.0005, +0.0549]
    50/50     confirm 35 extractants /  453 pairs -> +0.0475 [+0.0263, +0.0565]

Only 76 of the 162 extractants have any adjacent pair at all and the top five
carry ~36 % of them, so the effective cluster count is small.  A one-third
confirm side cannot detect an effect we already know is real, which means it
would report a null whatever the sweep found -- an uninformative test.  50/50
can, so 50/50 it is.

This buys a **positive control built into the endpoint**: the confirm half must
reproduce S0's known effect before the persistence-image number is allowed to
mean anything.  If it does, the harness demonstrably has the power to see an
effect of that size, and a null from the PI arm is then interpretable.

The split rule below is deterministic, depends on no random seed, and is hashed
into ``split.json``.  ``verify()`` recomputes it and refuses to proceed if the
hash moved, so the split cannot drift between the sweep and the confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from automl import evaluation as ev
from automl.dataset import GROUP_COL
from automl.topo.compare_arms import attach_meta

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "automl/reports"
OUT_DIR = REPO / "automl/artifacts/pi_sweep"
SPLIT_JSON = OUT_DIR / "split.json"

# The reference frame that defines the extractant universe and the adjacent
# pairs.  This is the repaired-baseline out-of-fold table every arm is scored
# against, so the split is defined on exactly the rows the endpoint uses.
REF_PARQUET = REPORTS / "oof_fcnn_std_scaler_ens16.parquet"


def reference() -> pd.DataFrame:
    return attach_meta(pd.read_parquet(REF_PARQUET)
                       .drop_duplicates("safe_exp_id")
                       .set_index("safe_exp_id"))


def pair_counts(ref: pd.DataFrame) -> pd.Series:
    """Adjacent pairs contributed by each extractant.

    Counted with ``adjacent_pair_arrays`` itself -- the same function the metric
    uses -- rather than a reimplemented |delta index| == 1 rule, so the split
    cannot disagree with the endpoint about what a pair is.
    """
    g = ref[GROUP_COL].astype(str)
    y = ref["y"].to_numpy(float)
    comp = ref["composition_key"].to_numpy()
    li = ref["lanthanide_index"].to_numpy()
    out = {}
    for ext in sorted(g.unique()):
        m = (g == ext).to_numpy()
        if m.sum() < 2:
            out[ext] = 0
            continue
        dy, _ = ev.adjacent_pair_arrays(y[m], y[m], comp[m], li[m])
        out[ext] = int(len(dy))
    return pd.Series(out, dtype=int)


def make_split(counts: pd.Series) -> tuple[list[str], list[str]]:
    """Deterministic 50/50 split balancing *adjacent pairs*, not extractants.

    Two passes, because the two kinds of extractant matter for different
    reasons:

    1. Pair-bearing extractants are greedily snake-drafted from largest to
       smallest so the two halves end up with nearly equal pair counts.  These
       determine the endpoint's precision.
    2. Zero-pair extractants contribute training rows but no pairs, so they are
       simply alternated by name.  Letting the pass-1 loop absorb them would
       dump every one of them into whichever half happened to be behind on
       pairs, silently starving the other half of training data.

    Ordering is by ``(-pairs, name)`` so ties never depend on dict or file
    ordering, and no random seed is involved anywhere.
    """
    bearing = sorted([e for e in counts.index if counts[e] > 0],
                     key=lambda e: (-int(counts[e]), str(e)))
    barren = sorted([e for e in counts.index if counts[e] == 0], key=str)

    tune: list[str] = []
    confirm: list[str] = []
    tp = cp = 0
    for ext in bearing:
        if cp <= tp:
            confirm.append(ext)
            cp += int(counts[ext])
        else:
            tune.append(ext)
            tp += int(counts[ext])
    for i, ext in enumerate(barren):
        (tune if i % 2 == 0 else confirm).append(ext)
    return sorted(tune), sorted(confirm)


def digest(tune: list[str], confirm: list[str]) -> str:
    h = hashlib.sha256()
    h.update(b"tune\n" + "\n".join(sorted(tune)).encode())
    h.update(b"\nconfirm\n" + "\n".join(sorted(confirm)).encode())
    return h.hexdigest()


def build() -> dict:
    ref = reference()
    counts = pair_counts(ref)
    tune, confirm = make_split(counts)
    assert not (set(tune) & set(confirm)), "tune and confirm overlap"
    assert set(tune) | set(confirm) == set(counts.index), "split loses extractants"
    return {
        "rule": "pairs-balanced snake draft on adjacent-pair count, "
                "ordered by (-pairs, name); zero-pair extractants alternated by name",
        "n_extractants": int(len(counts)),
        "n_pairs_total": int(counts.sum()),
        "tune": tune,
        "confirm": confirm,
        "n_tune": len(tune),
        "n_confirm": len(confirm),
        "tune_pairs": int(counts[tune].sum()),
        "confirm_pairs": int(counts[confirm].sum()),
        "sha256": digest(tune, confirm),
    }


def load() -> dict:
    if not SPLIT_JSON.exists():
        raise SystemExit(f"no frozen split at {SPLIT_JSON}; run --freeze first")
    return json.loads(SPLIT_JSON.read_text())


def verify() -> bool:
    """Recompute the split and compare to the frozen file."""
    frozen = load()
    fresh = build()
    same = frozen["sha256"] == fresh["sha256"]
    print(f"[pi-split] frozen  {frozen['sha256'][:16]}  "
          f"tune {frozen['n_tune']}/{frozen['tune_pairs']}p  "
          f"confirm {frozen['n_confirm']}/{frozen['confirm_pairs']}p")
    print(f"[pi-split] fresh   {fresh['sha256'][:16]}")
    print("[pi-split] " + ("MATCH" if same else "DRIFTED -- refusing to proceed"))
    return same


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--freeze", action="store_true",
                    help="compute and write the split (refuses to overwrite)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--positive-control", action="store_true",
                    help="measure the published S0 stack gain on the confirm "
                         "half; this is the number the endpoint is gated on")
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    if args.freeze:
        if SPLIT_JSON.exists():
            raise SystemExit(f"{SPLIT_JSON} already exists; a frozen split is "
                             "never silently overwritten")
        rec = build()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        SPLIT_JSON.write_text(json.dumps(rec, indent=2) + "\n")
        (OUT_DIR / "tune_extractants.txt").write_text("\n".join(rec["tune"]) + "\n")
        (OUT_DIR / "confirm_extractants.txt").write_text(
            "\n".join(rec["confirm"]) + "\n")
        print(f"[pi-split] froze {rec['n_extractants']} extractants "
              f"({rec['n_pairs_total']} pairs)")
        print(f"[pi-split]   TUNE    {rec['n_tune']:3d} ext  "
              f"{rec['tune_pairs']:4d} pairs")
        print(f"[pi-split]   CONFIRM {rec['n_confirm']:3d} ext  "
              f"{rec['confirm_pairs']:4d} pairs "
              f"({100*rec['confirm_pairs']/rec['n_pairs_total']:.0f}%)")
        print(f"[pi-split]   sha256  {rec['sha256']}")

    if args.verify and not verify():
        return 1

    if args.positive_control:
        rc = positive_control(args.n_boot)
        if rc:
            return rc
    return 0


def positive_control(n_boot: int = 400) -> int:
    """Does the confirm half see S0's known effect?

    Imported lazily: this pulls in the whole stacking machinery, which the
    freeze path does not need.
    """
    from automl.topo.best_stack import nested_stack, _score
    from automl.topo.control_factorial import (ensemble, load_cells,
                                               paired_adjacent_fast)
    from automl.topo.compare_arms import collect

    cells = load_cells(verbose=False)
    s0 = ensemble(cells["S0"])
    a0, _ = _score(s0)
    print(f"\nharness check: published S0 re-ensembles to {a0:+.4f} "
          "(must be +0.2382)")
    if abs(a0 - 0.2382) > 5e-4:
        raise SystemExit("published S0 drifted; refusing to report")

    ref = reference()
    base = {"CatBoost": attach_meta(collect()["baseline::catboost::none"]),
            "repaired": ref}
    noto, _ = nested_stack(base, ["CatBoost", "repaired"])
    st, _ = nested_stack({**base, "T": s0}, ["CatBoost", "repaired", "T"])

    rec = load()
    rows = []
    for name, exts in (("full data", None),
                       ("TUNE half", rec["tune"]),
                       ("CONFIRM half", rec["confirm"])):
        if exts is None:
            n, s = noto, st
        else:
            keep = ref.index[ref[GROUP_COL].astype(str).isin(set(exts))]
            n = noto.loc[noto.index.intersection(keep)]
            s = st.loc[st.index.intersection(keep)]
        res = paired_adjacent_fast(n, s, n_boot, seed=0)
        clears = res["lo"] > 0
        print(f"  S0 stack gain, {name:13s} {res['delta']:+.4f} "
              f"[{res['lo']:+.4f}, {res['hi']:+.4f}]  "
              f"{'CLEARS 0' if clears else 'does not clear'}")
        rows.append({"scope": name, **{k: res[k] for k in ("delta", "lo", "hi")},
                     "clears_zero": bool(clears)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "positive_control.csv", index=False)

    conf = rows[-1]
    if not conf["clears_zero"]:
        print("\n[pi-split] POSITIVE CONTROL FAILED: the confirm half cannot "
              "detect an effect we know is real, so a null from the PI arm "
              "would be uninterpretable.")
        return 1
    print(f"\n[pi-split] positive control passes: the confirm half detects S0 at "
          f"{conf['delta']:+.4f} [{conf['lo']:+.4f}, {conf['hi']:+.4f}], so it "
          f"has the power to see a PI effect of comparable size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
