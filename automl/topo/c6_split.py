"""A fresh three-way extractant split for CAMPAIGN6.

Why a new one.  ``automl/artifacts/pi_sweep/split.json`` is a 50/50 tune/confirm
partition, and its confirm half has been scored on by the PI sweep, campaign 5,
the stack-fitting analysis and the encoder test.  A held-out set that has been
looked at four or five times is not held out, and 453 pairs cannot carry another
hundred screening cells.  The old split is left untouched -- it is what the
published numbers reproduce from.

Three ways, not two, because this campaign has three distinct jobs:

  screen  every sweep cell trains and is scored here (--restrict-groups
          screen_extractants.txt).  Spent freely, hundreds of looks.
  select  the shortlist is re-run on screen+select and ranked there, so the
          ordering is decided at near-full scale rather than at a third of the
          data.  One look per surviving cell.
  report  never enters ANY selection decision.  The final number is the metric
          restricted to these extractants' pairs.

Note what the third way does and does not require.  Under leave-extractants-out
CV every extractant is held out in some fold, so the final run trains on all 187
and its out-of-fold predictions for the report extractants are still honest.
What must be protected is not the fitting, it is the *choosing*: no cell was
ever ranked on these pairs.  That is why a single pre-declared look here needs
no multiplicity correction at all.

Balanced by adjacent-pair count rather than by extractant count, because the
metric is a mean over pairs and extractants carry between 0 and ~40 of them.
Same snake-draft rule as pi_split.py so the two are comparable, extended to
three ways.  Deterministic: no RNG, ordering is by (-pairs, name).

    python3 -m automl.topo.c6_split            # write it
    python3 -m automl.topo.c6_split --verify   # prove it is unchanged
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import automl.evaluation as ev

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/artifacts/c6_split"
WAYS = ("screen", "select", "report")
# Two screen slots per select/report slot -> ~50/25/25 of the pairs.
DRAFT = ("screen", "screen", "select", "report",
         "report", "select", "screen", "screen")


def pairs_per_extractant() -> pd.Series:
    """Adjacent-pair count each extractant contributes, from the evaluator.

    Built on the rows ``build_row_table`` actually models, NOT on the raw
    matrix cache.  The two differ: the cache has 187 extractants with a log D
    and a geometry build id, while the modelled set is the 162 whose complexes
    are present in the Vietoris-Rips asset.  Splitting the wrong population
    writes extractant names into the restrict files that train.py then rejects
    -- which is exactly how the first version of this file failed, in the smoke
    wave rather than in a campaign.
    """
    from automl.topo.train import build_row_table
    d, _X, _cols = build_row_table("baseline_2d", "snn")
    d = d.copy()
    y = d["log_D"].to_numpy(float)
    comp = d["composition_key"].to_numpy()
    li = d["lanthanide_index"].to_numpy(float)
    grp = d["extractant_group"].to_numpy()

    out = {}
    for g in pd.unique(grp):
        m = grp == g
        # p is irrelevant to the COUNT, so pass y for both.
        dy, _ = ev.adjacent_pair_arrays(y[m], y[m], comp[m], li[m])
        out[g] = len(dy)
    return pd.Series(out).sort_index()


def assign(counts: pd.Series) -> dict[str, list[str]]:
    """Snake draft over extractants ordered by (-pairs, name)."""
    order = sorted(counts.index, key=lambda g: (-int(counts[g]), str(g)))
    ways: dict[str, list[str]] = {w: [] for w in WAYS}
    load = {w: 0 for w in WAYS}
    zero_i = 0
    for g in order:
        n = int(counts[g])
        if n == 0:
            # Nothing to balance; alternate by name so the counts stay even.
            ways[WAYS[zero_i % len(WAYS)]].append(g)
            zero_i += 1
            continue
        # Draft slot by position, then break ties toward the emptiest way, so a
        # single 40-pair extractant cannot skew one partition for good.
        want = DRAFT[len([x for w in WAYS for x in ways[w]]) % len(DRAFT)]
        if load[want] > min(load.values()) + n:
            want = min(load, key=lambda w: load[w])
        ways[want].append(g)
        load[want] += n
    return ways


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and check the files on disk are identical")
    args = ap.parse_args()

    counts = pairs_per_extractant()
    ways = assign(counts)
    meta = {
        "rule": "three-way snake draft on adjacent-pair count, ordered by "
                "(-pairs, name); zero-pair extractants alternated by name",
        "purpose": {"screen": "every sweep cell; spent freely",
                    "select": "shortlist re-run and ranked; one look per cell",
                    "report": "ONE pre-declared configuration"},
        "n_extractants": int(len(counts)),
        "n_pairs_total": int(counts.sum()),
        "ways": {w: {"n_extractants": len(ways[w]),
                     "n_pairs": int(counts[ways[w]].sum())} for w in WAYS},
        "supersedes": "automl/artifacts/pi_sweep/split.json, whose confirm half "
                      "has been scored on by four prior campaigns",
    }
    for w in WAYS:
        meta[w] = sorted(ways[w])

    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(meta, indent=2, sort_keys=False) + "\n"
    files = {OUT / "split.json": payload}
    for w in WAYS:
        files[OUT / f"{w}_extractants.txt"] = "\n".join(sorted(ways[w])) + "\n"
    # Stage 2 trains on both, so the shortlist is ranked at near-full scale
    # rather than at the third of the data stage 1 can afford to burn.
    files[OUT / "screen_select_extractants.txt"] = (
        "\n".join(sorted(ways["screen"] + ways["select"])) + "\n")

    if args.verify:
        bad = [str(p) for p, txt in files.items()
               if not p.exists() or p.read_text() != txt]
        if bad:
            print("MISMATCH:", *bad, sep="\n  ")
            return 1
        h = hashlib.sha256(payload.encode()).hexdigest()[:16]
        print(f"[c6_split] verified, split.json sha256:{h}")
        return 0

    for p, txt in files.items():
        p.write_text(txt)
    print(f"[c6_split] {meta['n_extractants']} extractants, "
          f"{meta['n_pairs_total']} adjacent pairs")
    for w in WAYS:
        d = meta["ways"][w]
        print(f"  {w:7s} {d['n_extractants']:4d} extractants  "
              f"{d['n_pairs']:5d} pairs  "
              f"({100 * d['n_pairs'] / meta['n_pairs_total']:.1f}%)")
    print(f"[c6_split] wrote {OUT}")
    print(f"[c6_split] sha256:"
          f"{hashlib.sha256(payload.encode()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
