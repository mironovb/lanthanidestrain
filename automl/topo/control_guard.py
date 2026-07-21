#!/usr/bin/env python3
"""Prove that running the control did not move a single committed number.

The control asks whether the +0.243 adjacent-pair gain is attributable to
topology or to the training objective.  That is a question about *attribution*;
the measurements already committed in ``4769e76`` are facts and adding a new arm
cannot change them.  This module makes that claim checkable rather than
rhetorical.

Two tiers, because they have different rules
--------------------------------------------
**frozen** -- the out-of-fold predictions every published test was computed
from, plus the result CSVs and figures derived from them.  These must be
byte-identical before and after.  Any change means the harness drifted and no
downstream number is trustworthy, so ``--verify`` exits non-zero.

**append-only** -- the four report documents.  Per the agreed policy these keep
every number and word; the adverse outcome adds *one* pointer line at the top
rather than editing anything.  So they are checked for prefix-preservation
(old content still present, in order) instead of equality, and a body edit is
reported as a failure while a pure addition is not.

Why hash the inputs and not only the outputs: a result CSV can be reproduced
identically from changed inputs if two errors cancel, and an OOF parquet that
silently changed would invalidate every one of the five tests at once.  The
parquets are the root of trust here, so they are what gets pinned.

Usage
-----
    python3 -m automl.topo.control_guard --snapshot   # before anything runs
    python3 -m automl.topo.control_guard --verify     # after the analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAP_DIR = REPO / "automl/artifacts/topo_control/_baseline_snapshot"
MANIFEST = SNAP_DIR / "manifest.json"

# Every artifact directory a reported test reads from.  topo_control is now
# included: the control is finished and published, so its runs are as frozen as
# the ones that came before.  The directories the *next* experiment writes to
# (topo_s2, vr_conformers) are deliberately absent -- they are expected to grow.
FROZEN_ART_DIRS = ("topo_runs", "topo_adjacent", "topo_adj_seeds",
                   "topo_runs_radial", "topo_tight", "topo_control")

FROZEN_REPORTS = ("adjacent_pair_test.csv", "adjacent_ensemble.csv",
                  "adjacent_blend.csv", "topo_comparison.csv",
                  "topo_comparison_paired.csv", "scatter_diagnostic.csv",
                  # the control's own outputs, published in 6ddf853 / 4b737bf
                  "control_cells.csv", "control_factorial.csv",
                  "control_attribution.csv", "control_blend.csv",
                  "bootstrap_audit.csv", "fcnn_diagnostic.csv",
                  "fixed_baseline_test.csv",
                  # the repaired-baseline predictions the S2 endpoint is scored
                  # against; if these moved, the endpoint would move with them
                  "oof_fcnn_std_scaler_ens16.parquet")

APPEND_ONLY = ("PI_REPORT.md", "PUBLICATION_ASSESSMENT.md",
               "TOPOLOGY_RESULTS.md", "TOPOLOGY_METHODS.md",
               "CONTROL_RESULTS.md", "CONTROL_PREREGISTRATION.md")

# Source files whose content decides what a run means.  Recorded rather than
# frozen: this study once lost a batch to an array picking up a mid-flight edit,
# so every run should be attributable to a known source state.
SOURCE_FILES = ("automl/topo/train.py", "automl/topo/snn.py",
                "automl/topo/pi_cnn.py", "automl/topo/tabular_net.py",
                "automl/topo/simplicial_data.py", "automl/evaluation.py")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _frozen_paths() -> list[Path]:
    out: list[Path] = []
    for d in FROZEN_ART_DIRS:
        root = REPO / "automl/artifacts" / d
        if root.exists():
            out += sorted(root.glob("oof_*.parquet"))
            out += sorted(root.glob("run_*.json"))
            out += sorted(root.glob("results.jsonl"))
    for name in FROZEN_REPORTS:
        p = REPO / "automl/reports" / name
        if p.exists():
            out.append(p)
    figs = REPO / "automl/reports/figures"
    if figs.exists():
        # Only the topological figures; the earlier study's figures are not at
        # risk from this work and hashing them would add noise, not safety.
        out += sorted(figs.glob("topo_*.png")) + sorted(figs.glob("topo_*.pdf"))
    # The FCNN diagnostic's per-mode summaries, which the attribution quotes.
    out += sorted((REPO / "automl/reports").glob("fcnn_diagnostic_*.json"))
    return out


def _rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def snapshot() -> int:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    frozen = {_rel(p): _sha(p) for p in _frozen_paths()}
    append = {}
    for name in APPEND_ONLY:
        p = REPO / "automl/reports" / name
        if p.exists():
            append[_rel(p)] = p.read_text()
    source = {}
    for rel in SOURCE_FILES:
        p = REPO / rel
        if p.exists():
            source[rel] = _sha(p)
    MANIFEST.write_text(json.dumps(
        {"frozen": frozen, "append_only": append, "source": source},
        indent=2, sort_keys=True))
    print(f"[guard] pinned {len(frozen)} frozen files "
          f"({sum(1 for k in frozen if k.endswith('.parquet'))} OOF parquets), "
          f"{len(append)} append-only reports, {len(source)} source files")
    print(f"[guard] manifest -> {_rel(MANIFEST)}")
    return 0


def verify() -> int:
    if not MANIFEST.exists():
        print(f"[guard] no manifest at {_rel(MANIFEST)} -- run --snapshot first")
        return 2
    man = json.loads(MANIFEST.read_text())
    fail: list[str] = []

    changed, missing = [], []
    for rel, want in man["frozen"].items():
        p = REPO / rel
        if not p.exists():
            missing.append(rel)
        elif _sha(p) != want:
            changed.append(rel)
    # A file appearing in a frozen directory is fine (nothing published reads
    # it); a file changing or vanishing is not.
    if changed:
        fail += [f"MODIFIED  {r}" for r in changed]
    if missing:
        fail += [f"MISSING   {r}" for r in missing]

    for rel, old in man["append_only"].items():
        p = REPO / rel
        if not p.exists():
            fail.append(f"MISSING   {rel}")
            continue
        new = p.read_text()
        if new == old:
            continue
        # Additions are allowed anywhere; deletions and edits are not.  Checking
        # that every old line still occurs in order catches a changed number
        # (the old line disappears) while permitting an inserted pointer.
        it = iter(new.splitlines())
        if all(any(line == n for n in it) for line in old.splitlines()):
            print(f"[guard] {rel}: content preserved, "
                  f"{len(new.splitlines()) - len(old.splitlines())} line(s) added")
        else:
            fail.append(f"EDITED    {rel}  (an existing line was changed or removed)")

    drift = [r for r, want in man["source"].items()
             if (REPO / r).exists() and _sha(REPO / r) != want]
    if drift:
        # Not a failure by itself -- source is expected to change while the
        # control is being built.  It is reported so that a run can always be
        # attributed to the source state it actually used.
        print(f"[guard] source changed since snapshot: {', '.join(drift)}")

    if fail:
        print(f"\n[guard] FAILED -- {len(fail)} committed artefact(s) moved:")
        for f in fail:
            print(f"   {f}")
        return 1
    print(f"[guard] OK -- {len(man['frozen'])} frozen artefacts byte-identical, "
          f"{len(man['append_only'])} reports content-preserved")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot", action="store_true")
    g.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    return snapshot() if args.snapshot else verify()


if __name__ == "__main__":
    sys.exit(main())
