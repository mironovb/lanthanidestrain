#!/usr/bin/env python3
"""Mulliken charges for the re-optimised conformers.

Why this exists
---------------
Stage 1 re-optimised every complex in water and n-octanol, and Stage 2 asked
only whether the tighter geometries were better *replacements* -- they were not
(median RMSD 1.87 A: different conformers, not refinements).  They were never
used as an **ensemble**, which is what this run enables: ~3,100 structures
instead of 956, a 3.2x increase in the sample size that governs overfitting for
the simplicial encoder.

The blocker is that ``reoptimize.write_extxyz`` saved coordinates and a
molecular charge but no per-atom populations, while the shipped Vietoris-Rips
asset carries a Mulliken charge on every node.  Building conformer complexes
without them would set ``charge_missing`` on every augmented structure and
leave nothing on the originals -- a marker that tells the model which structures
are augmented.  That is a confound, not a missing feature, so the charges are
recomputed rather than imputed.

Cheap, and safe: a single point converges where the *optimisation* path did not
(the 86 octanol failures were SCF non-convergence during ANCopt, diagnosed in
Stage 1), and the geometry is not touched -- only the electronic structure at a
geometry that already exists.

Chemistry is carried over, never re-decided: the molecular charge comes from the
``.json`` sidecar ``reoptimize.py`` wrote next to each structure, which recorded
both the value and its provenance.  Structures whose sidecar has no charge are
skipped and reported, not guessed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.qc import xtb_backend as xb          # noqa: E402
from automl.qc.reoptimize import OUT_ROOT        # noqa: E402

OUT_DIR = _REPO / "automl/artifacts/conformer_charges"


def _pairs(solvent: str) -> list[tuple[Path, Path]]:
    """(xyz, sidecar) for every completed re-optimisation of this solvent."""
    root = OUT_ROOT / solvent
    out = []
    for xyz in sorted(root.rglob("*.xyz")):
        js = xyz.with_suffix(".json")
        if js.exists():
            out.append((xyz, js))
    return out


def run(solvent: str, shard: int, num_shards: int, threads: int) -> int:
    dest = OUT_DIR / solvent
    dest.mkdir(parents=True, exist_ok=True)
    todo = _pairs(solvent)
    mine = [t for i, t in enumerate(todo) if i % num_shards == shard]
    print(f"[charges] solvent={solvent} shard={shard}/{num_shards} "
          f"{len(mine)} of {len(todo)} structures", flush=True)

    ok = skipped = failed = 0
    for xyz, js in mine:
        out_path = dest / (xyz.stem + ".npz")
        if out_path.exists():                       # idempotent: safe to re-run
            ok += 1
            continue
        rec = json.loads(js.read_text())
        charge = rec.get("charge")
        if charge is None or not rec.get("ok"):
            skipped += 1
            continue
        symbols, coords = xb.read_plain_xyz(xyz)
        try:
            res = xb.single_point(symbols, coords, charge=int(charge),
                                  solvent=solvent, threads=threads)
        except Exception as exc:                    # a crashed structure must
            print(f"  FAIL {xyz.name}: {type(exc).__name__}: {exc}", flush=True)
            failed += 1                             # not kill the shard
            continue
        q = res.get("partial_charges")
        if q is None or not res.get("charge_sum_ok"):
            # Refuse a charge vector that does not sum to the molecular charge.
            # A wrong charge is wrong chemistry, silently.
            print(f"  REJECT {xyz.name}: "
                  f"{'no charges parsed' if q is None else f'sum={float(np.sum(q)):+.3f} != {charge}'}",
                  flush=True)
            failed += 1
            continue
        np.savez_compressed(
            out_path, partial_charges=q.astype(np.float32),
            symbols=np.asarray(symbols, dtype="U3"),
            coordinates=np.asarray(coords, dtype=np.float32),
            charge=np.int32(charge),
            geometry_key=np.asarray(str(rec.get("geometry_key", "")), dtype="U64"),
            energy_eh=np.float64(res.get("energy_eh") or np.nan))
        ok += 1

    print(f"[charges] DONE solvent={solvent} shard={shard} "
          f"ok={ok} skipped={skipped} failed={failed}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solvent", required=True, choices=("water", "octanol"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    return run(args.solvent, args.shard, args.num_shards, args.threads)


if __name__ == "__main__":
    raise SystemExit(main())
