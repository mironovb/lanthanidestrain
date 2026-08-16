#!/usr/bin/env python3
"""g-xTB frozen-cage metal-substitution probe over the modelled complexes.

The D2 pilot (energy_series.py) found ligand-specific g-xTB energy steps
rank-correlate with ligand-specific measured selectivity (spearman -0.124,
p = 0.010, n = 430 cells / 59 ligands) where GFN2 is exactly null -- but the
pilot used RELAXED series energies (optimisation noise) on 71 ligands.  This
probe is the clean instrument at full scale: every lanthanide substituted
into every modelled cage at FIXED geometry (the ligand cage bit-identical
across the series, as gxtb_probe.series does), high-spin g-xTB, gas phase.

12,795 single points (853 cages x 15 metals).  Resumable: one JSON per cage
under automl/artifacts/gxtb_cage/records/; a killed run loses one cage.
Cages are processed round-robin across ligands so partial coverage is
ligand-diverse from the first hour.

Usage (inside slurm; see automl/slurm/gxtb_cage.sh):
  PYTHONPATH=$PWD python3 -m automl.qc.gxtb_cage_probe --workers 48
  PYTHONPATH=$PWD python3 -m automl.qc.gxtb_cage_probe --limit 2 --workers 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "automl/artifacts/gxtb_cage/records"


def cage_list() -> list[dict]:
    """The refxtb job list, round-robin across ligands."""
    from automl.qc.reoptimize import job_table
    jobs = job_table()
    jobs = jobs[jobs["geometry_ok"].astype(bool)].reset_index(drop=True)
    jobs["lig"] = jobs["geometry_key"].astype(str).str.split("|").str[1]
    buckets: dict[str, list[dict]] = {}
    for _, r in jobs.iterrows():
        buckets.setdefault(r["lig"], []).append(dict(r))
    order = []
    while any(buckets.values()):
        for lig in list(buckets):
            if buckets[lig]:
                order.append(buckets[lig].pop(0))
            else:
                del buckets[lig]
    return order


def probe_cage(row: dict, threads: int) -> dict:
    from automl.qc import xtb_backend as xb
    from automl.qc.gxtb_probe import series
    from src.geometry_features import read_extxyz

    bid = str(row["geometry_feature_build_id"])
    dest = OUT / f"{bid}.json"
    if dest.exists():
        return {"bid": bid, "skipped": True}
    path = row["local"]
    geom = read_extxyz(Path(path))
    charge, prov = xb.infer_charge(geom)
    if charge is None:
        rec = {"bid": bid, "ok": False, "reason": f"charge:{prov}"}
    else:
        try:
            recs = series(Path(path), method="gxtb", charge=int(charge),
                          solvent=None, threads=threads)
            slim = [{k: r.get(k) for k in
                     ("metal", "z", "f_count", "unpaired", "ok", "reason",
                      "energy_eh", "homo_lumo_gap_ev", "metal_mulliken",
                      "seconds")} for r in recs]
            rec = {"bid": bid, "path": str(path), "charge": int(charge),
                   "geometry_key": str(row.get("geometry_key")),
                   "ok": True, "records": slim}
        except Exception as e:                          # noqa: BLE001
            rec = {"bid": bid, "ok": False, "reason": repr(e)}
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(dest)
    return {"bid": bid, "ok": rec.get("ok")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cages = cage_list()
    done = {p.stem for p in OUT.glob("*.json")}
    todo = [c for c in cages
            if str(c["geometry_feature_build_id"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[cage] {len(cages)} cages, {len(done)} done, {len(todo)} to run",
          flush=True)

    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(probe_cage, c, args.threads): c for c in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            n_ok += bool(r.get("ok"))
            n_fail += not r.get("ok", True)
            if i % 10 == 0 or i == len(futs):
                print(f"  [{i}/{len(futs)}] ok={n_ok} fail={n_fail}",
                      flush=True)
    print(f"[cage] DONE ok={n_ok} fail={n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
