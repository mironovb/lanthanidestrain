#!/usr/bin/env python3
"""Stage 1: re-optimise every complex properly, in solvent.

The shipped geometries all stopped on a loose ``fmax = 0.2 eV/A`` criterion
(residual forces hard-capped at 0.19999, 94 % between 0.15 and 0.20).  They are
therefore not relaxed structures but structures that ran out of criterion, and
part of the descriptor scatter previously blamed on conformational diversity is
optimisation noise.  This re-optimises with a real convergence threshold.

Two changes from the original run
---------------------------------
1. **Convergence.** ANCopt at ``tight``/``vtight`` instead of 0.2 eV/A.  The
   achieved residual force is re-measured with an independent ``--grad`` single
   point and recorded; nothing is called converged on the optimiser's say-so.
2. **Solvation.** ALPB in **water** and **n-octanol**.  log D is a partition
   coefficient between an aqueous and an organic phase, so a gas-phase geometry
   is the wrong reference state for it.  Running both phases also makes the
   *difference* between them a descriptor in its own right -- a direct
   partition proxy rather than just a tidier structure.

Chemistry is carried over, never re-decided: per-structure molecular charge is
recovered from the stored Mulliken populations (verified to reproduce the
original energies to ~6e-7 Eh across all three charge states present), spin is
closed-shell, and element composition comes from the input file.  Structures
whose charge cannot be recovered are skipped and reported, not guessed.

Outputs go to ``automl/artifacts/geom_reopt/<solvent>/`` -- never into ``data/``.
Resumable: a structure with a completed status JSON is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.geometry_features import read_extxyz  # noqa: E402
from automl.qc import xtb_backend as xb  # noqa: E402

OUT_ROOT = _REPO_ROOT / "automl/artifacts/geom_reopt"
SOLVENTS = {"water": "water", "octanol": "octanol", "gas": None}


def job_table() -> pd.DataFrame:
    """Every distinct geometry on disk, with its dataset provenance."""
    disk: dict[str, Path] = {}
    for p in _REPO_ROOT.rglob("*.xyz"):
        if "geom_reopt" in p.parts:
            continue
        disk.setdefault(p.name, p)
    df = pd.read_parquet(
        _REPO_ROOT / "data/processed/final_ml_dataset_3d.parquet",
        columns=["geometry_key", "xyz_path", "metal", "geometry_ok",
                 "geometry_qc_class", "geometry_feature_build_id"])
    df = df.dropna(subset=["xyz_path"]).drop_duplicates("geometry_key")
    df["basename"] = df["xyz_path"].map(lambda p: os.path.basename(str(p)))
    df["local"] = df["basename"].map(lambda b: str(disk[b]) if b in disk else None)
    return df.dropna(subset=["local"]).reset_index(drop=True)


def write_extxyz(path: Path, symbols, coords, *, energy_ev: float | None,
                 charge: int, solvent: str | None, forces=None) -> None:
    """Extended XYZ in the same dialect the descriptor pipeline already parses."""
    n = len(symbols)
    props = "species:S:1:pos:R:3"
    if forces is not None:
        props += ":forces:R:3"
    bits = [f'Properties={props}']
    if energy_ev is not None:
        bits.append(f"energy={energy_ev:.12f}")
        bits.append(f"free_energy={energy_ev:.12f}")
    bits.append(f"total_charge={charge}")
    bits.append(f'solvent="{solvent or "gas"}"')
    bits.append('pbc="F F F"')
    lines = [str(n), " ".join(bits)]
    for i, (s, xyz) in enumerate(zip(symbols, coords)):
        row = f"{s:2s} {xyz[0]:20.12f} {xyz[1]:20.12f} {xyz[2]:20.12f}"
        if forces is not None:
            row += f" {forces[i][0]:18.10f} {forces[i][1]:18.10f} {forces[i][2]:18.10f}"
        lines.append(row)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def reoptimize_one(row, solvent_key: str, opt_level: str, maxcycle: int,
                   timeout: int, threads: int, binary: Path,
                   overwrite: bool = False,
                   retry_failed: bool = False) -> dict[str, Any]:
    src = Path(row.local)
    out_dir = OUT_ROOT / solvent_key
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / f"{src.stem}.json"
    xyz_path = out_dir / f"{src.stem}.xyz"
    if status_path.exists() and not overwrite:
        try:
            rec = json.loads(status_path.read_text())
            if rec.get("done"):
                # A completed *failure* is retryable when asked: 86 octanol
                # structures exited rc=128 (a signal kill -- wall time or
                # memory on the largest systems), and every one of them
                # succeeded in water with the same charge and geometry.  That
                # is a resource limit, not a chemistry problem, so a retry with
                # a longer timeout is legitimate.  Successful records are never
                # recomputed.
                if retry_failed and not rec.get("ok"):
                    pass
                else:
                    rec["skipped_existing"] = True
                    return rec
        except json.JSONDecodeError:
            pass

    rec: dict[str, Any] = {"basename": src.name, "geometry_key": row.geometry_key,
                           "metal": row.metal, "solvent": solvent_key,
                           "opt_level": opt_level, "done": False}
    try:
        g = read_extxyz(src)
        charge, provenance = xb.infer_charge(g)
        rec["charge_provenance"] = provenance
        if charge is None:
            # Never guess a charge: a wrong charge is wrong chemistry, silently.
            rec.update({"done": True, "ok": False, "reason": "charge_unrecoverable"})
            status_path.write_text(json.dumps(rec, indent=2))
            return rec
        rec["charge"] = charge
        rec["n_atoms"] = int(len(g.symbols))
        rec["input_force_max_ev_ang"] = _input_force_max(src, len(g.symbols))

        res = xb.optimize(g.symbols, g.coordinates, charge=charge,
                          uhf=xb.DEFAULT_UHF, solvent=SOLVENTS[solvent_key],
                          opt_level=opt_level, maxcycle=maxcycle,
                          binary=binary, threads=threads, timeout=timeout)
        if not res.get("ok"):
            rec.update({"done": True, "ok": False,
                        "reason": res.get("reason", "unknown"),
                        "seconds": res.get("seconds")})
            status_path.write_text(json.dumps(rec, indent=2))
            return rec

        write_extxyz(xyz_path, res["symbols"], res["coords"],
                     energy_ev=res.get("energy_ev"), charge=charge,
                     solvent=SOLVENTS[solvent_key])
        rec.update({
            "done": True, "ok": True, "xyz": str(xyz_path),
            "energy_ev": res.get("energy_ev"),
            "xtb_converged": res.get("xtb_converged"),
            "cycles": res.get("cycles"),
            "force_max_ev_ang": res.get("force_max_ev_ang"),
            "force_rms_ev_ang": res.get("force_rms_ev_ang"),
            "target_force_ev_ang": res.get("target_force_ev_ang"),
            "meets_target": res.get("meets_target"),
            "rmsd_from_input_ang": res.get("rmsd_from_input_ang"),
            "seconds": res.get("seconds"),
        })
    except Exception as exc:  # keep the sweep alive, record why
        rec.update({"done": True, "ok": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=3)})
    status_path.write_text(json.dumps(rec, indent=2))
    return rec


def _input_force_max(src: Path, n_atoms: int) -> float | None:
    """Max residual force of the *input* structure, for a before/after record.

    Reuses the extxyz force parser from the descriptor pipeline so the numbers
    are directly comparable with the 0.2 eV/A ceiling documented there.
    """
    try:
        from automl.geom3d_features import _read_forces
        f = _read_forces(src, n_atoms)
        if f is None:
            return None
        return float(np.linalg.norm(f, axis=1).max())
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solvent", choices=sorted(SOLVENTS), default="water")
    ap.add_argument("--opt-level", default="tight",
                    choices=sorted(xb.OPT_LEVELS))
    ap.add_argument("--maxcycle", type=int, default=750)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--retry-failed", action="store_true",
                    help="recompute structures whose stored record is a "
                         "failure; successful records are never touched")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent xtb processes within this shard")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ok-only", action="store_true",
                    help="restrict to qc_class == OK geometries")
    args = ap.parse_args()

    binary = xb.find_xtb()
    if binary is None:
        print("FATAL: no xtb binary (set XTB_BIN)"); return 1

    jobs = job_table()
    if args.ok_only:
        jobs = jobs[jobs.geometry_ok.astype(bool)].reset_index(drop=True)
    if args.limit:
        jobs = jobs.head(args.limit)
    mine = jobs.iloc[args.shard::args.num_shards].reset_index(drop=True)
    target = xb.OPT_LEVELS[args.opt_level][1]
    print(f"[reopt] solvent={args.solvent} level={args.opt_level} "
          f"(target {target:.3f} eV/A vs shipped 0.200) "
          f"shard {args.shard}/{args.num_shards} -> {len(mine)} structures",
          flush=True)

    t0 = time.time()
    n_ok = n_skip = n_fail = 0

    def _run(row):
        return reoptimize_one(row, args.solvent, args.opt_level, args.maxcycle,
                              args.timeout, args.threads, binary, args.overwrite,
                              args.retry_failed)

    rows = list(mine.itertuples(index=False))
    if args.workers > 1:
        # xtb runs single-threaded here (one OMP thread per process), so the
        # throughput win comes from running many structures at once rather than
        # from parallelising any one of them.  Structure wall times vary ~20x
        # with atom count, so a pool keeps every core busy where a sequential
        # shard would stall behind one 300-atom outlier.
        # Threads, not processes: the work is an external xtb subprocess, so
        # the GIL is released while it runs, and threads avoid pickling the
        # per-row job objects across a process boundary.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_run, r): r for r in rows}
            for fut in as_completed(futs):
                results.append(fut.result())
        rows_iter = list(enumerate(results))
    else:
        rows_iter = None

    for i, row in (rows_iter if rows_iter is not None
                   else enumerate(rows)):
        rec = row if rows_iter is not None else _run(row)
        if rec.get("skipped_existing"):
            n_skip += 1
        elif rec.get("ok"):
            n_ok += 1
            print(f"  [{i+1}/{len(mine)}] {rec['basename'][:46]:48s} "
                  f"q={rec.get('charge'):+d} n={rec.get('n_atoms'):3d} "
                  f"fmax={rec.get('force_max_ev_ang', float('nan')):.4f} "
                  f"conv={rec.get('xtb_converged')} "
                  f"rmsd={rec.get('rmsd_from_input_ang', float('nan')):.3f} "
                  f"[{rec.get('seconds', 0):.0f}s]", flush=True)
        else:
            n_fail += 1
            print(f"  [{i+1}/{len(mine)}] {rec['basename'][:46]:48s} "
                  f"FAILED {rec.get('reason')}", flush=True)
    print(f"[reopt] done: ok={n_ok} skipped={n_skip} failed={n_fail} "
          f"in {time.time()-t0:.0f}s", flush=True)
    print(f"[reopt] len(mine)={len(mine)} workers={args.workers}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
