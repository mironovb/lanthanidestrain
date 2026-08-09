#!/usr/bin/env python3
"""Measure the GFN2-xTB optimiser's own reproducibility, from perturbed starts.

Why this exists
---------------
Three reports (``SYNTHESIS.md``, ``WO_PREREGISTRATION.md``, ``WO_RESULTS.md``)
dismiss the 0.013 A adjacent-lanthanide radius step as "below the ~0.04 A
optimisation-noise floor".  **That number was never measured.**  It traces back
to an asserted "~0.05 A conformer scatter" carrying no derivation, and

    xtb_backend.OPT_LEVELS["tight"] = 8.0e-4 Eh/bohr  ->  0.041 eV/A

is a **force** convergence target, not a **distance**.  The numeral matches to
two significant figures, which is the most likely origin of the claim.  The
achieved forces are an order of magnitude tighter than that target anyway
(median ``force_max_ev_ang`` = 0.0022 over the 1,232 shipped re-optimisations).

So the floor is measured here properly: displace a converged structure by a
seeded Gaussian, re-optimise under **byte-identical** settings to the shipped
control, and see how far apart the relaxed structures land.

What is measured, and why two numbers not one
---------------------------------------------
A perturbation can leave the basin.  If it does, the spread is no longer an
optimiser property, it is a landscape property.  So the report separates:

  * within-basin reproducibility -- the actual floor, and
  * the basin-escape RATE at each sigma,

and refuses to collapse them into a single number.  Replacing a one-number
error with a different one-number error would repeat the mistake.

Pre-registered overturning criterion (``C7_PREREGISTRATION.md`` section 4): at
sigma = 0.05 A the median |delta <M-D>| between same-basin replicates must be
<= 0.005 A and its P90 <= 0.013 A, with the escape rate reported alongside.

Note on reuse: ``automl/qc/reoptimize.py`` cannot do this.  It writes one output
path per structure (``geom_reopt/<solvent>/<stem>.xyz``), so replicates would
overwrite one another.  Hence a per-replicate directory here.  Everything else
-- charge inference, the xtb invocation, the independent ``--grad`` force check
-- is ``xtb_backend`` unchanged, so the comparison to the control is exact.

    python3 -m automl.qc.opt_reproducibility --pilot 2
    python3 -m automl.qc.opt_reproducibility --shard 0 --num-shards 2 --workers 48
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.qc import xtb_backend as xb                      # noqa: E402
from src.geometry_features import read_extxyz                # noqa: E402

CONTROL = _REPO / "automl/artifacts/geom_reopt/water"
OUT = _REPO / "automl/artifacts/opt_repro"

# Settings that MUST match the control exactly, or this measures the wrong thing.
SOLVENT = "water"
OPT_LEVEL = "tight"
MAXCYCLE = 750

SIGMAS = (0.02, 0.05, 0.10)
N_REPLICATES = 4
N_STRUCTURES = 30            # 3 per decile of the control's atom-count spread
DONOR_CUTOFF = 3.10          # A, the repo's own inner-shell rule
BASIN_RMSD = 0.30            # A, above this a replicate is a different basin


def _charge_from_sidecar(src: Path) -> int | None:
    """Molecular charge recorded by the run that produced this structure."""
    j = src.with_suffix(".json")
    if not j.exists():
        return None
    try:
        rec = json.loads(j.read_text())
    except Exception:                                        # noqa: BLE001
        return None
    c = rec.get("charge")
    return int(c) if c is not None else None


def _jsonable(v):
    """Records must survive json.dumps. xb.optimize returns several ndarray
    fields (coordinates, forces, ...) and a blanket key filter missed them."""
    import numpy as _np
    if isinstance(v, _np.ndarray):
        return v.tolist() if v.size <= 8 else f"<ndarray shape={v.shape}>"
    if isinstance(v, (_np.floating, _np.integer)):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def _dump(path, rec: dict) -> None:
    path.write_text(json.dumps({k: _jsonable(v) for k, v in rec.items()}, indent=1))


def _seed(stem: str, sigma: float, rep: int) -> int:
    """Deterministic per-(structure, sigma, replicate) seed.

    Reproducible from the record alone: anyone can regenerate the exact
    starting geometry that produced any row in the output.
    """
    return abs(hash((stem, round(sigma, 4), rep))) % (2 ** 31)


def perturb(coords: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Isotropic Gaussian displacement of every atom.

    The centre of mass is deliberately NOT re-fixed.  ANCopt is translation
    invariant, so a net translation costs nothing and is removed by the Kabsch
    comparison later; re-fixing it would silently remove part of the very
    perturbation being calibrated.
    """
    if sigma <= 0:
        return coords.copy()
    rng = np.random.default_rng(seed)
    return coords + rng.normal(0.0, sigma, size=coords.shape)


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD after optimal superposition. Same convention as the rest of the repo."""
    p = a - a.mean(0)
    q = b - b.mean(0)
    v, _s, wt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(v @ wt))
    r = v @ np.diag([1.0, 1.0, d]) @ wt
    return float(np.sqrt(((p @ r - q) ** 2).sum() / len(p)))


def mean_m_donor(symbols: list[str], coords: np.ndarray) -> tuple[float, int]:
    """Mean metal-donor distance over the inner shell, and the donor count.

    Donors are N/O/S/P/Cl/Br within DONOR_CUTOFF of the single lanthanide --
    the same rule the shipped feature code uses.  Returns NaN if the metal is
    not unique, rather than guessing.
    """
    z = [xb.SYMBOL_TO_Z.get(s, 0) if hasattr(xb, "SYMBOL_TO_Z") else 0
         for s in symbols]
    met = [i for i, s in enumerate(symbols)
           if s in ("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
                    "Dy", "Ho", "Er", "Tm", "Yb", "Lu")]
    if len(met) != 1:
        return float("nan"), 0
    m = met[0]
    d = np.linalg.norm(coords - coords[m], axis=1)
    don = [i for i, s in enumerate(symbols)
           if i != m and s in ("N", "O", "S", "P", "Cl", "Br")
           and d[i] <= DONOR_CUTOFF]
    if not don:
        return float("nan"), 0
    return float(d[don].mean()), len(don)


def pick_structures(n: int = N_STRUCTURES) -> list[Path]:
    """Stratified by atom count, so the answer is not a small-molecule answer.

    Three per decile of the control set's size distribution, deterministic.
    """
    files = sorted(CONTROL.glob("*.xyz"))
    sizes = []
    for p in files:
        try:
            sizes.append((int(p.read_text().split("\n", 1)[0].strip()), p))
        except Exception:                                    # noqa: BLE001
            continue
    sizes.sort()
    out, per = [], max(1, n // 10)
    for k in range(10):
        lo = k * len(sizes) // 10
        hi = (k + 1) * len(sizes) // 10
        band = sizes[lo:hi]
        if not band:
            continue
        step = max(1, len(band) // per)
        out += [p for _s, p in band[::step][:per]]
    return out[:n]


def run_one(src: Path, sigma: float, rep: int, *, threads: int = 1,
            timeout: int = 14400, overwrite: bool = False) -> dict:
    """One perturbed restart. Never raises into the pool."""
    stem = src.stem
    d = OUT / stem
    d.mkdir(parents=True, exist_ok=True)
    rec_path = d / f"s{sigma:g}_r{rep}.json"
    if rec_path.exists() and not overwrite:
        try:
            return json.loads(rec_path.read_text())
        except Exception:                                    # noqa: BLE001
            pass
    rec: dict = {"stem": stem, "sigma": sigma, "rep": rep,
                 "seed": _seed(stem, sigma, rep)}
    try:
        geom = read_extxyz(src)
        symbols = list(geom.symbols)
        coords = np.asarray(geom.coordinates, dtype=float)
        # The control structures were written by reoptimize.py WITHOUT per-atom
        # Mulliken populations, so infer_charge cannot read them back.  The
        # charge that produced them is recorded in the sidecar, with its own
        # provenance field -- use that, and fail loudly rather than guessing.
        charge = _charge_from_sidecar(src)
        if charge is None:
            rec.update(ok=False, reason="charge_unrecoverable:no_sidecar")
            _dump(rec_path, rec)
            return rec
        start = perturb(coords, sigma, rec["seed"])
        rec["start_rmsd"] = kabsch_rmsd(start, coords)
        res = xb.optimize(symbols, start, charge=charge, uhf=xb.DEFAULT_UHF,
                          solvent=SOLVENT, opt_level=OPT_LEVEL,
                          maxcycle=MAXCYCLE, threads=threads, timeout=timeout)
        rec.update({k: v for k, v in res.items() if k not in ("coords", "charges", "symbols")})
        if res.get("ok"):
            relaxed = np.asarray(res["coords"], dtype=float)
            rec["rmsd_to_reference"] = kabsch_rmsd(relaxed, coords)
            md0, n0 = mean_m_donor(symbols, coords)
            md1, n1 = mean_m_donor(symbols, relaxed)
            rec.update(mean_md_reference=md0, mean_md_relaxed=md1,
                       n_donor_reference=n0, n_donor_relaxed=n1,
                       d_mean_md=(md1 - md0) if np.isfinite(md0 + md1) else None,
                       same_basin=bool(rec["rmsd_to_reference"] <= BASIN_RMSD
                                       and n0 == n1))
            xyz = d / f"s{sigma:g}_r{rep}.xyz"
            tmp = xyz.with_suffix(".xyz.tmp")
            tmp.write_text(f"{len(symbols)}\nsigma={sigma} rep={rep} "
                           f"seed={rec['seed']}\n" +
                           "\n".join(f"{s} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}"
                                     for s, c in zip(symbols, relaxed)) + "\n")
            tmp.replace(xyz)
    except Exception as exc:                                 # noqa: BLE001
        rec.update(ok=False, reason=f"exception:{type(exc).__name__}:{exc}")
    _dump(rec_path, rec)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", type=int, default=0,
                    help="run this many structures only, all sigmas")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if xb.find_xtb() is None:
        raise SystemExit("no xtb binary; set XTB_BIN")
    OUT.mkdir(parents=True, exist_ok=True)

    structures = pick_structures()
    if args.pilot:
        structures = structures[:args.pilot]
    # sigma 0 is the idempotency control: same start, same settings.
    jobs = [(p, 0.0, 0)] if False else []
    for p in structures:
        jobs.append((p, 0.0, 0))
        for s in SIGMAS:
            for r in range(N_REPLICATES):
                jobs.append((p, s, r))
    jobs = [j for i, j in enumerate(jobs) if i % args.num_shards == args.shard]
    print(f"[repro] {len(structures)} structures, {len(jobs)} jobs on shard "
          f"{args.shard}/{args.num_shards}, {args.workers} workers", flush=True)

    t0 = time.time()
    ok = bad = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, p, s, r, threads=args.threads,
                            timeout=args.timeout, overwrite=args.overwrite)
                for p, s, r in jobs]
        for i, f in enumerate(futs, 1):
            rec = f.result()
            ok += bool(rec.get("ok"))
            bad += not rec.get("ok")
            if i % 25 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} ok={ok} failed={bad} "
                      f"{time.time() - t0:.0f}s", flush=True)
    print(f"[repro] done ok={ok} failed={bad} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
