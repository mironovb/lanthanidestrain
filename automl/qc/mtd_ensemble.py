#!/usr/bin/env python3
"""CREST-lite: xTB metadynamics conformer ensembles, and Boltzmann-weighted energies.

Why this now has a number attached
-----------------------------------
Multi-conformer sampling has been named as "the physics lever I could not
properly test" in `S2_RESULTS.md`, `WO_RESULTS.md` and `PI_EMAIL.md` sec 9, always
qualitatively.  `energy_diagnostic.py` turned it into a target.

Every reference-energy feature carries the lanthanide-series trend with **less
signal than noise inside a ligand family** -- the best case is
`e_int_octanol` at SNR 0.25, and 98-100% of families sit below 1.  The
denominator is conformer-to-conformer scatter: each complex in the dataset is
one stochastic Architector/GFN2 conformer, so a "binding energy" is really one
draw from a distribution ~0.75 eV wide, against a per-metal step of ~0.20 eV.

That is why the metal-substitution probe passed (0.306 eV between adjacent
lanthanides, 17x the relevant scale) and the features still destroyed the
selectivity metric: the probe held the geometry fixed and the dataset does not.

To make the trend dominate, the scatter has to fall by **at least 3.9x**, which
is ~16 effectively independent conformers per complex.  That is the pilot's
success criterion, and it is measurable *without training a single model*.

What this does
--------------
CREST is not installed and there is no outbound network on this cluster, so the
search is built from xtb's own metadynamics:

1. **sample**   -- GFN-FF metadynamics from the shipped geometry.  GFN-FF rather
                   than GFN2 because sampling needs 10^4 steps and GFN2 would
                   make the pilot cost more than the campaign it is testing.
2. **snapshot** -- take frames at a fixed stride.
3. **relax**    -- GFN2 ANCopt each snapshot in ALPB water, the same level the
                   dataset's own geometries use.
4. **dedup**    -- drop duplicates by heavy-atom RMSD and by energy.
5. **weight**   -- Boltzmann weights at 298 K.

The metal is never substituted, no ligand is ever re-perceived, and nothing is
written to `data/`.  Output goes to `automl/artifacts/mtd_ensemble/`.

Honest limits, stated before any result
---------------------------------------
* GFN-FF sampling with GFN2 re-optimisation is what CREST does in outline, not
  in detail; there is no RMSD-based structure-space clustering, no
  torsional-mode pre-screening and no z-matrix sorting.  Calling this "a CREST
  ensemble" would be an overclaim, and it is called CREST-lite everywhere.
* A metadynamics run that fails to escape its starting basin returns the
  starting structure back.  That is recorded per complex (`n_unique`), because
  an ensemble of one dressed up as an ensemble is exactly how this could produce
  a false null.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.geometry_features import read_extxyz          # noqa: E402
from automl.qc import xtb_backend as xb                # noqa: E402
from automl.qc.reoptimize import job_table             # noqa: E402

OUT_ROOT = _REPO_ROOT / "automl/artifacts/mtd_ensemble"
KT_298 = 0.025693          # eV


# ---------------------------------------------------------------------------
def _mtd_input(time_ps: float, dump_fs: float, temp_k: float,
               kpush: float, alp: float) -> str:
    """xtb metadynamics control block.

    ``save`` is the number of previous structures the bias is built from; kpush
    and alpha set its height and width.  These are xtb's own defaults for a
    conformer search, not tuned here -- tuning a sampler on the metric it feeds
    is how the persistence-image sweep manufactured a winner.
    """
    return (f"$md\n"
            f"  temp={temp_k}\n"
            f"  time={time_ps}\n"
            f"  dump={dump_fs}\n"
            f"  step=2.0\n"
            f"  shake=2\n"
            f"  hmass=4\n"
            f"$metadyn\n"
            f"  save=10\n"
            f"  kpush={kpush}\n"
            f"  alp={alp}\n"
            f"$end\n")


def _read_multi_xyz(path: Path) -> list[tuple[list[str], np.ndarray]]:
    """Frames from an xtb trajectory file."""
    out: list[tuple[list[str], np.ndarray]] = []
    if not path.exists():
        return out
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        try:
            n = int(lines[i].strip())
        except (ValueError, IndexError):
            break
        sy, xyz = [], []
        for row in lines[i + 2: i + 2 + n]:
            bits = row.split()
            if len(bits) < 4:
                break
            sy.append(bits[0])
            xyz.append([float(bits[1]), float(bits[2]), float(bits[3])])
        if len(sy) == n:
            out.append((sy, np.asarray(xyz, dtype=float)))
        i += n + 2
    return out


def _heavy_rmsd(a: np.ndarray, b: np.ndarray, symbols: list[str]) -> float:
    """Kabsch RMSD over heavy atoms.

    No atom reordering: these are frames of one trajectory and one optimisation
    of it, so the atom order is fixed by construction.
    """
    keep = [i for i, s in enumerate(symbols) if s != "H"]
    if not keep:
        return float("inf")
    x, y = a[keep], b[keep]
    x = x - x.mean(0)
    y = y - y.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((x @ r - y) ** 2, axis=1))))


# ---------------------------------------------------------------------------
def ensemble_one(row, *, time_ps: float, dump_fs: float, temp_k: float,
                 kpush: float, alp: float, max_snapshots: int,
                 rmsd_tol: float, e_tol_ev: float, threads: int,
                 timeout: int, overwrite: bool = False) -> dict[str, Any]:
    """Sample, relax, dedup and Boltzmann-weight one complex."""
    key = str(row["geometry_key"])
    safe = key.replace("/", "_").replace("|", "_")
    dest = OUT_ROOT / "per_geometry" / f"{safe}.json"
    if dest.exists() and not overwrite:
        return {"geometry_key": key, "status": "cached"}

    t0 = time.time()
    binary = xb.find_xtb()
    if binary is None:
        return {"geometry_key": key, "status": "no_xtb_binary"}
    try:
        geom = read_extxyz(Path(row["local"]))
    except Exception as exc:                       # noqa: BLE001
        return {"geometry_key": key, "status": f"unreadable:{exc}"}

    symbols = list(geom.symbols)
    coords = np.asarray(geom.coordinates, dtype=float)
    charge, prov = xb.infer_charge(geom)
    if charge is None:
        return {"geometry_key": key, "status": f"no_charge:{prov}"}

    rec: dict[str, Any] = {"geometry_key": key, "charge": charge,
                           "n_atoms": len(symbols), "status": "ok",
                           "metal": str(row.get("metal")),
                           "geometry_feature_build_id":
                               str(row.get("geometry_feature_build_id"))}

    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        xb.write_plain_xyz(symbols, coords, wd / "in.xyz")
        (wd / "md.inp").write_text(_mtd_input(time_ps, dump_fs, temp_k,
                                              kpush, alp))
        # GFN-FF for the sampling leg only.  10^4 MD steps at GFN2 on a
        # 300-atom complex would cost more than the campaign this pilot exists
        # to justify.
        args = ["in.xyz", "--gfnff", "--chrg", str(charge), "--metadyn",
                "--input", "md.inp", "--alpb", "water", "--norestart"]
        try:
            proc = subprocess.run([str(binary)] + args, cwd=wd,
                                  env=xb.xtb_env(binary, threads),
                                  capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            rec["status"] = "mtd_timeout"
            proc = None
        frames = []
        for name in ("xtb.trj", "xtbmdok.log", "xtb.trj.xyz"):
            frames = _read_multi_xyz(wd / name)
            if frames:
                break
        rec["mtd_frames"] = len(frames)
        if proc is not None and proc.returncode != 0 and not frames:
            rec["status"] = "mtd_failed"
            rec["log_tail"] = (proc.stdout + proc.stderr)[-2000:]

    if not frames:
        rec["n_unique"] = 0
        rec["seconds"] = time.time() - t0
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=2))
        os.replace(tmp, dest)
        return {"geometry_key": key, "status": rec["status"]}

    # Even stride, plus the shipped geometry itself so the ensemble can never be
    # worse-sampled than what it replaces.
    stride = max(1, len(frames) // max_snapshots)
    picks = [(symbols, coords)] + [frames[i] for i in
                                   range(0, len(frames), stride)][:max_snapshots]

    relaxed: list[tuple[float, np.ndarray]] = []
    for sy, xyz in picks:
        r = xb.optimize(sy, xyz, charge=charge, solvent="water",
                        opt_level="normal", maxcycle=300, binary=binary,
                        threads=threads, timeout=timeout)
        # xb.optimize returns the relaxed geometry under "coords" (not
        # "coordinates" -- read_extxyz uses that name, optimize does not).
        if r.get("ok") and r.get("energy_ev") is not None:
            relaxed.append((float(r["energy_ev"]),
                            np.asarray(r["coords"], dtype=float)
                            if r.get("coords") is not None else xyz))
    rec["relaxed"] = len(relaxed)
    if not relaxed:
        rec["status"] = "no_relaxed"
        rec["n_unique"] = 0
    else:
        relaxed.sort(key=lambda t: t[0])
        uniq: list[tuple[float, np.ndarray]] = []
        for e, c in relaxed:
            if any(abs(e - e2) < e_tol_ev
                   and _heavy_rmsd(c, c2, symbols) < rmsd_tol
                   for e2, c2 in uniq):
                continue
            uniq.append((e, c))
        energies = np.array([e for e, _ in uniq])
        rel = energies - energies.min()
        w = np.exp(-rel / KT_298)
        w = w / w.sum()
        rec["n_unique"] = len(uniq)
        rec["energies_ev"] = energies.tolist()
        rec["weights"] = w.tolist()
        rec["e_boltzmann_ev"] = float(np.sum(w * energies))
        rec["e_min_ev"] = float(energies.min())
        rec["e_spread_ev"] = float(energies.max() - energies.min())
        # Effective sample size: an ensemble whose weight sits on one structure
        # is one structure, and reporting it as N would be the false-null route
        # this module's docstring warns about.
        rec["n_effective"] = float(1.0 / np.sum(w ** 2))

    rec["seconds"] = time.time() - t0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    os.replace(tmp, dest)
    return {"geometry_key": key, "status": rec["status"],
            "n_unique": rec.get("n_unique", 0), "seconds": rec["seconds"]}


# ---------------------------------------------------------------------------
def pilot_selection(jobs: pd.DataFrame, n: int) -> pd.DataFrame:
    """Complexes spanning families that actually have a lanthanide series.

    A pilot drawn at random would mostly hit one-member families, where a
    per-family SNR is undefined and the pilot could not answer its own question.
    """
    parts = jobs["geometry_key"].astype(str).str.split("|", n=2, expand=True)
    jobs = jobs.assign(fam=parts[1].fillna("") + "|" + parts[2].fillna(""))
    size = jobs.groupby("fam")["geometry_key"].transform("size")
    rich = jobs[size >= 5]
    fams = list(pd.unique(rich["fam"]))
    take, i = [], 0
    while len(take) < n and fams:
        f = fams[i % len(fams)]
        rows = rich[rich["fam"] == f]
        used = sum(1 for t in take if t[1] == f)
        if used < len(rows):
            take.append((rows.iloc[used]["geometry_key"], f))
        elif len(take) and i > 4 * len(fams):
            break
        i += 1
    keys = [k for k, _ in take]
    return rich[rich["geometry_key"].isin(keys)].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", type=int, default=80,
                    help="number of complexes; 0 means every geometry")
    ap.add_argument("--time-ps", type=float, default=10.0)
    ap.add_argument("--dump-fs", type=float, default=200.0)
    ap.add_argument("--temp-k", type=float, default=400.0)
    ap.add_argument("--kpush", type=float, default=0.02)
    ap.add_argument("--alp", type=float, default=0.8)
    ap.add_argument("--max-snapshots", type=int, default=16,
                    help="target set by energy_diagnostic: the family-level "
                         "scatter must fall ~3.9x, i.e. ~16 independent "
                         "conformers")
    ap.add_argument("--rmsd-tol", type=float, default=0.25)
    ap.add_argument("--e-tol-ev", type=float, default=0.005)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()

    if args.collect:
        root = OUT_ROOT / "per_geometry"
        rows = [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]
        df = pd.DataFrame(rows)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        out = OUT_ROOT / "ensembles.parquet"
        df.to_parquet(out, index=False)
        print(f"[mtd] {len(df)} complexes -> {out}")
        if "status" in df:
            print(df["status"].value_counts().to_string())
        if "n_unique" in df:
            u = pd.to_numeric(df["n_unique"], errors="coerce").dropna()
            eff = pd.to_numeric(df.get("n_effective"), errors="coerce").dropna()
            print(f"\n  unique conformers  : median {u.median():.1f}, "
                  f"{(u <= 1).mean():.0%} of complexes returned only one")
            if len(eff):
                print(f"  effective ensemble : median {eff.median():.2f}")
                print(f"  target from energy_diagnostic: ~16 independent "
                      f"conformers to cut the family scatter 3.9x")
                print(f"  ==> {'MET' if eff.median() >= 16 else 'NOT MET'} "
                      f"on effective sample size")
        return 0

    if xb.find_xtb() is None:
        print("[mtd] no xtb binary (set XTB_BIN)", flush=True)
        return 2

    jobs = job_table()
    jobs = jobs[jobs["geometry_ok"].astype(bool)].reset_index(drop=True)
    if args.pilot:
        jobs = pilot_selection(jobs, args.pilot)
        print(f"[mtd] pilot: {len(jobs)} complexes from "
              f"{jobs['fam'].nunique()} series-bearing families", flush=True)
    mine = jobs.iloc[args.shard::args.n_shards].reset_index(drop=True)
    print(f"[mtd] shard {args.shard}/{args.n_shards}: {len(mine)} complexes, "
          f"{args.workers} workers, {args.time_ps} ps GFN-FF metadynamics",
          flush=True)

    done, t0 = 0, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(ensemble_one, r, time_ps=args.time_ps,
                          dump_fs=args.dump_fs, temp_k=args.temp_k,
                          kpush=args.kpush, alp=args.alp,
                          max_snapshots=args.max_snapshots,
                          rmsd_tol=args.rmsd_tol, e_tol_ev=args.e_tol_ev,
                          threads=args.threads, timeout=args.timeout,
                          overwrite=args.overwrite)
                for _, r in mine.iterrows()]
        for f in as_completed(futs):
            res = f.result()
            done += 1
            if done % 5 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)} last={res.get('status')} "
                      f"n_unique={res.get('n_unique')} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    have = (len(list((OUT_ROOT / 'per_geometry').glob('*.json')))
            if (OUT_ROOT / "per_geometry").exists() else 0)
    print(f"[mtd] shard complete: {done} attempted, {have} JSONs on disk "
          f"across all shards", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
