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
3. **relax**    -- GFN-FF ANCopt each snapshot in ALPB water.
4. **score**    -- GFN2 single point on each relaxed structure; that is the
                   energy the Boltzmann weight uses.
5. **dedup**    -- drop duplicates by heavy-atom RMSD and by energy.
6. **weight**   -- Boltzmann weights at 298 K.

Steps 3 and 4 were originally one GFN2 relaxation per snapshot.  Measured on
this cluster that costs **15+ minutes for a single 300-atom structure** (job
5278267), i.e. ~340 CPU-hours for an 80-complex pilot against a two-node cap --
for a pilot whose whole purpose is to decide whether a larger campaign is worth
running.  Splitting it puts the expensive method where it decides the answer
(relative energies) and the cheap one where it does not (which local minimum a
snapshot falls into).

The metal is never substituted, no ligand is ever re-perceived, and nothing is
written to `data/`.  Output goes to `automl/artifacts/mtd_ensemble/`.

Honest limits, stated before any result
---------------------------------------
* GFN-FF sampling with GFN2 reranking is what CREST does in outline, not in
  detail; there is no RMSD-based structure-space clustering, no torsional-mode
  pre-screening and no z-matrix sorting, and CREST relaxes at the working level
  where this relaxes at GFN-FF.  Calling this "a CREST ensemble" would be an
  overclaim, and it is called CREST-lite everywhere.
* Relaxing at GFN-FF means the *geometries* are GFN-FF minima, not GFN2 minima.
  For the pilot's question -- can the conformer scatter in a GFN2 energy be cut
  ~4x -- that is acceptable, because the energies are GFN2.  It would not be
  acceptable if the geometries themselves were the deliverable.
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
                 timeout: int, overwrite: bool = False,
                 opt_method: str = "gfnff", opt_level: str = "loose",
                 maxcycle: int = 200) -> dict[str, Any]:
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

    # Relax cheaply, score properly.
    #
    # Measured, not assumed: a GFN2 ANCopt of one 300-atom snapshot runs 15+
    # minutes on this cluster (observed in job 5278267, four workers each at
    # 99.9% CPU for a quarter of an hour on a single structure).  At 16
    # snapshots x 80 complexes that is ~340 CPU-hours for the pilot alone,
    # against a GrpTRES cap of two nodes -- and the pilot exists to decide
    # whether a much larger campaign is worth running.
    #
    # So: relax at GFN-FF, then take a GFN2 single point on the relaxed
    # structure for the energy that enters the Boltzmann weight.  This is
    # closer to what CREST actually does than a full GFN2 relaxation of every
    # snapshot would be, and it puts the expensive method where it decides the
    # answer (relative energies) rather than where it does not (which local
    # minimum a snapshot falls into).
    relaxed: list[tuple[float, np.ndarray]] = []
    for sy, xyz in picks:
        r = xb.optimize(sy, xyz, charge=charge, solvent="water",
                        opt_level=opt_level, maxcycle=maxcycle, binary=binary,
                        threads=threads, timeout=timeout, method=opt_method)
        # xb.optimize returns the relaxed geometry under "coords" (not
        # "coordinates" -- read_extxyz uses that name, optimize does not).
        if not r.get("ok"):
            continue
        geo = (np.asarray(r["coords"], dtype=float)
               if r.get("coords") is not None else xyz)
        if opt_method == "gfnff":
            sp = xb.single_point(sy, geo, charge=charge, solvent="water",
                                 binary=binary, threads=threads,
                                 timeout=timeout)
            e = sp.get("energy_ev") if sp.get("ok") else None
        else:
            e = r.get("energy_ev")
        if e is not None:
            relaxed.append((float(e), geo))
    rec["relaxed"] = len(relaxed)
    rec["opt_method"] = opt_method
    rec["opt_level"] = opt_level
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
        rec["e_mean_ev"] = float(energies.mean())
        rec["e_spread_ev"] = float(energies.max() - energies.min())
        # Boltzmann weighting was the assumed remedy and the smoke falsified it:
        # conformer energy gaps here are 0.8-1.9 eV against kT = 0.026 eV, so
        # exp(-dE/kT) puts ~99% of the weight on one structure and n_effective
        # comes out at 1.0-1.8.  A Boltzmann average of this ensemble IS its
        # minimum, so it cannot reduce scatter by sqrt(n).
        #
        # The surviving hypothesis is different and better posed: the dataset's
        # scatter comes from every complex being an ARBITRARY local minimum from
        # Architector, so taking each complex's *global* minimum from a common
        # search should reduce the within-family scatter by making the members
        # comparable.  e_min_ev is what tests that; e_mean_ev is the unweighted
        # average, kept as the contrast.
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
    """WHOLE ligand families, so a within-family SNR can be computed at all.

    The first version of this cycled through families taking one complex from
    each, and produced 80 complexes spread over 80 families -- one member
    apiece.  The pilot's entire question is whether the *within-family* scatter
    falls, and a family of one has no within-family scatter, so that selection
    could not have answered it however many structures it ran.

    Whole families it is: take the largest series-bearing families until the
    budget is spent, so every selected complex has partners to be compared with.
    """
    parts = jobs["geometry_key"].astype(str).str.split("|", n=2, expand=True)
    jobs = jobs.assign(fam=parts[1].fillna("") + "|" + parts[2].fillna(""))
    size = jobs.groupby("fam")["geometry_key"].transform("size")
    rich = jobs[size >= 5].copy()
    order = (rich.groupby("fam").size().sort_values(ascending=False).index)
    take, total = [], 0
    for f in order:
        members = rich[rich["fam"] == f]
        if total + len(members) > n and take:
            continue
        take.append(members)
        total += len(members)
        if total >= n:
            break
    if not take:
        return rich.head(0)
    return pd.concat(take).reset_index(drop=True)


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
    ap.add_argument("--opt-method", default="gfnff",
                    choices=("gfnff", "gfn2"),
                    help="relaxation level for snapshots; energies "
                         "are always a GFN2 single point")
    ap.add_argument("--opt-level", default="loose")
    ap.add_argument("--maxcycle", type=int, default=200)
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
                          overwrite=args.overwrite,
                          opt_method=args.opt_method,
                          opt_level=args.opt_level,
                          maxcycle=args.maxcycle)
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
