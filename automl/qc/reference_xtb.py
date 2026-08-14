#!/usr/bin/env python3
"""Reference xTB energetics: the queued-but-never-computed feature block.

`data/processed/feature_blocks/xtb_reference_calculation_queue.csv` holds 957
rows, every one marked ``not_run_requires_reference_xtb``.  `binding_energy_eV`,
`strain_energy_eV`, `homo_eV`, `lumo_eV` and `homo_lumo_gap_eV` are null for all
5,992 rows of the dataset, and `FINDINGS.md` calls them "the most promising
untested feature available".

Why they should matter: the separation factor between two lanthanides *is* a
difference of complexation free energies.  Every feature the models currently
see is geometric or topological -- there is not one energetic descriptor in the
whole design matrix.

What is computed, and why not the queue's own definition
--------------------------------------------------------
The queue defines binding as
``E_complex - (E_Ln_ion + n_ligs*E_free_ligand + n_fill*E_free_fill)``, which
needs the complex fragmented into chemically correct free ligands -- a
connectivity problem with its own failure modes, on structures that already have
a documented coordination-QC history.

This module computes the same physics without fragmenting anything:

    E_int = E(complex) - E(cage) - E(bare ion)

where **cage** is the complex with the metal atom deleted, at frozen geometry,
at charge ``q - 3``.  Deleting one atom needs no bond perception.  For a fixed
ligand the cage term is nearly common-mode between two adjacent lanthanides, so
the *difference* in E_int isolates precisely the metal-dependent part -- which
is the quantity the adjacent-pair metric scores.  Strain, which does need a
relaxation, is left to a second phase.

Also computed, and arguably the more directly relevant quantity:

    dG_transfer = E(complex, ALPB octanol) - E(complex, ALPB water)

That is a solvent-transfer energy, and log D is a partition coefficient.  No
feature in the current design matrix expresses it.

The probe that must run first
-----------------------------
GFN2-xTB's treatment of the lanthanide series is not obviously
element-resolving.  If the method returns essentially the same energy for Nd and
Pm in the same cage, then no energetic descriptor derived from it can carry
adjacent-pair information, and the entire campaign is a null that costs twenty
minutes to establish instead of eighty CPU-hours.

``--probe`` substitutes all 14 lanthanides into the same frozen cage and reports
how much the energy actually moves.  Run it, and read it, before ``--run``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

OUT_ROOT = _REPO_ROOT / "automl/artifacts/xtb_reference"
SOLVENTS = ("water", "octanol")

# The 14 lanthanides in the dataset, in series order.  Pm is absent from the
# measurements but is included in the probe: the probe is about the *method*,
# and a gap there would hide a discontinuity.
LANTHANIDES = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
               "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]


# ---------------------------------------------------------------------------
def _grep_orbitals(text: str) -> dict[str, float | None]:
    """HOMO, LUMO and gap in eV from an xtb log.

    xtb prints them on lines of the form
    ``(HOMO)``/``(LUMO)`` inside the orbital table and again as
    ``HOMO-LUMO GAP`` in the summary.  Both are read; the summary is preferred
    because it survives formatting changes in the table.
    """
    out: dict[str, float | None] = {"homo_ev": None, "lumo_ev": None,
                                    "gap_ev": None}
    for line in text.splitlines():
        s = line.strip()
        if "(HOMO)" in s or "(LUMO)" in s:
            bits = s.split()
            # ... <occ> <Eh> <eV> (HOMO)
            try:
                ev = float(bits[-2])
            except (ValueError, IndexError):
                continue
            out["homo_ev" if "(HOMO)" in s else "lumo_ev"] = ev
        elif "HOMO-LUMO GAP" in s:
            for tok in s.split():
                try:
                    out["gap_ev"] = float(tok)
                    break
                except ValueError:
                    continue
    if (out["gap_ev"] is None and out["homo_ev"] is not None
            and out["lumo_ev"] is not None):
        out["gap_ev"] = out["lumo_ev"] - out["homo_ev"]
    return out


def _metal_position(symbols: list[str]) -> int | None:
    hits = [i for i, s in enumerate(symbols) if s in LANTHANIDES]
    return hits[0] if len(hits) == 1 else None


def _sp(symbols, coords, charge, solvent, binary, threads, timeout,
        keep_log=True) -> dict[str, Any]:
    r = xb.single_point(symbols, coords, charge=charge, solvent=solvent,
                        binary=binary, threads=threads, timeout=timeout,
                        keep_log=keep_log)
    if keep_log and r.get("ok"):
        r.update(_grep_orbitals(r.pop("log", "") or ""))
    r.pop("log", None)
    return r


# ---------------------------------------------------------------------------
def probe_one(path: str, threads: int, timeout: int) -> dict[str, Any]:
    """Substitute every lanthanide into one frozen cage and record the energy.

    Geometry, charge and every other atom are held fixed, so the only thing that
    varies is the identity of the metal.  Any spread in the resulting energies is
    the method's element resolution; a flat line means GFN2 cannot tell these
    metals apart and no descriptor built on it will either.
    """
    binary = xb.find_xtb()
    geom = read_extxyz(Path(path))
    symbols = list(geom.symbols)
    coords = np.asarray(geom.coordinates, dtype=float)
    charge, prov = xb.infer_charge(geom)
    mi = _metal_position(symbols)
    if charge is None or mi is None:
        return {"path": path, "ok": False,
                "reason": f"charge={prov} metal_index={mi}"}
    out = {"path": path, "n_atoms": len(symbols), "charge": charge,
           "original_metal": symbols[mi], "ok": True}
    for ln in LANTHANIDES:
        sy = list(symbols)
        sy[mi] = ln
        r = _sp(sy, coords, charge, "water", binary, threads, timeout,
                keep_log=False)
        out[f"e_{ln}"] = r.get("energy_ev") if r.get("ok") else None
        # The raw total energies of two different elements differ mostly for a
        # trivial reason -- a different atom has a different reference energy --
        # so a spread in ``e_{ln}`` alone does NOT show that GFN2 resolves the
        # *chemistry* of the series.  Subtracting the bare ion removes that
        # offset, and the remainder is the interaction energy, which is the
        # quantity a descriptor would actually carry.  Both are recorded so the
        # write-up cannot quote the flattering one by accident.
        ion = _sp([ln], np.zeros((1, 3)), 3, "water", binary, threads, timeout,
                  keep_log=False)
        out[f"eion_{ln}"] = ion.get("energy_ev") if ion.get("ok") else None
        if out[f"e_{ln}"] is not None and out[f"eion_{ln}"] is not None:
            out[f"eint_{ln}"] = out[f"e_{ln}"] - out[f"eion_{ln}"]
        else:
            out[f"eint_{ln}"] = None
    return out


# ---------------------------------------------------------------------------
def reference_one(row, threads: int, timeout: int,
                  overwrite: bool = False) -> dict[str, Any]:
    """Every reference quantity for one geometry.

    Written to its own JSON immediately, temp-file-plus-atomic-replace, so a
    killed array task loses at most the structure it was working on and a rerun
    resumes rather than restarts.
    """
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
    mi = _metal_position(symbols)
    if charge is None:
        return {"geometry_key": key, "status": f"no_charge:{prov}"}
    if mi is None:
        return {"geometry_key": key, "status": "metal_not_unique"}

    metal = symbols[mi]
    cage_sy = [s for i, s in enumerate(symbols) if i != mi]
    cage_xyz = np.delete(coords, mi, axis=0)
    # The metal enters as Ln(III); removing it takes three positive charges with
    # it.  Getting this wrong would shift every cage energy by an ionisation
    # energy and quietly ruin the interaction term.
    cage_charge = charge - 3

    rec: dict[str, Any] = {
        "geometry_key": key, "metal": metal, "n_atoms": len(symbols),
        "charge": charge, "charge_provenance": prov, "cage_charge": cage_charge,
        "geometry_feature_build_id": str(row.get("geometry_feature_build_id")),
        "xyz": str(row["local"]), "status": "ok",
    }
    for sol in SOLVENTS:
        cx = _sp(symbols, coords, charge, sol, binary, threads, timeout)
        cg = _sp(cage_sy, cage_xyz, cage_charge, sol, binary, threads, timeout,
                 keep_log=False)
        ion = _sp([metal], np.zeros((1, 3)), 3, sol, binary, threads, timeout,
                  keep_log=False)
        rec[f"e_complex_{sol}_ev"] = cx.get("energy_ev")
        rec[f"e_cage_{sol}_ev"] = cg.get("energy_ev")
        rec[f"e_ion_{sol}_ev"] = ion.get("energy_ev")
        for k in ("homo_ev", "lumo_ev", "gap_ev"):
            rec[f"{k[:-3]}_{sol}_ev"] = cx.get(k)
        if all(rec.get(f"e_{p}_{sol}_ev") is not None
               for p in ("complex", "cage", "ion")):
            rec[f"e_int_{sol}_ev"] = (rec[f"e_complex_{sol}_ev"]
                                      - rec[f"e_cage_{sol}_ev"]
                                      - rec[f"e_ion_{sol}_ev"])
        else:
            rec[f"e_int_{sol}_ev"] = None
            rec["status"] = f"partial_{sol}"
        q = cx.get("partial_charges")
        if q is not None:
            rec[f"q_metal_{sol}"] = float(q[mi])
            rec[f"q_transfer_{sol}"] = float(3.0 - q[mi])

    if (rec.get("e_complex_octanol_ev") is not None
            and rec.get("e_complex_water_ev") is not None):
        # The partition-relevant quantity: log D is a partition coefficient and
        # nothing in the current design matrix expresses one.
        rec["dg_transfer_ev"] = (rec["e_complex_octanol_ev"]
                                 - rec["e_complex_water_ev"])
    if (rec.get("e_int_octanol_ev") is not None
            and rec.get("e_int_water_ev") is not None):
        rec["d_e_int_ev"] = rec["e_int_octanol_ev"] - rec["e_int_water_ev"]
    rec["seconds"] = time.time() - t0

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    os.replace(tmp, dest)
    return {"geometry_key": key, "status": rec["status"],
            "seconds": rec["seconds"]}


# ---------------------------------------------------------------------------
def collect() -> pd.DataFrame:
    """Every finished per-geometry JSON as one table."""
    root = OUT_ROOT / "per_geometry"
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true",
                    help="metal-substitution sensitivity check; run this FIRST")
    ap.add_argument("--probe-n", type=int, default=12,
                    help="how many cages to probe")
    ap.add_argument("--run", action="store_true", help="the reference campaign")
    ap.add_argument("--collect", action="store_true",
                    help="merge per-geometry JSONs into one parquet")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--threads", type=int, default=1,
                    help="xtb threads per structure; parallelism is across "
                         "structures, as in reopt.sh")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if xb.find_xtb() is None:
        print("[refxtb] no xtb binary found (set XTB_BIN)", flush=True)
        return 2

    jobs = job_table()
    jobs = jobs[jobs["geometry_ok"].astype(bool)].reset_index(drop=True)
    print(f"[refxtb] {len(jobs)} geometries with geometry_ok", flush=True)

    if args.probe:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        # Spread the probe across metals so the answer is not a property of one
        # ligand family.
        # One cage per metal when probe_n is small (the original sensitivity
        # check); a proportional per-metal cap when probe_n is large, so
        # --probe-n >= len(jobs) probes every cage instead of silently
        # stopping at 14 (one per metal).
        per = max(1, -(-args.probe_n // max(1, jobs["metal"].nunique())))
        pick = (jobs.groupby("metal", group_keys=False)
                .head(per).head(args.probe_n))
        print(f"[refxtb] probing {len(pick)} cages x {len(LANTHANIDES)} metals",
              flush=True)
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(probe_one, r["local"], args.threads,
                              args.timeout): r["geometry_key"]
                    for _, r in pick.iterrows()}
            for f in as_completed(futs):
                rows.append(f.result())
                print(f"  probed {len(rows)}/{len(futs)}", flush=True)
        pr = pd.DataFrame(rows)
        out = OUT_ROOT / "metal_probe.csv"
        pr.to_csv(out, index=False)

        ok = pr[pr["ok"]] if "ok" in pr.columns else pr
        print("\n=== metal-substitution sensitivity (ALPB water, frozen cage) ===")
        if not len(ok):
            print("  no usable probes -- see metal_probe.csv")
            return 1
        kt = 0.025693
        sf2 = kt * np.log(2)      # differential binding for a separation factor of 2
        print(f"  cages probed : {len(ok)}")
        print(f"  kT(298 K) = {kt:.4f} eV;  a separation factor of 2 is "
              f"{sf2:.4f} eV of differential binding\n")
        verdicts = {}
        for prefix, label in (("e_", "raw total energy"),
                              ("eint_", "interaction energy (bare ion removed)")):
            cols = [f"{prefix}{ln}" for ln in LANTHANIDES
                    if f"{prefix}{ln}" in ok.columns]
            if not cols:
                continue
            E = ok[cols].to_numpy(dtype=float)
            span = np.nanmax(E, axis=1) - np.nanmin(E, axis=1)
            adj = np.abs(np.diff(E, axis=1))
            med_adj = float(np.nanmedian(adj))
            verdicts[prefix] = med_adj
            print(f"  --- {label} ---")
            print(f"    La->Lu span        median : {np.nanmedian(span):.4f} eV")
            print(f"    ADJACENT-pair |dE| median : {med_adj:.4f} eV "
                  f"= {med_adj / sf2:.1f}x the SF=2 scale")
            print(f"    ADJACENT-pair |dE| min/max: "
                  f"{np.nanmin(adj):.4f} / {np.nanmax(adj):.4f} eV")
        # The honest reading uses the interaction energy.  Two different elements
        # have different total energies for a trivial reason -- a different
        # atomic reference -- so a spread in the raw column is not evidence that
        # GFN2 resolves the chemistry of the series.
        key = "eint_" if "eint_" in verdicts else "e_"
        med = verdicts.get(key, 0.0)
        print(f"\n  ==> on the {'interaction' if key == 'eint_' else 'RAW (see caveat)'} "
              f"energy, GFN2 separates adjacent lanthanides by "
              f"{med:.4f} eV = {med / sf2:.1f}x the SF=2 scale.")
        print("  ==> " + ("energetic descriptors CAN carry adjacent-pair "
                          "information; the campaign is justified."
                          if med > sf2 else
                          "GFN2 does NOT resolve adjacent lanthanides above the "
                          "scale that matters; no energetic descriptor built on "
                          "it can carry this signal, and the campaign is a null."))
        print(f"\n[refxtb] wrote {out}")
        return 0

    if args.collect:
        df = collect()
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        out = OUT_ROOT / "reference_energies.parquet"
        df.to_parquet(out, index=False)
        print(f"[refxtb] {len(df)} geometries -> {out}")
        if "status" in df.columns:
            print(df["status"].value_counts().to_string())
        return 0

    if not args.run:
        ap.error("choose --probe, --run or --collect")

    mine = jobs.iloc[args.shard::args.n_shards].reset_index(drop=True)
    if args.limit:
        mine = mine.head(args.limit)
    print(f"[refxtb] shard {args.shard}/{args.n_shards}: {len(mine)} geometries, "
          f"{args.workers} workers", flush=True)
    done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(reference_one, r, args.threads, args.timeout,
                          args.overwrite) for _, r in mine.iterrows()]
        for f in as_completed(futs):
            res = f.result()
            done += 1
            if done % 10 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)}  last={res.get('status')} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    # Never report a clean exit for a partial shard without saying so.
    have = len(list((OUT_ROOT / "per_geometry").glob("*.json"))) \
        if (OUT_ROOT / "per_geometry").exists() else 0
    print(f"[refxtb] shard complete: {done} attempted, {have} JSONs on disk "
          f"across all shards", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
