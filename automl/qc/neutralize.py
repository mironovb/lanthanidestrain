#!/usr/bin/env python3
"""Build the neutral extracted species, and a matched control, from the shipped complexes.

Pre-registered in ``automl/reports/CAMPAIGN4_PREREGISTRATION.md``.

Why
---
953 of 956 modelled complexes are still cationic.  The species that partitions
into kerosene is neutral, and charge neutralisation is the physics of solvent
extraction.  Three campaigns have changed *how* the complex is represented and
found nothing; this changes *what molecule is represented*.

Two arms per structure, both through the identical ladder:

* ``neutral``  -- n_add NO3- added, total charge 0
* ``control``  -- nothing added, charge unchanged

The control is the primary comparator, not a diagnostic.  This repo already
measured that re-optimising the same molecule at a new level moves it by median
1.87 A -- "different conformers, not refinements" -- so without it, "the neutral
structure differs from the shipped one" carries no information whatever.

Nothing is written to ``data/``.

Usage
-----
    python3 -m automl.qc.neutralize --pilot 40 --workers 48
    python3 -m automl.qc.neutralize --shard 0 --num-shards 8 --workers 48
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from automl.qc import xtb_backend as xb                      # noqa: E402
from automl.qc.nitrate_placement import place_one            # noqa: E402
from automl.qc.reoptimize import job_table, write_extxyz     # noqa: E402
from src.geometry_features import read_extxyz                # noqa: E402

OUT_ROOT = _REPO_ROOT / "automl/artifacts/neutral_species"
SOLVENT = "water"
DONOR_CUTOFF_A = 3.10          # this repo's own cutoff, automl/geom3d_features.py:113
NITRATE_NO_A = 1.75            # scripts/build_unique_geometries.py:444 convention
LANTHANIDES = set(range(57, 72))
_SYM_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17,
          "Br": 35, "I": 53}
DONOR_SYMBOLS = {"O", "N", "S", "P", "F", "Cl", "Br", "I"}


def _metal_index(symbols) -> int:
    """Index of the single lanthanide. Raises if there is not exactly one.

    ``_metal_index`` in geom3d_features silently takes the first; a second metal
    would make every metal-referenced quantity below meaningless.
    """
    from src.geometry_features import LANTHANIDE_SYMBOLS
    idx = [i for i, s in enumerate(symbols) if str(s) in LANTHANIDE_SYMBOLS]
    if len(idx) != 1:
        raise ValueError(f"expected exactly one lanthanide, found {len(idx)}")
    return idx[0]


def inner_sphere_nitrate_count(symbols, coords, mi: int) -> int:
    """Nitrates already bound: N with >=3 O within 1.75 A, some O within 3.4 A of the metal."""
    sym = np.asarray([str(s) for s in symbols])
    n = 0
    for i in np.flatnonzero(sym == "N"):
        d = np.linalg.norm(coords - coords[i], axis=1)
        os_ = np.flatnonzero((sym == "O") & (d < NITRATE_NO_A))
        if len(os_) >= 3:
            if np.linalg.norm(coords[os_] - coords[mi], axis=1).min() < 3.40:
                n += 1
    return int(n)


def ligand_cn(symbols, coords, mi: int, exclude: set[int] | None = None) -> int:
    """Donors within 3.10 A of the metal, optionally excluding added atoms."""
    sym = np.asarray([str(s) for s in symbols])
    d = np.linalg.norm(coords - coords[mi], axis=1)
    ok = (d < DONOR_CUTOFF_A) & (d > 1e-6) & np.isin(sym, list(DONOR_SYMBOLS))
    if exclude:
        keep = np.ones(len(sym), dtype=bool)
        keep[list(exclude)] = False
        ok &= keep
    return int(ok.sum())


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD after optimal superposition. Same atom order assumed."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    v, _, w = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(v @ w))
    r = v @ np.diag([1.0, 1.0, d]) @ w
    return float(np.sqrt((((a @ r) - b) ** 2).sum() / len(a)))


def _ladder(symbols, coords, charge: int, *, threads: int, timeout: int,
            binary, seed_coords=None) -> dict:
    """GFN-FF pre-relax, then GFN2/ALPB normal, then tight. Records every rung."""
    rec: dict = {"rungs": []}
    cur = np.asarray(coords, dtype=float)
    n0 = len(cur) if seed_coords is None else len(seed_coords)

    ff = xb.optimize(symbols, cur, charge=charge, solvent=None,
                     opt_level="loose", maxcycle=500, method="gfnff",
                     threads=threads, timeout=min(timeout, 1800), binary=binary)
    rec["rungs"].append({"rung": "gfnff", **{k: ff.get(k) for k in
                        ("ok", "reason", "xtb_converged", "cycles", "seconds")}})
    if ff.get("ok"):
        # GFN-FF has weak lanthanide parameterisation and can collapse the
        # coordination sphere.  A distorted pre-relax is discarded rather than
        # carried forward, because rung 2 would then refine the wrong structure.
        r = kabsch_rmsd(cur[:n0], np.asarray(ff["coords"])[:n0])
        rec["ff_rmsd_ang"] = r
        if r <= 0.5:
            cur = np.asarray(ff["coords"], dtype=float)
            rec["ff_prerelax"] = "used"
        else:
            rec["ff_prerelax"] = "skipped_distorted"
    else:
        rec["ff_prerelax"] = "skipped_failed"

    for level, mx in (("normal", 500), ("tight", 750)):
        res = xb.optimize(symbols, cur, charge=charge, solvent=SOLVENT,
                          opt_level=level, maxcycle=mx, method="gfn2",
                          threads=threads, timeout=timeout, binary=binary)
        if not res.get("ok") and res.get("reason") == "scf_not_converged":
            # etemp perturbs the electronic structure, so it is recorded as a
            # first-class field and an exclusion knob, never a silent retry.
            res = xb.optimize(symbols, cur, charge=charge, solvent=SOLVENT,
                              opt_level=level, maxcycle=mx, method="gfn2",
                              threads=threads, timeout=timeout, binary=binary,
                              etemp=750.0)
            rec["etemp_used"] = 750.0
        rec["rungs"].append({"rung": level, **{k: res.get(k) for k in
                            ("ok", "reason", "xtb_converged", "cycles",
                             "seconds", "force_max_ev_ang", "meets_target",
                             "rmsd_from_input_ang")}})
        if not res.get("ok"):
            if level == "tight" and rec.get("final_coords") is not None:
                # normal converged and tight did not: ship normal, flagged.
                rec["downgraded_opt_level"] = True
                return rec
            rec["fail_reason"] = res.get("reason")
            return rec
        cur = np.asarray(res["coords"], dtype=float)
        rec["final_coords"] = cur
        rec["energy_ev"] = res.get("energy_ev")
        rec["opt_level"] = level
        rec["force_max_ev_ang"] = res.get("force_max_ev_ang")
        rec["meets_target"] = res.get("meets_target")
        rec["xtb_converged"] = res.get("xtb_converged")
    return rec


def neutralize_one(row, *, threads: int = 1, timeout: int = 14400,
                   overwrite: bool = False) -> dict:
    """Both arms for one complex. Never raises; failures become a reject_code."""
    key = str(row["geometry_key"])
    safe = key.replace("/", "_").replace("|", "_")
    dest = OUT_ROOT / "records" / f"{safe}.json"
    if dest.exists() and not overwrite:
        return {"geometry_key": key, "status": "cached"}
    dest.parent.mkdir(parents=True, exist_ok=True)

    out: dict = {"geometry_key": key, "basename": row.get("basename"),
                 "metal": row.get("metal"),
                 "geometry_feature_build_id": row.get("geometry_feature_build_id"),
                 "done": True, "reject_code": "accepted"}
    try:
        binary = xb.find_xtb()
        if binary is None:
            out["reject_code"] = "NO_XTB"; return _write(dest, out)
        geom = read_extxyz(Path(str(row["local"])))
        symbols = [str(s) for s in geom.symbols]
        coords = np.asarray(geom.coordinates, dtype=float)
        out["n_atoms"] = len(symbols)

        try:
            mi = _metal_index(symbols)
        except ValueError as exc:
            out["reject_code"] = "MULTI_METAL"; out["reject_detail"] = str(exc)
            return _write(dest, out)
        out["metal_index"] = mi

        charge, prov = xb.infer_charge(geom)
        out["charge_provenance"] = prov
        if charge is None:
            out["reject_code"] = "CHARGE_UNRECOVERABLE"; return _write(dest, out)
        n_existing = inner_sphere_nitrate_count(symbols, coords, mi)
        out.update(charge_in=int(charge), n_nitrate_in=n_existing)
        # The relation charge == 3 - n_nitrate held for all 956 complexes when
        # measured.  A violation means the charge model is not what we think it
        # is, and guessing n_add would silently build the wrong molecule.
        if int(charge) != 3 - n_existing:
            out["reject_code"] = "CHARGE_MODEL_BROKEN"
            out["reject_detail"] = f"charge {charge} vs 3-{n_existing}"
            return _write(dest, out)
        n_add = int(charge)
        out["n_add"] = n_add
        out["cn_ligand_in"] = ligand_cn(symbols, coords, mi)

        # ---- place ------------------------------------------------------
        placed, dirs, recs = [], [], []
        for _ in range(n_add):
            extra = np.vstack(placed) if placed else None
            p = place_one(coords, symbols, mi, extra=extra, placed_dirs=dirs)
            if not p.get("ok"):
                out["reject_code"] = p.get("reason", "SEED_NO_FEASIBLE_POSE")
                out["n_feasible"] = p.get("n_feasible", 0)
                return _write(dest, out)
            placed.append(p["pos"]); dirs.append(p["u"])
            recs.append({k: (float(v) if isinstance(v, (int, float)) else
                             (list(map(float, v)) if k == "u" else v))
                         for k, v in p.items() if k != "pos"})
        out["placements"] = recs

        neu_sym = symbols + ["N", "O", "O", "O"] * n_add
        neu_xyz = np.vstack([coords] + placed)

        # ---- relax both arms --------------------------------------------
        ctl = _ladder(symbols, coords, int(charge), threads=threads,
                      timeout=timeout, binary=binary)
        out["control"] = {k: v for k, v in ctl.items() if k != "final_coords"}
        neu = _ladder(neu_sym, neu_xyz, 0, threads=threads, timeout=timeout,
                      binary=binary, seed_coords=coords)
        out["neutral"] = {k: v for k, v in neu.items() if k != "final_coords"}

        if ctl.get("final_coords") is None:
            out["reject_code"] = "CONTROL_FAILED"; return _write(dest, out)
        if neu.get("final_coords") is None:
            out["reject_code"] = neu.get("fail_reason", "XTB_CRASH") or "XTB_CRASH"
            return _write(dest, out)

        for arm, sym_, res in (("control", symbols, ctl), ("neutral", neu_sym, neu)):
            d = OUT_ROOT / arm
            d.mkdir(parents=True, exist_ok=True)
            write_extxyz(d / f"{safe}.xyz", sym_, res["final_coords"],
                         energy_ev=res.get("energy_ev"),
                         charge=(0 if arm == "neutral" else int(charge)),
                         solvent=SOLVENT)
            out[f"{arm}_xyz"] = str((d / f"{safe}.xyz").relative_to(_REPO_ROOT))
    except Exception as exc:                      # never raise into the pool
        out["reject_code"] = "EXCEPTION"
        out["reject_detail"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-2000:]
    return _write(dest, out)


def _write(dest: Path, rec: dict) -> dict:
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str))
    os.replace(tmp, dest)
    return rec


def _hno3_build_ids() -> set[str]:
    """Build ids whose rows were measured in nitric acid."""
    from automl.topo.train import build_row_table
    df, _, _ = build_row_table(preset="baseline_2d", arch="snn")
    sel = df["cond__acid__hno3"].fillna(0) > 0
    return set(df.loc[sel, "geometry_feature_build_id"].astype(str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--pilot", type=int, default=0,
                    help="whole ligand families, not random complexes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    jobs = job_table()
    # HNO3 only: placing nitrate on a complex measured in HCl is knowingly wrong
    # chemistry, and 98% of complexes map to exactly one acid so the choice is
    # unambiguous.  The join is on geometry_feature_build_id, NOT on
    # geometry_key -- geometry_key's trailing field is Architector's inner-sphere
    # FILL ligand ("...|nitrate" / "...|water"), which is a different thing from
    # the acid the measurement was made in.  Filtering on it would silently
    # select the wrong 618 structures.
    jobs = jobs[jobs["geometry_feature_build_id"].astype(str).isin(_hno3_build_ids())]
    if args.pilot:
        # WHOLE ligand families, not the first N by key.  Family-correlated
        # failure is the mode that silently reshapes a dataset, and a pilot that
        # takes one member of many families cannot see it -- the same mistake
        # was made and corrected once already in the conformer pilot.  The
        # ligand is the middle field of geometry_key: "<Z>|<SMILES>|<fill>".
        fam = jobs["geometry_key"].astype(str).str.split("|").str[1]
        jobs = jobs.assign(_fam=fam)
        picked, n = [], 0
        for f, grp in jobs.groupby("_fam", sort=True):
            if n >= args.pilot:
                break
            picked.append(grp); n += len(grp)
        jobs = pd.concat(picked).drop(columns=["_fam"])
        print(f"[neutralize] pilot: {len(jobs)} structures over "
              f"{len(picked)} whole ligand families", flush=True)
    jobs = jobs.iloc[args.shard::args.num_shards]
    print(f"[neutralize] {len(jobs)} structures, shard {args.shard}"
          f"/{args.num_shards}, {args.workers} workers", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(neutralize_one, r, threads=args.threads,
                          timeout=args.timeout, overwrite=args.overwrite): r
                for _, r in jobs.iterrows()}
        for f in as_completed(futs):
            rec = f.result(); done += 1
            if done % 10 == 0 or rec.get("reject_code") not in ("accepted", None):
                print(f"  [{done}/{len(jobs)}] {rec.get('geometry_key','?')[:48]} "
                      f"-> {rec.get('reject_code', rec.get('status'))}", flush=True)
    print(f"[neutralize] done {done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
