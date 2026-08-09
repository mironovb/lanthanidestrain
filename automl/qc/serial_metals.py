#!/usr/bin/env python3
"""Build lanthanide series IN CORRESPONDENCE, by metal substitution.

The problem this fixes
----------------------
The shipped complexes were generated independently per (ligand, metal).  Two
adjacent lanthanides of the same ligand therefore sit in **different conformer
basins**: median heavy-atom RMSD between them is **5.46 A**, and — the decisive
control — that RMSD is *flat* in |delta lanthanide index| (5.46 at delta = 1,
5.77 at delta = 7).  La-vs-Ce looks exactly like La-vs-Lu.

So the structural difference between adjacent lanthanides, which is the only
thing an encoder could use for selectivity, is ~99 % conformer sampling.  The
contraction physics is present but buried: mean M-donor bond tracks the Shannon
radius step with slope 0.505 (chemically right, bond contracts about half the
radius step) at a per-pair sd of 0.076 A, i.e. **SNR 0.14**.

The construction
----------------
Per ligand family, take ONE member's already-relaxed structure, substitute only
the metal symbol, and re-optimise.  Every member then descends from one basin,
so the pair difference is the contraction response rather than the sampling.

Three arms come out of one pass:

  water   already on disk -- each complex relaxed from its own start (incumbent)
  frozen  anchor coordinates verbatim, metal token swapped, single point only
          -> perfect correspondence, ZERO geometric response
  serial  substitute then optimise -> correspondence PLUS response

``serial - frozen`` isolates the geometric response; ``frozen - water`` isolates
correspondence itself.  Without ``frozen`` a positive result is uninterpretable,
and it costs ~3 CPU-hours.

Design decisions that are load-bearing
--------------------------------------
* **Anchor from a converged minimum, not the shipped start.**  Re-optimising a
  shipped structure moves it a median 1.87 A; two near-identical starts can then
  bifurcate, reproducing the exact failure being engineered out.
* **Median-index anchor.**  Substitution distance |Z - Z_anchor| is then V-shaped
  in the index, so any anchor-distance artefact is EVEN in the index and cannot
  masquerade as a monotone contraction trend.
* **The anchor is re-run through the identical path.**  Otherwise it has had one
  relaxation and its family members two, and it is systematically better
  converged.  This also yields idempotency (gate G8) for free.
* **Charge is constant per family**, taken once from the anchor.  All metals are
  Ln(III) and a family shares its ligand set by construction, so the charge
  cannot change; 158 of 159 families verify this and the one that does not is
  rejected rather than patched.

xtb settings are byte-identical to ``geom_reopt/water`` -- GFN2 / ALPB water /
--opt tight / maxcycle 750 / --norestart / uhf 0 / independent --grad check --
so the control arm is a fair comparison and not a different calculation.

    python3 -m automl.qc.serial_metals --pilot 12
    python3 -m automl.qc.serial_metals --shard 0 --num-shards 2 --workers 48
    python3 -m automl.qc.serial_metals --mode frozen --workers 48
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.qc import xtb_backend as xb                                # noqa: E402
from automl.qc.neutralize import (_metal_index, kabsch_rmsd,           # noqa: E402
                                  ligand_cn, write_extxyz_with_charges)
from src.geometry_features import read_extxyz                          # noqa: E402

CONTROL = _REPO / "automl/artifacts/geom_reopt/water"
OUT = _REPO / "automl/artifacts/serial_metals"
DATASET = _REPO / "data/processed/final_ml_dataset_3d.parquet"

SOLVENT, OPT_LEVEL, MAXCYCLE = "water", "tight", 750
RMSD_SOFT, RMSD_HARD = 0.5, 1.0      # neutralize.py's own "distorted" rule

LANTH = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
         "Er", "Tm", "Yb", "Lu"]


def family_table() -> pd.DataFrame:
    """Complexes grouped into families that are ONE molecule up to the metal.

    The family key carries the composition columns, not just the ligand: the
    dataset has a hard coordination switch at lanthanide index 8->9 (CN 9 below,
    CN 8 above, zero exceptions), so (ligand, fill) alone would put two
    genuinely different molecules in one family and the "substitution" would be
    adding or removing a whole ligand.
    """
    df = pd.read_parquet(DATASET)
    keep = ["geometry_key", "geometry_feature_build_id", "metal",
            "lanthanide_index", "coreCN", "n_ligs", "n_fill",
            "inner_sphere_anion", "fill_ligand", "geometry_ok"]
    d = df[[c for c in keep if c in df.columns]].copy()
    d = d[d["geometry_ok"].astype(bool)].drop_duplicates("geometry_key")
    parts = d["geometry_key"].str.split("|", n=2, expand=True)
    d["lig"] = parts[1].fillna("")
    d["fill"] = parts[2].fillna("")
    d["family"] = (d["lig"] + "||" + d["fill"] + "||"
                   + d["coreCN"].astype(str) + "||" + d["n_ligs"].astype(str)
                   + "||" + d["n_fill"].astype(str) + "||"
                   + d["inner_sphere_anion"].astype(str))
    # local path of the already-relaxed control structure
    stem = {p.stem: p for p in CONTROL.glob("*.xyz")}
    idx = _build_id_to_stem()
    d["src"] = d["geometry_feature_build_id"].map(
        lambda b: stem.get(idx.get(str(b), ""), None))
    d = d[d["src"].notna()].reset_index(drop=True)
    d["lanthanide_index"] = d["lanthanide_index"].astype(int)
    return d


def _build_id_to_stem() -> dict[str, str]:
    """build_id -> control-file stem, via the geometry status index."""
    p = _REPO / "data/processed/feature_blocks/geometry_feature_status.csv"
    s = pd.read_csv(p)
    out = {}
    for r in s.itertuples():
        xp = str(getattr(r, "xyz_path", "") or "")
        if xp:
            out[str(r.build_id)] = Path(xp).stem
    return out


def _charge_from_sidecar(src: Path) -> int | None:
    """Molecular charge recorded by the run that produced this structure."""
    j = src.with_suffix(".json")
    if not j.exists():
        return None
    try:
        c = json.loads(j.read_text()).get("charge")
    except Exception:                                        # noqa: BLE001
        return None
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


def choose_anchor(fam: pd.DataFrame) -> pd.Series:
    """Median-index member; ties resolve to the lower index (deterministic)."""
    order = fam.sort_values("lanthanide_index").reset_index(drop=True)
    return order.iloc[(len(order) - 1) // 2]


def substitute(symbols: list[str], to_metal: str) -> list[str]:
    """Relabel the single lanthanide. Coordinates are untouched."""
    mi = _metal_index(symbols)                # raises unless exactly one
    out = list(symbols)
    out[mi] = to_metal
    return out


def _reject(fam_id: str, code: str, extra: dict | None = None) -> dict:
    rec = {"family": fam_id, "ok": False, "reject_code": code}
    rec.update(extra or {})
    (OUT / "records").mkdir(parents=True, exist_ok=True)
    p = OUT / "records" / f"{abs(hash(fam_id)) % 10**12}.json"
    _dump(p, rec)
    return rec


def serial_one(fam_id: str, fam: pd.DataFrame, *, mode: str = "serial",
               threads: int = 1, timeout: int = 14400,
               overwrite: bool = False) -> dict:
    """Every member of one family, all descending from the same anchor basin."""
    t0 = time.time()
    anchor = choose_anchor(fam)
    try:
        g0 = read_extxyz(Path(anchor.src))
    except Exception as exc:                                  # noqa: BLE001
        return _reject(fam_id, "ANCHOR_UNREADABLE", {"error": str(exc)})
    sym0 = list(g0.symbols)
    xyz0 = np.asarray(g0.coordinates, dtype=float)

    # --- family-level gates -------------------------------------------------
    formulas, charges = set(), set()
    for r in fam.itertuples():
        try:
            g = read_extxyz(Path(r.src))
        except Exception:                                     # noqa: BLE001
            return _reject(fam_id, "MEMBER_UNREADABLE", {"member": r.metal})
        s = [x for x in g.symbols if str(x) not in LANTH]
        formulas.add(tuple(sorted(s)))
        # Control structures carry no Mulliken populations (reoptimize.py wrote
        # them without), so the charge comes from the sidecar the run recorded,
        # not from a re-inference that would silently fail.
        charges.add(_charge_from_sidecar(Path(r.src)))
    if len(formulas) != 1:
        return _reject(fam_id, "FORMULA_MISMATCH", {"n_formulas": len(formulas)})
    if len(charges) != 1 or None in charges:
        return _reject(fam_id, "CHARGE_NOT_CONSTANT",
                       {"charges": sorted(str(c) for c in charges)})
    charge = int(charges.pop())

    try:
        mi = _metal_index(sym0)
    except ValueError as exc:
        return _reject(fam_id, "MULTI_METAL", {"error": str(exc)})
    cn0 = ligand_cn(sym0, xyz0, mi)

    members, n_ok = [], 0
    for r in fam.itertuples():
        stem = f"{Path(r.src).stem}__from_{Path(anchor.src).stem}"
        dest = OUT / mode / f"{stem}.xyz"
        rec_p = OUT / "records" / f"{mode}__{stem}.json"
        rec_p.parent.mkdir(parents=True, exist_ok=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if rec_p.exists() and not overwrite:
            members.append(json.loads(rec_p.read_text()))
            n_ok += bool(members[-1].get("ok"))
            continue
        rec = {"family": fam_id, "mode": mode, "metal": r.metal,
               "lanthanide_index": int(r.lanthanide_index),
               "geometry_key": r.geometry_key,
               "build_id": str(r.geometry_feature_build_id),
               "anchor_metal": anchor.metal,
               "anchor_index": int(anchor.lanthanide_index),
               "anchor_offset": int(r.lanthanide_index) - int(anchor.lanthanide_index),
               "charge": charge, "n_atoms": len(sym0)}
        try:
            sym = substitute(sym0, str(r.metal))
            if mode == "frozen":
                res = xb.single_point(sym, xyz0, charge=charge,
                                      uhf=xb.DEFAULT_UHF, solvent=SOLVENT,
                                      threads=threads, timeout=timeout)
                coords = xyz0
                res.setdefault("energy_ev", res.get("energy_ev"))
            else:
                res = xb.optimize(sym, xyz0, charge=charge, uhf=xb.DEFAULT_UHF,
                                  solvent=SOLVENT, opt_level=OPT_LEVEL,
                                  maxcycle=MAXCYCLE, threads=threads,
                                  timeout=timeout)
                coords = (np.asarray(res["coords"], dtype=float)
                          if res.get("ok") else xyz0)
            rec.update({k: v for k, v in res.items()
                        if k not in ("coords", "charges", "symbols")})
            if res.get("ok"):
                rec["rmsd_from_anchor"] = kabsch_rmsd(coords, xyz0)
                rec["cn"] = ligand_cn(sym, coords, mi)
                if rec["rmsd_from_anchor"] > RMSD_HARD:
                    rec.update(ok=False, reject_code="RMSD_FROM_START")
                elif rec["cn"] != cn0:
                    rec.update(ok=False, reject_code="CN_CHANGED",
                               cn_anchor=cn0)
                else:
                    rec["soft_rmsd_flag"] = bool(
                        rec["rmsd_from_anchor"] > RMSD_SOFT)
                    # Per-atom Mulliken charges are NOT optional.  optimize()
                    # does not return them, and write_extxyz without them makes
                    # infer_charge fail downstream -- which turns "charge
                    # missing" into a marker identifying exactly which
                    # structures this pipeline generated.  CAMPAIGN4's
                    # pre-registration forbids repeating that, so a single
                    # point on the relaxed geometry supplies them.
                    q = res.get("partial_charges")
                    if q is None:
                        sp = xb.single_point(sym, coords, charge=charge,
                                             uhf=xb.DEFAULT_UHF,
                                             solvent=SOLVENT, threads=threads,
                                             timeout=timeout)
                        q = sp.get("partial_charges")
                        rec["charges_from_single_point"] = bool(q is not None)
                    if q is None:
                        rec.update(ok=False, reject_code="NO_MULLIKEN")
                    else:
                        write_extxyz_with_charges(
                            dest, sym, coords, q,
                            energy_ev=res.get("energy_ev"), charge=charge,
                            solvent=SOLVENT)
                        rec["path"] = str(dest.relative_to(_REPO))
        except Exception as exc:                              # noqa: BLE001
            rec.update(ok=False, reject_code=f"EXC:{type(exc).__name__}",
                       error=str(exc))
        _dump(rec_p, rec)
        members.append(rec)
        n_ok += bool(rec.get("ok"))
    return {"family": fam_id, "ok": n_ok == len(fam), "n_members": len(fam),
            "n_ok": n_ok, "anchor": anchor.metal, "charge": charge,
            "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("serial", "frozen"), default="serial")
    ap.add_argument("--pilot", type=int, default=0,
                    help="this many whole FAMILIES (never partial families)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    d = family_table()
    fams = {k: g for k, g in d.groupby("family") if len(g) >= 2}
    if args.stats:
        n = sum(len(g) for g in fams.values())
        print(f"[serial] {len(d)} complexes, {len(fams)} multi-metal families "
              f"covering {n} complexes")
        sizes = pd.Series([len(g) for g in fams.values()])
        print(f"  members/family: median {sizes.median():.0f} max {sizes.max()}")
        adj = sum(sum(1 for a, b in zip(sorted(g.lanthanide_index)[:-1],
                                        sorted(g.lanthanide_index)[1:])
                      if b - a == 1) for g in fams.values())
        print(f"  in-family ADJACENT pairs: {adj}")
        return 0

    if xb.find_xtb() is None:
        raise SystemExit("no xtb binary; set XTB_BIN")
    keys = sorted(fams)
    if args.pilot:
        keys = keys[:args.pilot]
    keys = [k for i, k in enumerate(keys) if i % args.num_shards == args.shard]
    print(f"[serial] mode={args.mode} {len(keys)} families on shard "
          f"{args.shard}/{args.num_shards}, {args.workers} workers", flush=True)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(serial_one, k, fams[k], mode=args.mode,
                            threads=args.threads, timeout=args.timeout,
                            overwrite=args.overwrite) for k in keys]
        for i, f in enumerate(futs, 1):
            r = f.result()
            done += bool(r.get("ok"))
            if i % 5 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} families, {done} fully ok, "
                      f"{time.time() - t0:.0f}s", flush=True)
    print(f"[serial] done {done}/{len(keys)} families in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
