#!/usr/bin/env python3
"""Turn the reference xTB energies into a geometry-keyed feature block.

Pre-registered in ``ENERGY_PREREGISTRATION.md``.  Consumes the per-geometry JSONs
written by ``automl.qc.reference_xtb --run`` and emits one row per
``geometry_key`` with a ``gE__`` prefix, ready to merge in ``dataset.build_matrix``
exactly the way the geometric blocks are merged.

Two families of column, and the second is the one that matters
---------------------------------------------------------------
**Absolute** (``gE__abs__``): interaction energy, transfer energy, frontier
orbitals, metal charge.  These are largely *ligand* properties -- swap the ligand
and they move by electronvolts; swap Nd for Pm and they move by a fraction of
one.  The ECFP block already identifies the ligand perfectly, so on their own
these mostly restate something the model knows.

**Family-relative** (``gE__rel_d``/``_z``/``_r``): the same quantities expressed
against their own ``ligand|anion`` family, which is the same complex across the
lanthanide series.  Subtracting the family mean removes everything that only
says "which ligand is this" and keeps how *this particular cation* sits in
*that particular cavity* -- which is precisely the quantity the adjacent-pair
metric scores.

The relative transform is ``dataset.add_within_ligand_relative``, reused rather
than reimplemented: it already carries the rule that a one-member family has no
relative information and must be NaN rather than 0, and getting that wrong would
hand the model a fake zero for every complex with no series partner.

Legality under leave-extractants-out: the transform touches no target value, and
for an unseen extractant the family is built from that extractant's own
generated geometries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/xtb_reference"
OUT = ART / "energy_features.parquet"

# The quantities carried forward.  Deliberately short: 953 complexes is a thin
# structural sample, and every extra column is another chance to fit noise.
KEEP = [
    "e_int_water_ev", "e_int_octanol_ev",      # metal-cage interaction
    "d_e_int_ev",                              # its solvent dependence
    "dg_transfer_ev",                          # water -> octanol; log D is a
                                               # partition coefficient and
                                               # nothing else expresses one
    "homo_water_ev", "lumo_water_ev", "gap_water_ev",
    "homo_octanol_ev", "lumo_octanol_ev", "gap_octanol_ev",
    "q_metal_water", "q_transfer_water",
    "q_metal_octanol", "q_transfer_octanol",
]


def build(verbose: bool = True) -> pd.DataFrame:
    from automl.qc.reference_xtb import collect
    from automl.dataset import add_within_ligand_relative, DATASET_PATH

    raw = collect()
    if raw.empty:
        raise SystemExit("no reference energies on disk; run reference_xtb --run")
    ok = raw[raw["status"] == "ok"].copy() if "status" in raw else raw.copy()
    if verbose:
        print(f"[energy] {len(raw)} geometries, {len(ok)} with status=ok")

    have = [c for c in KEEP if c in ok.columns]
    missing = [c for c in KEEP if c not in ok.columns]
    if missing and verbose:
        print(f"[energy] absent from the JSONs: {missing}")

    base = ok[["geometry_key"] + have].copy()
    for c in have:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    # One row per geometry.  A duplicate would silently double-weight a complex
    # in the family mean below.
    base = base.drop_duplicates("geometry_key").reset_index(drop=True)

    # Family keys, built the same way build_matrix builds them.
    src = pd.read_parquet(DATASET_PATH, columns=["geometry_key"])
    parts = src["geometry_key"].astype(str).str.split("|", n=2, expand=True)
    fam = pd.DataFrame({
        "geometry_key": src["geometry_key"],
        "ligand_anion_family": parts[1].fillna("") + "|" + parts[2].fillna(""),
    }).drop_duplicates("geometry_key")

    rel = add_within_ligand_relative(base, fam)
    # add_within_ligand_relative labels its output g10__; these are not g10.
    rel = rel.rename(columns={c: c.replace("g10__", "gE__")
                              for c in rel.columns if c.startswith("g10__")})
    out = base.rename(columns={c: f"gE__abs__{c}" for c in have})
    out = out.merge(rel, on="geometry_key", how="left")

    if verbose:
        n_rel = sum(c.startswith("gE__rel") for c in out.columns)
        print(f"[energy] {len(out)} geometries x "
              f"{len([c for c in out.columns if c.startswith('gE__')])} columns "
              f"({len(have)} absolute, {n_rel} family-relative)")
        # A column that is all-NaN would be dropped downstream anyway, but
        # silently -- say so here instead.
        dead = [c for c in out.columns
                if c.startswith("gE__") and out[c].isna().all()]
        if dead:
            print(f"[energy] {len(dead)} all-NaN columns (will be dropped): "
                  f"{dead[:4]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    out = build(verbose=not args.quiet)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(OUT)
    print(f"[energy] wrote {OUT}")

    # A one-line sanity report on the physics, because a bookkeeping error here
    # would look exactly like a chemistry result downstream.
    for c in ("gE__abs__e_int_water_ev", "gE__abs__dg_transfer_ev"):
        if c in out.columns and out[c].notna().any():
            v = out[c].dropna()
            print(f"  {c:34s} n={len(v):4d} median={v.median():+10.3f} eV "
                  f"IQR=[{v.quantile(.25):+.3f}, {v.quantile(.75):+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
