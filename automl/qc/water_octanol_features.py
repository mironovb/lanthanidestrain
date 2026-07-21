#!/usr/bin/env python3
"""The water->octanol reorganisation block: geometry's one non-redundant signal.

Why this is the feature worth building
--------------------------------------
Four independent tests this session established that the adjacent-lanthanide
*selectivity* signal is not in the GFN2-xTB geometry: the within-block geometric
contrast predicts logSF at R^2 ~= 0 (redundant with the tabular ionic radius, and
below the ~0.04 A optimisation-noise floor).  So encoding the geometry to improve
selectivity is capped, which is why every prior arm failed there.

But ``log D`` **is** a water/octanol partition coefficient, and every complex was
re-optimised in **both** solvents (``automl/qc/conformer_charges/{water,
octanol}``).  The complex's water->octanol geometric *response* is a direct 3D
probe of that exact partition -- and it is absent from 2D fingerprints and from
the single-solvent 3D block, which only ever sees one geometry.  A single scalar
(mean donor-distance shift) already adds +0.015 to leave-extractants-out overall
``log D`` R^2.  This builds the full response vector.

What it computes, per complex present in both solvents
------------------------------------------------------
Everything is a function of the two geometries and their Mulliken charges only --
no ``log D``, no new QC.  The coordination shell is taken from the shipped
function ``_coordination_shell`` (nearest ``core_cn`` donors), reusing the
asset-derived ``core_cn`` map in ``build_vr_conformers`` -- the only source that
reproduces the shipped donor set, established earlier this session.

Signed features (octanol - water) keep a direction; magnitude features use the
absolute change.  The sign convention is fixed here and tested in
``test_water_octanol.py``.

``data/`` is never written; output goes to ``automl/artifacts/water_octanol/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.geom3d_features import _kabsch_rmsd                 # noqa: E402
from automl.topo.build_vr_conformers import (                   # noqa: E402
    _original_index, donor_indices)

CHARGE_DIR = _REPO / "automl/artifacts/conformer_charges"
OUT_DIR = _REPO / "automl/artifacts/water_octanol"
OUT = OUT_DIR / "features.parquet"


# ---------------------------------------------------------------------------
def _load(npz: Path):
    z = np.load(npz, allow_pickle=False)
    return ([str(s) for s in z["symbols"]],
            z["coordinates"].astype(np.float64),
            z["partial_charges"].astype(np.float64))


def _angles(vectors: np.ndarray) -> np.ndarray:
    """All donor-metal-donor angles (deg) from metal->donor vectors."""
    n = len(vectors)
    if n < 2:
        return np.zeros(0)
    u = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(np.degrees(np.arccos(np.clip(u[i] @ u[j], -1.0, 1.0))))
    return np.asarray(out)


def _shift_features(water, octanol, meta, *, sign: float = 1.0) -> dict | None:
    """Reorganisation features for one complex, from its two solvent geometries.

    The feature is ``sign * (octanol - water)``; ``sign = -1`` (the swap test)
    gives ``water - octanol``.  The **donor set is always taken from water**,
    whatever the sign -- so a signed feature negates exactly under swap, and a
    magnitude feature is unchanged.  Defining donors from whichever solvent came
    first was the earlier bug: on the ~5 % of complexes where the nearest-donor
    set differs between solvents, the two directions then averaged over
    different atoms and did not negate.

    Returns None when the coordination rule fails, so the complex is dropped
    rather than given a fabricated zero shift (which would read as 'perfectly
    rigid' and inject a fake signal).
    """
    ms, mc = meta["metal_symbol"], meta["core_cn"]
    (sw, xw, qw), (so, xo, qo) = water, octanol
    if sw != so:                                   # different molecule -> refuse
        return None
    mi_w, don_w = donor_indices(sw, xw, ms, mc)
    mi_o, _don_o = donor_indices(so, xo, ms, mc)
    if mi_w is None or mi_o is None or not don_w:
        return None
    if mi_w != mi_o:                               # same molecule -> same metal atom
        return None

    # It is the same molecule in both solvents, so the atom indices correspond.
    # Fixing the donor SET from one solvent (water) and reading those SAME atoms
    # in the other is what makes every comparison below a true correspondence:
    #   * Kabsch RMSD needs matched points, and sorting donors independently by
    #     distance in each solvent broke that (median shell RMSD came out at
    #     2.25 A, larger than the whole complex -- the give-away);
    #   * per-donor distance and charge shifts are then per-ATOM, not
    #     sorted-rank-to-rank, so a donor that swapped rank does not masquerade
    #     as a large shift.
    don = np.asarray(don_w, dtype=int)             # donor set from WATER always
    dw = np.linalg.norm(xw[don] - xw[mi_w], axis=1)
    do = np.linalg.norm(xo[don] - xo[mi_o], axis=1)
    dshift = sign * (do - dw)                       # signed, same atoms
    ad = np.abs(dshift)
    order = np.argsort(dw)                          # stable per-complex ordering
    ad_sorted = ad[order]

    # Coordination-sphere rigid-body reorganisation: RMSD of the metal+donor
    # point set (same atoms) after optimal alignment -- a magnitude, so ``sign``
    # does not enter it.
    pw = np.vstack([xw[mi_w], xw[don]]); pw = pw - pw.mean(0)
    po = np.vstack([xo[mi_o], xo[don]]); po = po - po.mean(0)
    shell_rmsd = _kabsch_rmsd(pw, po)

    # Polyhedron deformation, same donor set both solvents (magnitude).
    aw = _angles(xw[don] - xw[mi_w])
    ao = _angles(xo[don] - xo[mi_o])
    dang = np.abs(ao - aw)

    # Charge redistribution, same atoms.
    dq_metal = sign * float(qo[mi_o] - qw[mi_w])
    dq_don = sign * (qo[don] - qw[don])

    # Whole-complex compaction.
    def rg(x):
        c = x.mean(0)
        return float(np.sqrt(((x - c) ** 2).sum(1).mean()))
    drg = sign * (rg(xo) - rg(xw))                 # signed, like the other deltas
    heavy_rmsd = (_kabsch_rmsd(xw - xw.mean(0), xo - xo.mean(0))
                  if xw.shape == xo.shape else np.nan)

    top = 8
    feat = {}
    for k in range(top):
        feat[f"wo_donor_dshift_{k+1:02d}"] = (float(ad_sorted[k])
                                              if k < len(ad_sorted) else 0.0)
    feat.update({
        "wo_donor_dshift_mean": float(ad.mean()),
        "wo_donor_dshift_std": float(ad.std()),
        "wo_donor_dshift_max": float(ad.max()),
        "wo_donor_dshift_signed_mean": float(dshift.mean()),
        "wo_shell_rmsd": float(shell_rmsd),
        "wo_angle_change_mean": float(dang.mean()),
        "wo_angle_change_max": float(dang.max()),
        "wo_metal_dq": dq_metal,
        "wo_donor_dq_mean": float(np.mean(dq_don)),
        "wo_donor_dq_absmean": float(np.mean(np.abs(dq_don))),
        "wo_donor_dq_std": float(np.std(dq_don)),
        "wo_rg_change": float(drg),
        "wo_heavy_rmsd": float(heavy_rmsd),
        "wo_core_cn": int(mc),
    })
    return feat


def build(limit: int = 0, swap: bool = False) -> pd.DataFrame:
    """Reorganisation features for every complex present in both solvents.

    ``swap`` computes water - octanol instead of octanol - water; used only by
    the sign-convention test, never written to the shipped parquet.
    """
    orig = _original_index()                       # basename -> {build_id, ...}
    by_stem = {Path(k).stem: v for k, v in orig.items()}
    W = {p.stem: p for p in (CHARGE_DIR / "water").glob("*.npz")}
    O = {p.stem: p for p in (CHARGE_DIR / "octanol").glob("*.npz")}
    both = sorted(set(W) & set(O))
    if limit:
        both = both[:limit]

    rows, dropped = [], {"no_meta": 0, "shell": 0}
    for stem in both:
        meta = by_stem.get(stem)
        if meta is None:
            dropped["no_meta"] += 1
            continue
        w, o = _load(W[stem]), _load(O[stem])
        # Water and octanol are passed in a fixed order (donors come from water
        # either way); ``swap`` only flips the sign, so the sign-convention test
        # exercises the direction without changing which atoms are measured.
        feat = _shift_features(w, o, meta, sign=(-1.0 if swap else 1.0))
        if feat is None:
            dropped["shell"] += 1
            continue
        feat["geometry_feature_build_id"] = str(meta["build_id"])
        rows.append(feat)
    df = pd.DataFrame(rows)
    print(f"[wo] built {len(df)} complexes from {len(both)} both-solvent pairs "
          f"(dropped {dropped})", flush=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    df = build(limit=args.limit)
    if df.empty:
        print("[wo] no rows built")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[wo] wrote {OUT}  ({df.shape[1]-1} features, {len(df)} complexes)")
    # A quick, honest look at the magnitudes so a degenerate build is obvious.
    num = df.drop(columns=["geometry_feature_build_id"]).select_dtypes("number")
    print("  feature medians:")
    for c in ("wo_donor_dshift_mean", "wo_shell_rmsd", "wo_metal_dq",
              "wo_angle_change_mean", "wo_heavy_rmsd"):
        if c in num:
            print(f"    {c:26s} {num[c].median():+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
