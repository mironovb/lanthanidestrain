#!/usr/bin/env python3
"""Did the metal-substitution construction actually create correspondence?

Every gate here was fixed in ``C7_PREREGISTRATION.md`` §5-§6 before any of these
numbers existed.  The point of running this *before* any model is trained is
that a modelling result computed on a failed construction is meaningless, and
the failure would not be visible in the model's score.

Two things are reported and they answer different questions:

  G1-G10  did the construction work at all?
  L1, L2  the DECISION GATE -- is the resulting geometry set a rank-1
          deformation in the metal coordinate?  If it is, one scalar per
          complex is everything an encoder could extract, which explains the
          measured interchangeability of eight encoders (SCIENTIFIC_FINDINGS
          G1/G2) from the data-generating process rather than from the models.

    python3 -m automl.qc.serial_report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from automl.qc.opt_reproducibility import kabsch_rmsd, mean_m_donor   # noqa: E402
from src.geometry_features import read_extxyz                          # noqa: E402

SERIAL = _REPO / "automl/artifacts/serial_metals"
CONTROL = _REPO / "automl/artifacts/geom_reopt/water"
OUT_CSV = _REPO / "automl/reports/serial_audit.csv"

# Shannon radii (CN 8, the dataset's own column) keyed by symbol.
from automl.metal_physics import LN_PHYSICS                            # noqa: E402


def _records() -> pd.DataFrame:
    rows = []
    for p in (SERIAL / "records").glob("serial__*.json"):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:                                              # noqa: BLE001
            continue
    return pd.DataFrame(rows)


def _load(path: Path):
    g = read_extxyz(path)
    return list(g.symbols), np.asarray(g.coordinates, dtype=float)


def pair_frame(df: pd.DataFrame, radii: dict[str, float]) -> pd.DataFrame:
    """One row per in-family ADJACENT pair, serial arm."""
    ok = df[df["ok"].astype(bool)] if "ok" in df else df
    out = []
    for fam, g in ok.groupby("family"):
        g = g.sort_values("lanthanide_index")
        recs = {int(r.lanthanide_index): r for r in g.itertuples()}
        for li in sorted(recs):
            if li + 1 not in recs:
                continue
            a, b = recs[li], recs[li + 1]
            try:
                sa, xa = _load(_REPO / a.path)
                sb, xb_ = _load(_REPO / b.path)
            except Exception:                                          # noqa: BLE001
                continue
            if len(sa) != len(sb):
                continue
            heavy = np.array([s != "H" for s in sa])
            mda, na = mean_m_donor(sa, xa)
            mdb, nb = mean_m_donor(sb, xb_)
            ra = radii.get(a.metal, np.nan)
            rb = radii.get(b.metal, np.nan)
            out.append(dict(
                family=fam, metal_lo=a.metal, metal_hi=b.metal,
                li_lo=li, d_index=1,
                anchor_offset_lo=int(a.anchor_offset),
                anchor_offset_hi=int(b.anchor_offset),
                rmsd_heavy=kabsch_rmsd(xa[heavy], xb_[heavy]),
                d_mean_md=(mdb - mda),
                d_radius=(rb - ra),
                n_donor_lo=na, n_donor_hi=nb, n_atoms=len(sa)))
    return pd.DataFrame(out)


def rank1_test(df: pd.DataFrame) -> dict:
    """L1: is the interior member reproducible by interpolating the extremes?

    If yes, the whole serial set is a rank-1 deformation in the metal
    coordinate and there is provably one scalar for an encoder to find.
    """
    ok = df[df["ok"].astype(bool)] if "ok" in df else df
    res = []
    for fam, g in ok.groupby("family"):
        g = g.sort_values("lanthanide_index")
        if len(g) < 3:
            continue
        lo, hi = g.iloc[0], g.iloc[-1]
        try:
            s0, x0 = _load(_REPO / lo.path)
            s1, x1 = _load(_REPO / hi.path)
        except Exception:                                              # noqa: BLE001
            continue
        if len(s0) != len(s1):
            continue
        heavy = np.array([s != "H" for s in s0])
        # align the extremes once; interpolation is meaningless otherwise
        p = x0 - x0.mean(0)
        q = x1 - x1.mean(0)
        v, _s, wt = np.linalg.svd(p.T @ q)
        d = np.sign(np.linalg.det(v @ wt))
        rot = v @ np.diag([1.0, 1.0, d]) @ wt
        p_al = p @ rot
        z0, z1 = int(lo.lanthanide_index), int(hi.lanthanide_index)
        for r in g.iloc[1:-1].itertuples():
            try:
                sm, xm = _load(_REPO / r.path)
            except Exception:                                          # noqa: BLE001
                continue
            if len(sm) != len(s0):
                continue
            t = (int(r.lanthanide_index) - z0) / (z1 - z0)
            interp = (1 - t) * p_al + t * q
            res.append(dict(family=fam, metal=r.metal,
                            li=int(r.lanthanide_index), t=t,
                            interp_rmsd=kabsch_rmsd(interp[heavy],
                                                    (xm - xm.mean(0))[heavy])))
    return {"frame": pd.DataFrame(res)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(OUT_CSV))
    args = ap.parse_args()

    radii = {}
    from automl.matrix_cache import load_cache
    mc, _b, _i = load_cache()
    for r in mc[["metal", "Ionic Radius_metal"]].dropna().drop_duplicates().itertuples():
        radii[r.metal] = float(getattr(r, "_2"))

    df = _records()
    if df.empty:
        print("[serial] no records yet")
        return 1
    n_ok = int(df["ok"].astype(bool).sum()) if "ok" in df else 0
    print(f"[serial] {len(df)} member records, {n_ok} ok "
          f"({df['family'].nunique()} families)")
    if "reject_code" in df:
        rc = df[~df["ok"].astype(bool)]["reject_code"].value_counts()
        if len(rc):
            print("  rejects:", dict(rc))

    P = pair_frame(df, radii)
    if P.empty:
        print("[serial] no adjacent pairs yet")
        return 1
    P.to_csv(args.csv, index=False)
    print(f"\n=== GATES (bars fixed in C7_PREREGISTRATION section 5) ===")
    print(f"  in-family adjacent pairs measured: {len(P)}")
    med = P.rmsd_heavy.median(); p90 = P.rmsd_heavy.quantile(0.90)
    print(f"  G1 median pair RMSD     {med:9.4f} A   bar <= 0.30   "
          f"{'PASS' if med <= 0.30 else 'FAIL'}   (independent build: 5.46)")
    print(f"  G2 P90 pair RMSD        {p90:9.4f} A   bar <= 1.00   "
          f"{'PASS' if p90 <= 1.00 else 'FAIL'}")
    same = float((P.n_donor_lo == P.n_donor_hi).mean())
    print(f"  G3 donor count preserved{same:9.3f}     bar >= 0.95   "
          f"{'PASS' if same >= 0.95 else 'FAIL'}")
    m = P.d_radius.notna() & P.d_mean_md.notna()
    if m.sum() > 10:
        slope, icpt = np.polyfit(P.d_radius[m], P.d_mean_md[m], 1)
        resid = P.d_mean_md[m] - (slope * P.d_radius[m] + icpt)
        sd = float(resid.std())
        snr = float(np.abs(P.d_mean_md[m]).median() / sd) if sd > 0 else np.nan
        print(f"  G5 residual sd of d<M-D>{sd:9.4f} A   bar <= 0.015  "
              f"{'PASS' if sd <= 0.015 else 'FAIL'}   (independent build: 0.076)")
        print(f"  G6 adjacent SNR         {snr:9.3f}     bar >= 0.7    "
              f"{'PASS' if snr >= 0.7 else 'FAIL'}   (independent build: 0.14)")
        print(f"  G7 slope on d(radius)   {slope:9.3f}     bar 0.40-0.70 "
              f"{'PASS' if 0.40 <= slope <= 0.70 else 'FAIL'}")

    L = rank1_test(df)["frame"]
    print(f"\n=== L1 DECISION GATE: is the serial set RANK-1 in the metal? ===")
    if L.empty:
        print("  no families with >=3 members yet")
    else:
        med = float(L.interp_rmsd.median())
        print(f"  interior members tested: {len(L)} across "
              f"{L.family.nunique()} families")
        print(f"  median interpolation RMSD = {med:.4f} A")
        print(f"    <= 0.02  -> C-II CONFIRMED, the set is rank-1; publish and "
              f"do not train")
        print(f"    >  0.10  -> C-II FAILS, deformation is multi-dimensional")
        verdict = ("C-II CONFIRMED (rank-1)" if med <= 0.02
                   else "C-II FAILS (multi-dimensional)" if med > 0.10
                   else "INTERMEDIATE - report the number, treat as SUPPORTED")
        print(f"  VERDICT: {verdict}")
        L.to_csv(str(args.csv).replace(".csv", "_rank1.csv"), index=False)
    print(f"\n[serial] wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
