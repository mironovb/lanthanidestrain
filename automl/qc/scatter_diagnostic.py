#!/usr/bin/env python3
"""Stage 2 GO/NO-GO: did tighter geometries actually recover signal?

The single most informative number in the whole plan, and it is cheap.

Background
----------
Within one ligand+anion family, a geometric descriptor should vary *smoothly*
with the lanthanide's ionic radius -- that smooth response is exactly the
physics that drives selectivity.  The prior study measured how much of each
descriptor's family-wise variance a straight line in ionic radius explains
(``g13__fitr2__*``) and found a median of only 0.20-0.37: the descriptors were
mostly scatter at the scale that separates adjacent lanthanides, where the
radius step is ~0.013 A while single-conformer noise in an M-L distance is
~0.05 A.

The shipped geometries stopped on an ``fmax = 0.2 eV/A`` criterion, so some of
that scatter is *optimisation* noise rather than conformational diversity.  If
that diagnosis is right, re-optimising to ~0.003 eV/A should raise this fit R2.
If it does not move, the scatter is conformational, no amount of tightening
will fix it, and Stages 3-6 face the same headwind the prior study hit.  That
conclusion gets reported either way, before more compute is spent -- which is
the entire point of putting the checkpoint here.

The comparison is **paired per descriptor column and per family**, because
families differ enormously in how many metals they contain; an unpaired median
would mostly measure which families happen to be present in each set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

def _read_many(spec: str) -> pd.DataFrame:
    """Accept a file, a directory, or a glob -- the diagnostic needs whole
    families, and the descriptor tables are written one parquet per shard."""
    paths = sorted(Path().glob(spec)) if any(c in spec for c in "*?[") else [Path(spec)]
    if len(paths) == 1 and paths[0].is_dir():
        paths = sorted(paths[0].glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet matched {spec!r}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


RADIUS_COL = "g2__contraction__ionic_radius"
MIN_MEMBERS = 4


def family_fit_r2(geom: pd.DataFrame, keys: pd.DataFrame,
                  min_members: int = MIN_MEMBERS) -> pd.DataFrame:
    """Per (family, descriptor) R2 of a straight line in ionic radius.

    Same fit as ``automl.dataset.add_series_smoothed`` produces in its
    ``g13__fitr2__`` block, but returned long-form so the two geometry sets can
    be paired on (family, descriptor) rather than compared as two pooled piles.
    """
    merged = geom.merge(keys, on="geometry_key", how="left")
    if RADIUS_COL not in merged.columns:
        raise SystemExit(f"missing {RADIUS_COL}; cannot run the diagnostic")
    fam = merged["ligand_anion_family"].astype(str).to_numpy()
    r = merged[RADIUS_COL].to_numpy(dtype=float)
    num_cols = [c for c in geom.columns
                if c != "geometry_key" and pd.api.types.is_numeric_dtype(geom[c])
                and c != RADIUS_COL]
    values = merged[num_cols].to_numpy(dtype=float)

    rows = []
    for f, idx in pd.Series(np.arange(len(merged))).groupby(fam).groups.items():
        idx = np.asarray(list(idx))
        rr = r[idx]
        ok_r = np.isfinite(rr)
        if idx.size < min_members or ok_r.sum() < min_members:
            continue
        v = values[idx]
        for j, col in enumerate(num_cols):
            m = ok_r & np.isfinite(v[:, j])
            if m.sum() < min_members:
                continue
            x, yv = rr[m], v[m, j]
            if np.ptp(x) < 1e-9 or np.ptp(yv) < 1e-12:
                continue
            b, a = np.polyfit(x, yv, 1)
            ss_tot = float(np.sum((yv - yv.mean()) ** 2))
            if ss_tot <= 0:
                continue
            ss_res = float(np.sum((yv - (a + b * x)) ** 2))
            rows.append({"family": f, "descriptor": col,
                         "n": int(m.sum()), "fit_r2": 1.0 - ss_res / ss_tot})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loose", required=True,
                    help="descriptor parquet from the shipped geometries")
    ap.add_argument("--tight", required=True,
                    help="descriptor parquet from the re-optimised geometries")
    ap.add_argument("--out", default=str(_REPO / "automl/reports/scatter_diagnostic.csv"))
    args = ap.parse_args()

    # ligand_anion_family is derived, not stored: geometry_key is
    # "<Z>|<ligand>|<anion>", so the family is everything after the metal.
    # Constructed here exactly as automl.dataset does it, so the families are
    # the same ones the g13 block was built on.
    keys = pd.read_parquet(
        _REPO / "data/processed/final_ml_dataset_3d.parquet",
        columns=["geometry_key"]).drop_duplicates("geometry_key")
    parts = keys["geometry_key"].str.split("|", n=2, expand=True)
    keys["ligand_anion_family"] = parts[1].fillna("") + "|" + parts[2].fillna("")

    loose = _read_many(args.loose)
    tight = _read_many(args.tight)
    for name, frame in (("loose", loose), ("tight", tight)):
        print(f"[scatter] {name}: {len(frame)} geometries, "
              f"{frame.shape[1]-1} descriptor columns")
    a_raw = family_fit_r2(loose, keys)
    b_raw = family_fit_r2(tight, keys)
    for name, frame in (("loose", a_raw), ("tight", b_raw)):
        if frame.empty:
            print(f"[scatter] no family had >= {MIN_MEMBERS} members with a "
                  f"finite ionic radius in the {name} set -- the diagnostic "
                  f"needs whole families, so pass all shards, not one")
            return 1
    a = a_raw.rename(columns={"fit_r2": "fit_r2_loose",
                                                   "n": "n_loose"})
    b = b_raw.rename(columns={"fit_r2": "fit_r2_tight", "n": "n_tight"})
    m = a.merge(b, on=["family", "descriptor"], how="inner")
    if m.empty:
        print("[scatter] no (family, descriptor) pairs in common -- cannot compare")
        return 1

    d = m["fit_r2_tight"] - m["fit_r2_loose"]
    print(f"paired (family, descriptor) cells : {len(m)}")
    print(f"  families                        : {m['family'].nunique()}")
    print(f"  descriptors                     : {m['descriptor'].nunique()}")
    print("")
    print(f"  median fit R2, loose geometries  : {m['fit_r2_loose'].median():.4f}")
    print(f"  median fit R2, tight geometries  : {m['fit_r2_tight'].median():.4f}")
    print(f"  median paired change             : {d.median():+.4f}")
    print(f"  cells improved                   : {int((d > 0).sum())}/{len(d)} "
          f"({100*float((d > 0).mean()):.1f}%)")
    # Wilcoxon on the paired differences: the cells are not independent across
    # descriptors, so this is a descriptive check, not a licence to claim a
    # p-value for the scientific conclusion.
    try:
        w = stats.wilcoxon(m["fit_r2_tight"], m["fit_r2_loose"])
        print(f"  Wilcoxon signed-rank             : stat={w.statistic:.0f} "
              f"p={w.pvalue:.3g}  (descriptive: descriptor cells are correlated)")
    except Exception as exc:
        print(f"  Wilcoxon unavailable: {exc}")

    print("")
    print("  largest median gains by descriptor family:")
    m["block"] = m["descriptor"].str.split("__").str[0]
    byb = (m.assign(delta=d).groupby("block")["delta"]
           .agg(["median", "size"]).sort_values("median", ascending=False))
    for blk, row in byb.head(8).iterrows():
        print(f"    {blk:28s} {row['median']:+.4f}  (n={int(row['size'])})")

    print("")
    if d.median() > 0.02:
        print("  VERDICT: tighter geometries measurably reduce family scatter.")
        print("           The 0.2 eV/A criterion was destroying real signal.")
    elif d.median() < -0.02:
        print("  VERDICT: tighter geometries are WORSE on this diagnostic.")
        print("           Re-optimisation moved structures away from the")
        print("           conformers the measurements correspond to.")
    else:
        print("  VERDICT: no material change. The scatter is conformational,")
        print("           not convergence-driven; tightening cannot fix it and")
        print("           Stages 3-6 face the same headwind as the prior study.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.assign(delta=d).to_csv(out, index=False)
    print(f"\n[scatter] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
