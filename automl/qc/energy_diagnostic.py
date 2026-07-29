#!/usr/bin/env python3
"""Why the reference energetics hurt selectivity, measured rather than assumed.

The result being explained
--------------------------
Adding the ``gE`` block to CatBoost raised overall log D R2 (+0.4987 ->
+0.5068) and **collapsed** the adjacent-pair metric (+0.1422 -> -0.0350).
That is a large, reproducible effect in the wrong direction, and a mechanism for
it has to be measured, not narrated.

The hypothesis
--------------
It is the failure mode ``dataset.py`` already documents for the *geometric*
blocks, and the sections that build g12/g13/g14 state it plainly:

    the baseline's clean, strictly monotone lanthanide descriptors (Z, index,
    Shannon radius) were being replaced by geometry proxies that carry the same
    trend *plus* conformer noise.

If the energy features behave the same way, then within one ligand family the
energy should carry the series trend but with scatter large compared with the
per-step change -- and a tree model, which only ever compares values, will split
on the noisy proxy in place of the exact lookup.

Why this is not what the metal-substitution probe measured
----------------------------------------------------------
``reference_xtb --probe`` substituted all 14 lanthanides into **one frozen
cage** and found adjacent members separated by 0.306 eV, 17x the scale a useful
separation factor corresponds to.  That is a real property of the method and it
is why the campaign was worth running.

But it held the geometry fixed, and the dataset does not: every complex is a
separate Architector/GFN2 conformer.  So the probe could establish that GFN2
resolves the elements, and could *not* establish that the resolution survives
conformer-to-conformer variation in the actual structures.  The pre-registration
said the probe "rules out a specific failure mode; it is not evidence for the
hypothesis".  This module measures the quantity the probe could not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "automl/reports/energy_diagnostic.csv"
FEATURES = REPO / "automl/artifacts/xtb_reference/energy_features.parquet"
DATASET = REPO / "data/processed/final_ml_dataset_3d.parquet"

# Absolute quantities only.  The family-relative columns are this same signal
# with the family mean removed, so measuring their within-family SNR would be
# circular.
PROBES = ["gE__abs__e_int_water_ev", "gE__abs__e_int_octanol_ev",
          "gE__abs__dg_transfer_ev", "gE__abs__gap_water_ev",
          "gE__abs__q_metal_water"]


def load() -> pd.DataFrame:
    en = pd.read_parquet(FEATURES)
    ds = (pd.read_parquet(DATASET,
                          columns=["geometry_key", "lanthanide_index",
                                   "Ionic Radius_metal", "metal"])
          .drop_duplicates("geometry_key"))
    d = en.merge(ds, on="geometry_key", how="inner")
    parts = d["geometry_key"].astype(str).str.split("|", n=2, expand=True)
    # Same family definition build_matrix uses: ligand|anion, i.e. the same
    # complex across the lanthanide series.
    d["fam"] = parts[1].fillna("") + "|" + parts[2].fillna("")
    return d


def per_family(d: pd.DataFrame, col: str, min_members: int = 5) -> pd.DataFrame:
    """Series trend and residual scatter of one feature, family by family."""
    rows = []
    for fam, g in d.groupby("fam"):
        g = g.dropna(subset=[col, "lanthanide_index"])
        if len(g) < min_members:
            continue
        x = g["lanthanide_index"].to_numpy(float)
        y = g[col].to_numpy(float)
        if y.std() < 1e-12 or x.std() < 1e-12:
            continue
        A = np.column_stack([x, np.ones_like(x)])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        sd = float(resid.std(ddof=1))
        rows.append({"family": fam, "n": len(g),
                     "step_per_index": float(abs(beta[0])),
                     "residual_sd": sd,
                     "snr": float(abs(beta[0]) / max(sd, 1e-12))})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-members", type=int, default=5)
    args = ap.parse_args()

    d = load()
    print(f"[diag] {len(d)} geometries, {d['fam'].nunique()} ligand families")

    # The thing the energy features are competing against.
    u = (d.dropna(subset=["Ionic Radius_metal"]).drop_duplicates("metal")
         [["metal", "lanthanide_index", "Ionic Radius_metal"]]
         .sort_values("lanthanide_index"))
    step_r = float(np.abs(np.diff(u["Ionic Radius_metal"].to_numpy())).mean())
    print(f"\n[diag] the incumbent descriptor: Ionic Radius_metal is a lookup "
          f"table over {len(u)} metals,\n       mean adjacent step "
          f"{step_r:.4f} A, residual scatter 0 by construction (SNR infinite).")

    rows = []
    print(f"\n{'feature':34s} {'fams':>5s} {'step/index':>11s} "
          f"{'resid SD':>10s} {'SNR':>7s} {'SNR<1':>7s}")
    for col in PROBES:
        if col not in d.columns:
            continue
        f = per_family(d, col, args.min_members)
        if f.empty:
            continue
        snr = float(f["snr"].median())
        frac = float((f["snr"] < 1).mean())
        print(f"{col:34s} {len(f):5d} {f['step_per_index'].median():11.4f} "
              f"{f['residual_sd'].median():10.4f} {snr:7.3f} {frac:6.0%}")
        rows.append({"feature": col, "n_families": len(f),
                     "median_step_per_index": f["step_per_index"].median(),
                     "median_residual_sd": f["residual_sd"].median(),
                     "median_snr": snr, "frac_families_snr_below_1": frac})

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("\n=== reading ===")
    if out.empty:
        print("  no usable features")
        return 1
    best = out.loc[out["median_snr"].idxmax()]
    worst_frac = float(out["frac_families_snr_below_1"].min())
    print(f"  The best-behaved energy feature is {best['feature']} at "
          f"SNR {best['median_snr']:.2f};")
    print(f"  even that leaves at least "
          f"{100 * worst_frac:.0f}% of families with the series trend SMALLER "
          f"than the\n  conformer-to-conformer scatter it is buried in.")
    if float(out["median_snr"].max()) < 1.0:
        print("\n  ==> CONFIRMED. Every energy feature carries the lanthanide "
              "trend with less\n      signal than noise inside a ligand family. "
              "A tree model only compares\n      values, so it substitutes this "
              "proxy for the exact ionic-radius lookup\n      and the "
              "selectivity metric collapses -- the same failure the geometric\n"
              "      blocks showed, now replicated for energetics with numbers.")
        # The reduction needed is per-feature: a ratio built from one feature's
        # scatter and another's step is not a ratio of anything.  Take the
        # best-behaved feature, which is the easiest case and therefore the
        # weakest demand the remedy has to meet.
        need = 1.0 / float(best["median_snr"])
        print(f"\n  ==> The remedy is a smaller denominator, not a bigger "
              f"feature set: the\n      scatter is conformer noise, so an "
              f"energy-weighted conformer ENSEMBLE\n      would have to cut it "
              f"by at least {need:.1f}x -- on {best['feature']}, the most\n"
              f"      favourable feature -- before the series trend dominates. "
              f"Averaging n\n      independent conformers cuts scatter by "
              f"sqrt(n), so that is roughly {need ** 2:.0f}\n      conformers "
              f"per complex if they were independent, and more if they are not."
              f"\n      That is a quantitative target for the metadynamics "
              f"pilot, which until\n      now had only a qualitative "
              f"motivation.")
    else:
        print("\n  ==> NOT confirmed at this threshold; the substitution "
              "hypothesis does not\n      explain the collapse and another "
              "mechanism has to be found.")
    print(f"\n[diag] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
