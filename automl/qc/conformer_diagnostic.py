#!/usr/bin/env python3
"""Would consistent global minimisation remove the conformer contamination?

The chain of reasoning this closes
-----------------------------------
1. ``energy_test`` -- adding reference xTB energetics destroys the adjacent-pair
   metric (+0.1422 -> -0.0350 binned, -0.2993 strict and significant).
2. ``energy_diagnostic`` -- because within a ligand family every energy feature
   carries the lanthanide trend at SNR ~0.25: a ~0.20 eV per-step signal buried
   in ~0.73 eV of scatter, against an incumbent (ionic radius) that is a lookup
   table with zero scatter.
3. The scatter was *assumed* to be thermal, so a Boltzmann-weighted conformer
   ensemble should average it away.  **The metadynamics smoke falsified that**:
   8 distinct conformers per complex but an effective ensemble size of 1.0-1.8,
   because conformer energy gaps are 0.8-1.9 eV against kT = 0.026 eV.  A
   Boltzmann average of this ensemble simply *is* its minimum.
4. So the surviving hypothesis is different: the scatter is not thermal breadth
   but **arbitrariness**.  Each dataset geometry is one Architector/GFN2 local
   minimum, and *which* minimum differs from complex to complex for reasons
   unrelated to chemistry.

This module tests (4), and it can be tested without training anything.

The measurement
---------------
For each complex the pilot relaxes the **shipped** geometry alongside the
metadynamics snapshots, so

    gap = E(shipped, relaxed) - E(global minimum found)

is how far the dataset's structure sits above the best one a common search
finds.  If every complex sat a similar distance above its own minimum, the gap
would be near-constant within a family and would cancel in the family-relative
features -- harmless.  The hypothesis says it does not.

So the quantity that matters is **SD(gap) within a ligand family**, compared
against the 0.73 eV scatter ``energy_diagnostic`` measured in `e_int`. If it
accounts for a large share of that, a consistent global minimisation is the
remedy and the campaign is worth running. If it is small, the scatter comes from
somewhere else and conformer search will not fix it -- which would be a null
worth having, because it is the third and last of the levers the reports keep
naming.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "automl/artifacts/mtd_ensemble"
OUT = REPO / "automl/reports/conformer_diagnostic.csv"

# What energy_diagnostic measured on the shipped single conformers.
EINT_SCATTER_EV = 0.7313          # median within-family residual SD of e_int
EINT_STEP_EV = 0.1695             # median per-lanthanide-step trend
KT_298 = 0.025693
# Conformational energy differences in these complexes span ~0.1-3 eV.
# Anything past this is a failed calculation, not a conformer.
MAX_PHYSICAL_GAP_EV = 20.0


def load() -> pd.DataFrame:
    root = ART / "per_geometry"
    if not root.exists():
        raise SystemExit(f"{root} missing; run automl.qc.mtd_ensemble --pilot")
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]
    df = pd.DataFrame(rows)
    parts = df["geometry_key"].astype(str).str.split("|", n=2, expand=True)
    df["fam"] = parts[1].fillna("") + "|" + parts[2].fillna("")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-members", type=int, default=4)
    args = ap.parse_args()

    df = load()
    ok = df[df["status"] == "ok"].copy() if "status" in df else df.copy()
    print(f"[conf] {len(df)} complexes, {len(ok)} ok, "
          f"{ok['fam'].nunique()} ligand families")

    for c in ("n_unique", "n_effective", "e_spread_ev",
              "gap_shipped_above_min_ev"):
        if c in ok.columns:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")

    # Discard diverged calculations before they reach any statistic.
    #
    # A "conformer" 1340 eV above the minimum is not a conformer; it is an SCF
    # divergence or a dissociated structure.  Conformational energy differences
    # in these complexes are ~0.1-3 eV, so anything past 20 eV is a failure of
    # the calculation rather than a property of the molecule.  Reported, not
    # silently dropped -- a filter that removes data without saying how much is
    # how a null gets manufactured.
    n_before = len(ok)
    diverged = ok[ok["gap_shipped_above_min_ev"] > MAX_PHYSICAL_GAP_EV]
    ok = ok[ok["gap_shipped_above_min_ev"] <= MAX_PHYSICAL_GAP_EV]
    if len(diverged):
        print(f"\n[conf] discarded {len(diverged)} of {n_before} complexes with "
              f"a gap above {MAX_PHYSICAL_GAP_EV:.0f} eV (diverged, not "
              f"conformational):")
        for _, r in diverged.iterrows():
            print(f"        {str(r['geometry_key'])[:52]:54s} "
                  f"n_unique={r['n_unique']:.0f} "
                  f"gap={r['gap_shipped_above_min_ev']:.1f} eV")

    if "gap_shipped_above_min_ev" not in ok.columns:
        raise SystemExit("the pilot did not record gap_shipped_above_min_ev; "
                         "re-run automl.qc.mtd_ensemble with the current code")

    print("\n=== the search itself ===")
    print(f"  unique conformers per complex : median "
          f"{ok['n_unique'].median():.1f}  "
          f"({(ok['n_unique'] <= 1).mean():.0%} returned only one)")
    print(f"  effective ensemble size       : median "
          f"{ok['n_effective'].median():.2f}   "
          f"(kT = {KT_298:.4f} eV; spreads median "
          f"{ok['e_spread_ev'].median():.2f} eV)")
    print("  -> Boltzmann weighting cannot average this: the weight collapses "
          "onto the minimum.")

    print("\n=== how far the shipped geometry sits above the global minimum ===")
    g = ok["gap_shipped_above_min_ev"].dropna()
    print(f"  gap: median {g.median():.3f} eV, IQR "
          f"[{g.quantile(.25):.3f}, {g.quantile(.75):.3f}], max {g.max():.3f} eV")
    print(f"  {(g > 0.01).mean():.0%} of shipped geometries are NOT the global "
          f"minimum found by the search")

    rows = []
    print(f"\n=== within-family SD of that gap "
          f"(families with >= {args.min_members} members) ===")
    print(f"  {'family':46s} {'n':>3s} {'SD(gap) eV':>11s} {'share of 0.73 eV':>17s}")
    for fam, sub in ok.groupby("fam"):
        v = sub["gap_shipped_above_min_ev"].dropna()
        if len(v) < args.min_members:
            continue
        sd = float(v.std(ddof=1))
        share = sd / EINT_SCATTER_EV
        print(f"  {fam[:46]:46s} {len(v):3d} {sd:11.3f} {share:16.0%}")
        rows.append({"family": fam, "n": len(v), "sd_gap_ev": sd,
                     "share_of_eint_scatter": share,
                     "median_gap_ev": float(v.median())})

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    if out.empty:
        print("  no family had enough members")
        return 1

    med_sd = float(out["sd_gap_ev"].median())
    share = med_sd / EINT_SCATTER_EV
    print("\n=== verdict ===")
    print(f"  median within-family SD of the shipped-vs-minimum gap = "
          f"{med_sd:.3f} eV")
    print(f"  the scatter that has to be removed (energy_diagnostic) = "
          f"{EINT_SCATTER_EV:.3f} eV")
    print(f"  the per-lanthanide signal it has to be removed FROM   = "
          f"{EINT_STEP_EV:.3f} eV")
    print(f"  => arbitrary local minimisation accounts for "
          f"{share:.0%} of the scatter")
    # Removing a fraction f of the variance leaves sqrt(1-f) of the SD.
    if share >= 1.0:
        left = 0.0
    else:
        left = float(np.sqrt(max(EINT_SCATTER_EV ** 2 - med_sd ** 2, 0.0)))
    new_snr = EINT_STEP_EV / left if left > 0 else float("inf")
    print(f"  => removing it entirely would leave {left:.3f} eV of scatter, "
          f"i.e. SNR {new_snr:.2f}")
    if new_snr >= 1.0:
        print("\n  ==> WORTH RUNNING. A consistent global minimisation would "
              "take the energy\n      features above SNR 1 inside a ligand "
              "family, which is the threshold at\n      which they stop being "
              "a noisier substitute for the ionic radius.")
    else:
        print("\n  ==> NOT WORTH RUNNING. Even removing the arbitrary-minimum "
              "component entirely\n      leaves the trend below its own "
              "scatter, so a full conformer campaign\n      cannot rescue these "
              "features. The remaining scatter is something else --\n      "
              "most likely genuine sensitivity of GFN2 energies to the ligand "
              "conformation,\n      which no amount of searching removes "
              "because it is not an artefact.")
    print(f"\n[conf] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
