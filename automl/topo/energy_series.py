#!/usr/bin/env python3
"""D2: does the computed energy series shape predict the measured one?

The contraction campaign proved g-xTB fixes the *geometry* of the lanthanide
series and that this does not move the score, because 96% of the structural
change is metal-shared.  Energies were never given the same chance.  This
module asks the energy-side question end to end:

  For each ligand family in the metal-substitution series
  (automl/artifacts/gxtb_series/, 71 ligands x 15 Ln x {GFN2, g-xTB}),
  compute the adjacent-pair energy step  dE_f(a,b) = E_f(a) - E_f(b)
  from the optimised structures.  The atomic-reference part of dE is
  family-independent and cancels on subtracting the cross-family mean
  profile; what remains is the LIGAND-SPECIFIC energy step.

  On the label side, the measured separation dy_x(a,b) for extractant x
  decomposes the same way (series_shape.py measured the shared profile:
  LOEO R^2 +0.066, split-half r 0.75).  The residual after removing the
  shared profile is the ligand-specific measured selectivity.

  The test: corr( ligand-specific dE , ligand-specific dy ) over matched
  (ligand, adjacent pair) cells, per Hamiltonian.  If g-xTB's f-in-valence
  energies carry ligand-specific selectivity, this correlation is negative
  (more favourable relative binding of the lighter metal -> larger dy) and
  stronger than GFN2's.

Writes ``automl/reports/energy_series.csv`` (per (ligand, pair) matched cells)
and ``energy_series_summary.json``.

Usage:  PYTHONPATH=$PWD python3 -m automl.topo.energy_series
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SERIES_DIR = REPO / "automl/artifacts/gxtb_series"
MATRIX = REPO / "automl/artifacts/matrix/matrix.parquet"
OUT_CSV = REPO / "automl/reports/energy_series.csv"
OUT_JSON = REPO / "automl/reports/energy_series_summary.json"

EH_TO_EV = 27.211386245988

LN = ["", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
      "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
IDX = {s: i for i, s in enumerate(LN) if s}


def load_records() -> pd.DataFrame:
    rows = []
    for f in ("cf_shard0.json", "cf_shard1.json", "opt_water.json",
              "opt_gas.json"):
        p = SERIES_DIR / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d.get("records", []):
            if not r.get("ok") or r.get("energy_eh") is None:
                continue
            rows.append({"family": r["family"], "metal": r["metal"],
                         "arm": r["arm"], "source": f,
                         "energy_eh": float(r["energy_eh"])})
    df = pd.DataFrame(rows)
    # a (family, metal, arm) may appear in several sources; average
    df = (df.groupby(["family", "metal", "arm"], as_index=False)
          ["energy_eh"].mean())
    df["ligand"] = df["family"].str.split("||", regex=False).str[0]
    df["mi"] = df["metal"].map(IDX)
    return df


def energy_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Adjacent energy steps per (family, arm), skipping gaps > 1 index."""
    rows = []
    for (fam, arm), g in df.groupby(["family", "arm"]):
        g = g.sort_values("mi")
        mi = g["mi"].to_numpy(int)
        e = g["energy_eh"].to_numpy(float)
        lig = g["ligand"].iloc[0]
        for a in range(len(mi) - 1):
            if mi[a + 1] - mi[a] == 1:
                rows.append({"family": fam, "ligand": lig, "arm": arm,
                             "l_lo": mi[a],
                             "pair": f"{LN[mi[a]]}-{LN[mi[a] + 1]}",
                             "dE_ev": (e[a] - e[a + 1]) * EH_TO_EV})
    return pd.DataFrame(rows)


def label_residuals() -> pd.DataFrame:
    """Ligand-specific measured dy residual per (extractant, adjacent pair)."""
    cols = ["LIGAND_SMILES", "composition_key", "lanthanide_index",
            "log_D", "has_3d", "geometry_ok"]
    m = pd.read_parquet(MATRIX, columns=cols)
    m = m[m["geometry_ok"].astype(bool) & m["has_3d"]]
    cells = (m.groupby(["LIGAND_SMILES", "composition_key",
                        "lanthanide_index"], as_index=False)["log_D"].mean())
    rows = []
    for (ex, ck), blk in cells.groupby(["LIGAND_SMILES",
                                        "composition_key"]):
        blk = blk.sort_values("lanthanide_index")
        idx = blk["lanthanide_index"].to_numpy(int)
        yv = blk["log_D"].to_numpy(float)
        for a in range(len(idx) - 1):
            if idx[a + 1] - idx[a] == 1:
                rows.append({"ligand": ex, "l_lo": int(idx[a]),
                             "dy": float(yv[a] - yv[a + 1])})
    pairs = pd.DataFrame(rows)
    # per (extractant, position): mean over blocks; then remove the shared
    # cross-extractant profile
    cell = (pairs.groupby(["ligand", "l_lo"], as_index=False)
            ["dy"].agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "dy_mean", "count": "n_blocks"}))
    prof = cell.groupby("l_lo")["dy_mean"].transform("mean")
    cell["dy_resid"] = cell["dy_mean"] - prof
    return cell


def main() -> int:
    rec = load_records()
    st = energy_steps(rec)
    n_fam = st.groupby("arm")["family"].nunique().to_dict()
    print(f"energy records: {len(rec)} · families with steps: {n_fam}")

    # ligand-specific energy step = step minus the cross-family mean profile
    st["dE_shared"] = st.groupby(["arm", "l_lo"])["dE_ev"].transform("mean")
    st["dE_resid"] = st["dE_ev"] - st["dE_shared"]
    # families with both arms measured at this position, for a fair contrast
    lab = label_residuals()

    # join: series ligand SMILES == matrix LIGAND_SMILES
    merged = st.merge(lab, on=["ligand", "l_lo"], how="inner")
    print(f"matched (ligand, pair) cells: "
          f"{merged.groupby('arm').size().to_dict()} · "
          f"ligands: {merged.groupby('arm')['ligand'].nunique().to_dict()}")

    out = {}
    from scipy import stats
    for arm, g in merged.groupby("arm"):
        r_resid = float(np.corrcoef(g["dE_resid"], g["dy_resid"])[0, 1])
        rho = float(stats.spearmanr(g["dE_resid"], g["dy_resid"]).statistic)
        # weighted by number of blocks behind each label cell
        wr = None
        try:
            w = g["n_blocks"].to_numpy(float)
            xw = g["dE_resid"].to_numpy(float)
            yw = g["dy_resid"].to_numpy(float)
            xm, ym = np.average(xw, weights=w), np.average(yw, weights=w)
            cov = np.average((xw - xm) * (yw - ym), weights=w)
            wr = float(cov / np.sqrt(np.average((xw - xm) ** 2, weights=w)
                                     * np.average((yw - ym) ** 2, weights=w)))
        except Exception:
            pass
        n = len(g)
        t = r_resid * np.sqrt(max(n - 2, 1) / max(1e-12, 1 - r_resid ** 2))
        p = float(2 * stats.t.sf(abs(t), n - 2))
        out[arm] = {"n_cells": n, "n_ligands": int(g["ligand"].nunique()),
                    "pearson_resid": r_resid, "spearman_resid": rho,
                    "pearson_weighted": wr, "p_pearson": p,
                    "shared_profile_r": float(np.corrcoef(
                        g["dE_shared"], g["dy_mean"] - g["dy_resid"])[0, 1])}
        # robustness: cells whose label rests on >= 2 independent blocks
        g2 = g[g["n_blocks"] >= 2]
        if len(g2) > 10:
            sp2 = stats.spearmanr(g2["dE_resid"], g2["dy_resid"])
            out[arm]["spearman_nblocks2"] = float(sp2.statistic)
            out[arm]["p_spearman_nblocks2"] = float(sp2.pvalue)
            out[arm]["n_cells_nblocks2"] = int(len(g2))
        sp = stats.spearmanr(g["dE_resid"], g["dy_resid"])
        out[arm]["p_spearman"] = float(sp.pvalue)
        print(f"[{arm}] n={n} cells / {out[arm]['n_ligands']} ligands · "
              f"ligand-specific corr(dE, dy): pearson {r_resid:+.3f} "
              f"(p={p:.3g}), spearman {rho:+.3f}, weighted {wr:+.3f}" if wr is not None else "")

    merged.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT_CSV} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
