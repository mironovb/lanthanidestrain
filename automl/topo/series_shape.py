#!/usr/bin/env python3
"""Does the measured adjacent-separation series have a ligand-independent shape?

Two questions, both asked of the LABELS only (no model anywhere):

1. **The pair-identity floor.**  ``automl/metal_physics.py`` asserts (uncited)
   that a leave-extractants-out baseline predicting the adjacent-pair
   separation ``dy`` from PAIR IDENTITY ALONE reaches R^2 ~ 0.058.  Compute it.
   The predictor is a 12-value lookup (mean training ``dy`` per adjacent pair
   position; Pm's absence removes indices (4,5) and (5,6)), evaluated
   leave-one-extractant-out on the legacy 905-pair population.

2. **The shape itself.**  Is the per-position profile a smooth radius ramp, or
   does it carry structure (tetrad / half-shell)?  Tested by (a) comparing the
   lookup against a one-parameter radius-proportional model
   ``dy = k * d_radius``, (b) split-half reliability of the 12-vector across
   random extractant halves, (c) an F-test of positions-beyond-radius on the
   pooled per-pair data.

Population: legacy ``ok_only`` rows (iteration population; the frozen fresh
pairs are not touched here).  Cell-averaged exactly as the metric averages.

Writes ``automl/reports/series_shape.csv`` (per-position profile) and
``automl/reports/series_shape_summary.json``; prints the verdict.

Usage:  module load anaconda/Python-ML-2025a
        PYTHONPATH=$PWD python3 -m automl.topo.series_shape
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MATRIX = REPO / "automl/artifacts/matrix/matrix.parquet"
OUT_CSV = REPO / "automl/reports/series_shape.csv"
OUT_JSON = REPO / "automl/reports/series_shape_summary.json"

COLS = ["metal_symbol", "lanthanide_index", "has_3d", "geometry_ok",
        "composition_key", "extractant_group", "log_D", "Ionic Radius_metal"]

# lanthanide_index is 1-based: La=1 ... Lu=15, Pm=5 absent from the data
LN = ["", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
      "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]


def pair_table() -> pd.DataFrame:
    """One row per adjacent pair: (extractant, block, l_lo, dy, d_radius)."""
    m = pd.read_parquet(MATRIX, columns=COLS)
    m = m[m["geometry_ok"].astype(bool) & m["has_3d"]]
    radius = (m.drop_duplicates("lanthanide_index")
                .set_index("lanthanide_index")["Ionic Radius_metal"].to_dict())
    cells = (m.groupby(["extractant_group", "composition_key",
                        "lanthanide_index"], as_index=False)["log_D"].mean())
    rows = []
    for (ex, ck), blk in cells.groupby(["extractant_group", "composition_key"]):
        blk = blk.sort_values("lanthanide_index")
        idx = blk["lanthanide_index"].to_numpy()
        yv = blk["log_D"].to_numpy()
        for a in range(len(idx) - 1):
            if idx[a + 1] - idx[a] == 1:
                lo = int(idx[a])
                rows.append({
                    "extractant_group": ex, "composition_key": ck,
                    "l_lo": lo, "pair": f"{LN[lo]}-{LN[lo + 1]}",
                    # light minus heavy: log SF of the lighter over the heavier
                    "dy": float(yv[a] - yv[a + 1]),
                    "d_radius": float(radius[lo] - radius[lo + 1]),
                })
    return pd.DataFrame(rows)


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def loeo_scores(pairs: pd.DataFrame) -> dict[str, float]:
    """Leave-one-extractant-out R^2 for the label-only predictors."""
    preds: dict[str, list[float]] = {"lookup": [], "constant": [], "radius": []}
    truth: list[float] = []
    for ex in pairs["extractant_group"].unique():
        tr = pairs[pairs["extractant_group"] != ex]
        te = pairs[pairs["extractant_group"] == ex]
        prof = tr.groupby("l_lo")["dy"].mean()
        const = tr["dy"].mean()
        # one-parameter radius model fitted on train: dy = k * d_radius
        k = (tr["dy"] * tr["d_radius"]).sum() / (tr["d_radius"] ** 2).sum()
        truth.extend(te["dy"].tolist())
        preds["lookup"].extend(te["l_lo"].map(prof).fillna(const).tolist())
        preds["constant"].extend([const] * len(te))
        preds["radius"].extend((k * te["d_radius"]).tolist())
    y = np.asarray(truth)
    return {f"loeo_r2_{k}": _r2(y, np.asarray(v)) for k, v in preds.items()}


def split_half(pairs: pd.DataFrame, n_draws: int = 500, seed: int = 7
               ) -> dict[str, float]:
    """Reliability of the 12-position profile across random extractant halves."""
    rng = np.random.default_rng(seed)
    exs = pairs["extractant_group"].unique()
    rs = []
    for _ in range(n_draws):
        half = set(rng.choice(exs, size=len(exs) // 2, replace=False))
        a = pairs[pairs["extractant_group"].isin(half)].groupby("l_lo")["dy"].mean()
        b = pairs[~pairs["extractant_group"].isin(half)].groupby("l_lo")["dy"].mean()
        common = a.index.intersection(b.index)
        if len(common) >= 8 and a[common].std() > 0 and b[common].std() > 0:
            rs.append(float(np.corrcoef(a[common], b[common])[0, 1]))
    rs = np.asarray(rs)
    return {"splithalf_r_mean": float(rs.mean()),
            "splithalf_r_lo": float(np.quantile(rs, 0.05)),
            "splithalf_r_hi": float(np.quantile(rs, 0.95)),
            "splithalf_frac_positive": float(np.mean(rs > 0)),
            "splithalf_draws": int(len(rs))}


def shape_beyond_radius(pairs: pd.DataFrame) -> dict[str, float]:
    """Pooled F-test: do position dummies explain dy beyond d_radius?"""
    y = pairs["dy"].to_numpy()
    x_r = pairs["d_radius"].to_numpy()
    k = (y * x_r).sum() / (x_r ** 2).sum()
    rss_radius = float(np.sum((y - k * x_r) ** 2))
    prof = pairs.groupby("l_lo")["dy"].transform("mean").to_numpy()
    rss_lookup = float(np.sum((y - prof) ** 2))
    n = len(y)
    p_extra = pairs["l_lo"].nunique() - 1
    f = ((rss_radius - rss_lookup) / p_extra) / (rss_lookup / (n - p_extra - 1))
    from scipy import stats
    p = float(stats.f.sf(f, p_extra, n - p_extra - 1))
    return {"rss_radius_only": rss_radius, "rss_lookup": rss_lookup,
            "F_shape_beyond_radius": float(f), "p_shape_beyond_radius": p}


def main() -> int:
    pairs = pair_table()
    assert len(pairs) == 905, f"expected the legacy 905 pairs, got {len(pairs)}"

    profile = (pairs.groupby(["l_lo", "pair"])
               .agg(n=("dy", "size"), dy_mean=("dy", "mean"),
                    dy_sd=("dy", "std"), d_radius=("d_radius", "first"))
               .reset_index())
    profile["dy_sem"] = profile["dy_sd"] / np.sqrt(profile["n"])

    out = {"n_pairs": int(len(pairs)),
           "n_extractants": int(pairs["extractant_group"].nunique()),
           "n_positions": int(pairs["l_lo"].nunique()),
           "sd_dy": float(pairs["dy"].std())}
    out.update(loeo_scores(pairs))
    out.update(split_half(pairs))
    out.update(shape_beyond_radius(pairs))

    sign_changes = int(np.sum(np.diff(np.sign(profile["dy_mean"])) != 0))
    out["profile_sign_changes"] = sign_changes

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"{out['n_pairs']} pairs · {out['n_extractants']} extractants · "
          f"{out['n_positions']} positions · sd(dy)={out['sd_dy']:.4f}")
    print(f"LOEO R^2  lookup={out['loeo_r2_lookup']:+.4f}  "
          f"radius-only={out['loeo_r2_radius']:+.4f}  "
          f"constant={out['loeo_r2_constant']:+.4f}")
    print(f"split-half profile r = {out['splithalf_r_mean']:+.3f} "
          f"[{out['splithalf_r_lo']:+.3f}, {out['splithalf_r_hi']:+.3f}]  "
          f"({out['splithalf_frac_positive']:.0%} draws positive)")
    print(f"shape beyond radius: F={out['F_shape_beyond_radius']:.2f}, "
          f"p={out['p_shape_beyond_radius']:.2e}; "
          f"profile sign changes: {sign_changes}")
    print("per-position profile:")
    for _, r in profile.iterrows():
        bar = "#" * int(abs(r.dy_mean) * 40)
        print(f"  {r['pair']:>6} n={int(r.n):3d}  dy={r.dy_mean:+.3f}"
              f"+-{r.dy_sem:.3f}  {bar}")
    print("wrote", OUT_CSV, "and", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
