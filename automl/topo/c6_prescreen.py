"""Does a candidate per-metal descriptor carry signal in the metric's direction?

The gate sweep2's A1 cell did not have.  A1 added 119 geometry columns and cost
-0.3167 on the adjacent-pair metric; ``within_block_signal`` later showed why --
their within-block differences correlated with ``dy`` at a median |r| of 0.0495,
against the ``cond`` block's 0.0804 and best-column 0.357.  The columns were
free to be fitted and useless once fitted.

This runs the same measurement on the ``mphys__`` block BEFORE any GPU time is
spent, and on the incumbent ``metal`` columns as the reference bar.  Costs
seconds, uses no GPU, and takes no confirmatory look: it correlates features
against dy without fitting anything, so there is no model to overfit and no
selection to pay for.

    python3 -m automl.topo.c6_prescreen
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import automl.evaluation as ev
from automl.metal_physics import PREFIX, attach, series_shape

REPORTS = Path(__file__).resolve().parents[1] / "reports"
OUT = REPORTS / "c6_prescreen.csv"


def metric_pairs(d: pd.DataFrame):
    """Row groups behind each of the metric's adjacent pairs.

    Reconstructed the way ``adjacent_pair_arrays`` does -- group by
    composition_key, average per lanthanide, keep |d index| == 1 -- and then
    CHECKED against the evaluator itself.  A quiet mismatch here would look
    like a finding.
    """
    d = d.reset_index(drop=True)
    d["_r"] = np.arange(len(d))
    rows = []
    for _, sub in d.groupby("composition_key"):
        per = {int(k): (v["log_D"].mean(), v["_r"].to_numpy())
               for k, v in sub.groupby("lanthanide_index")}
        ks = sorted(per)
        for i in range(len(ks) - 1):
            if ks[i + 1] - ks[i] == 1:
                rows.append((per[ks[i]][1], per[ks[i + 1]][1],
                             per[ks[i]][0] - per[ks[i + 1]][0]))
    dy = np.array([r[2] for r in rows])
    ref, _ = ev.adjacent_pair_arrays(d["log_D"].to_numpy(float),
                                     d["log_D"].to_numpy(float),
                                     d["composition_key"].to_numpy(),
                                     d["lanthanide_index"].to_numpy(float))
    if len(dy) != len(ref) or not np.isclose(np.sort(np.abs(dy)),
                                             np.sort(np.abs(ref)),
                                             atol=1e-8).all():
        raise SystemExit(f"reconstruction mismatch: {len(dy)} pairs vs the "
                         f"evaluator's {len(ref)} -- refusing to report")
    return rows, dy


def score_columns(rows, dy, X: np.ndarray, names: list[str]) -> pd.DataFrame:
    out = []
    for j, nm in enumerate(names):
        dv = np.array([np.nanmean(X[ra, j]) - np.nanmean(X[rb, j])
                       for ra, rb, _ in rows])
        ok = np.isfinite(dv)
        if ok.sum() < 30 or np.nanstd(dv[ok]) < 1e-12:
            out.append({"column": nm, "n_pairs": int(ok.sum()),
                        "sd_within_block": float(np.nanstd(dv[ok]))
                        if ok.sum() else np.nan,
                        "abs_corr_with_dy": np.nan,
                        "note": "block-constant or too sparse"})
            continue
        r = float(np.corrcoef(dv[ok], dy[ok])[0, 1])
        out.append({"column": nm, "n_pairs": int(ok.sum()),
                    "sd_within_block": float(np.nanstd(dv[ok])),
                    "abs_corr_with_dy": abs(r), "signed_corr": r, "note": ""})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(OUT))
    args = ap.parse_args()

    from automl.matrix_cache import load_cache
    df, _, _ = load_cache()
    d = df.dropna(subset=["log_D"]).copy()
    if "geometry_feature_build_id" in d:
        d = d[d["geometry_feature_build_id"].notna()]
    d = d.reset_index(drop=True)

    new_cols = attach(d)
    incumbent = [c for c in ("Atomic Number_metal", "lanthanide_index",
                             "Ionic Radius_metal") if c in d.columns]
    # The bar the geometry columns failed to clear, and the one they were
    # compared against.
    bar = [c for c in d.columns
           if c in ("cond__extractant_concentration_M",
                    "cond__acid_concentration_M")]
    names = incumbent + bar + new_cols

    rows, dy = metric_pairs(d)
    print(f"[prescreen] reconstructed {len(rows)} metric pairs, "
          f"mean dy = {dy.mean():+.4f}")
    X = d[names].to_numpy(dtype=float)
    res = score_columns(rows, dy, X, names)
    res["family"] = np.where(res["column"].str.startswith(PREFIX), "mphys",
                             np.where(res["column"].str.startswith("cond__"),
                                      "cond (reference bar)", "incumbent"))
    res = res.sort_values("abs_corr_with_dy", ascending=False,
                          na_position="last")
    pd.set_option("display.width", 160)
    print(res[["family", "column", "n_pairs", "sd_within_block",
               "abs_corr_with_dy"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    m = res[res["family"] == "mphys"]["abs_corr_with_dy"].dropna()
    inc = res[res["family"] == "incumbent"]["abs_corr_with_dy"].dropna()
    print(f"\n[prescreen] mphys median |corr| = {m.median():.4f} "
          f"(max {m.max():.4f}, n={len(m)})")
    print(f"[prescreen] incumbent metal columns median |corr| = "
          f"{inc.median():.4f} (max {inc.max():.4f})")
    print("[prescreen] A1's 119 geometry columns, which cost -0.3167: "
          "median 0.0495, max 0.1827")
    print("\n[prescreen] non-monotone across the series "
          "(a monotone coordinate cannot bend where mean dy bends):")
    print(series_shape().sort_values(ascending=False).head(6)
          .to_string(float_format=lambda v: f"{v:.3f}"))

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.csv, index=False)
    print(f"\n[prescreen] wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
