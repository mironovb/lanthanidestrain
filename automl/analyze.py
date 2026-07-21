#!/usr/bin/env python3
"""Read every sweep result and turn it into ranked tables + a markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from automl.experiment import load_results

REPO = Path(__file__).resolve().parents[1]
SWEEP_DIR = REPO / "automl/artifacts/sweeps"
REPORT_DIR = REPO / "automl/reports"

HEADLINE = ["r2_overall", "r2_between", "r2_within", "r2_within_composition",
            "sel_spearman_mean", "sel_logSF_r2", "sel_sign_accuracy", "rmse", "mae"]

BLOCK_LEGEND = {
    "g1": "G1 first shell (raw M-L distances, donor composition)",
    "g2": "G2 contraction-corrected shell (d - r_ionic: cavity fit)",
    "g3": "G3 donor polyhedron shape (CShM, hull, anisotropy)",
    "g4": "G4 steric burial (%V_bur, solid angle, radial counts)",
    "g5": "G5 xTB electronics (charges, charge transfer, dipole, forces)",
    "g6": "G6 metal-centred RDF fingerprint",
    "g7": "G7 whole-complex shape + lipophilic surface (SASA)",
    "g8": "G8 chelate topology (bite angles, ring sizes, denticity)",
    "g9": "G9 persistent homology summaries",
    "g10": "G10 within-ligand relative descriptors (series-relative)",
    "g11": "G11 GFN2-xTB persistence images (flattened)",
    "p3d_phys": "shipped complex_physical block",
    "p3d_poly": "shipped ordered polyhedron block",
}


def collect(sweep_dir: Path = SWEEP_DIR) -> pd.DataFrame:
    frames = []
    for sub in sorted(Path(sweep_dir).glob("*")):
        if not sub.is_dir():
            continue
        d = load_results(sub)
        if len(d):
            d["sweep"] = sub.name
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fmt(df: pd.DataFrame, cols: Iterable[str], nd: int = 4) -> str:
    cols = [c for c in cols if c in df.columns]
    view = df[cols].copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(view[c]):
            view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    return view.to_markdown(index=False)


def ablation_table(res: pd.DataFrame, row_filter: str = "has3d",
                   model: str = "lgbm") -> pd.DataFrame:
    d = res[(res.get("tag") == "ablation") & (res["row_filter"] == row_filter)
            & (res["model"] == model)].copy()
    if d.empty:
        return d
    d = d.sort_values("r2_overall", ascending=False)
    base = d[d["preset"] == "baseline_2d"]
    if len(base):
        b = base.iloc[0]
        for m in ("r2_overall", "r2_between", "r2_within",
                  "r2_within_composition", "sel_spearman_mean", "sel_logSF_r2"):
            if m in d.columns:
                d[f"d_{m}"] = d[m] - b[m]
    return d


def build_report(res: pd.DataFrame, out_path: Path) -> str:
    lines: list[str] = []
    A = lines.append
    A("# Lanthanide log D AutoML - 3D geometry ablation report")
    A("")
    A(f"_Generated from {len(res)} completed cross-validated experiments._")
    A("")
    A("## Protocol")
    A("")
    A("- Target `log_D`; split = leave-extractants-out, grouped K-fold on "
      "`extractant_group` (canonical extractant SMILES, 190 groups).")
    A("- Every reported number is computed from **out-of-fold** predictions, so "
      "no extractant is ever in train and test at the same time.")
    A("- `R2_between` = R^2 of the per-extractant mean (size weighted): can the "
      "model rank whole extractants?")
    A("- `R2_within`  = R^2 after removing each extractant's own mean: can it "
      "reproduce the spread *inside* one extractant (metal + conditions)?")
    A("- `R2_within_composition` removes extractant **and** condition set, "
      "leaving pure lanthanide-series selectivity.")
    A("- `sel_*` metrics are computed inside composition blocks with >= 3 "
      "lanthanides: Spearman of the predicted series order, R^2 of pairwise "
      "log separation factors, and the sign accuracy of those pairs.")
    A("")

    for rf in sorted(res["row_filter"].dropna().unique()):
        d = ablation_table(res, row_filter=rf)
        if d.empty:
            continue
        A(f"## Feature-block ablation (rows = `{rf}`)")
        A("")
        cols = ["preset", "n_features"] + HEADLINE + ["d_r2_overall", "d_r2_within",
                                                      "d_sel_logSF_r2"]
        A(_fmt(d, cols))
        A("")

    tags = res["tag"].astype(str)

    arch = res[tags.str.startswith("arch")]
    if len(arch):
        A("## Architecture sweep")
        A("")
        A("`anchored:<base>` = level from the flat model + shape from a residual "
          "model trained on `y - mean_extractant(y)`; `pairwise:<base>` = "
          "delta-learning on lanthanide pairs inside one condition block.")
        A("")
        sub = arch.sort_values("r2_overall", ascending=False).head(30).copy()
        sub["variant"] = sub["tag"].str.replace("arch:?", "", regex=True)
        A(_fmt(sub, ["preset", "model", "variant"] + HEADLINE))
        A("")

    cn = res[tags == "cnfree"]
    if len(cn):
        A("## Coordination-number artefact correction")
        A("")
        A("`g15` regresses the CN main effect out of every descriptor; "
          "`g12c/g13c/g14c` replace raw values by the per-family fit against the "
          "ionic radius. If the CN staircase is what breaks the series ordering, "
          "these should recover `sel_logSF_r2`.")
        A("")
        A(_fmt(cn.sort_values("r2_overall", ascending=False),
               ["preset", "model", "n_features"] + HEADLINE))
        A("")

    mdl = res[tags == "models"]
    if len(mdl):
        A("## Model family x sample weighting")
        A("")
        for preset in sorted(mdl["preset"].unique()):
            sub = mdl[mdl["preset"] == preset].sort_values("r2_overall", ascending=False)
            A(f"### preset = `{preset}`")
            A("")
            A(_fmt(sub, ["model", "weight_scheme", "n_features"] + HEADLINE))
            A("")

    opt = res[tags == "optuna"]
    if len(opt):
        A("## Hyperparameter search (best 25 trials by overall R^2)")
        A("")
        top = opt.sort_values("r2_overall", ascending=False).head(25)
        A(_fmt(top, ["preset", "model", "weight_scheme"] + HEADLINE))
        A("")

    A("## Overall leaderboard (every completed experiment)")
    A("")
    lb = res.sort_values("r2_overall", ascending=False).head(30).copy()
    lb["variant"] = lb["tag"].astype(str).str.replace("arch:?", "", regex=True)
    A(_fmt(lb, ["preset", "model", "variant", "weight_scheme", "row_filter",
                "n_features"] + HEADLINE))
    A("")

    A("## Block legend")
    A("")
    for k, v in BLOCK_LEGEND.items():
        A(f"- `{k}`: {v}")
    A("")

    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=str(SWEEP_DIR))
    ap.add_argument("--out", default=str(REPORT_DIR / "automl_results.md"))
    ap.add_argument("--csv", default=str(REPORT_DIR / "all_results.csv"))
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    res = collect(Path(args.sweep_dir))
    if res.empty:
        print("no results yet")
        return 0
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.csv, index=False)
    build_report(res, Path(args.out))
    print(f"{len(res)} experiments -> {args.out}")
    show = ["sweep", "preset", "model", "weight_scheme", "row_filter",
            "n_features", "r2_overall", "r2_between", "r2_within",
            "r2_within_composition", "sel_logSF_r2"]
    show = [c for c in show if c in res.columns]
    print(res.sort_values("r2_overall", ascending=False)[show]
          .head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
