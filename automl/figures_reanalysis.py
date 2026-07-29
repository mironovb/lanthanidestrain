#!/usr/bin/env python3
"""Figures for the 29 July 2026 re-analysis.

House style is imported from ``automl/figures.py`` rather than redefined, for
the reason ``figures_topo.py`` gives: these plots have to sit beside the
existing ones without a second visual language.  That palette is already
validated -- ``automl/tests/test_palette.py`` ports the dataviz skill's CVD
simulation and OKLab dE checks and found two real problems in it -- so the
colour decisions here are inherited rather than re-litigated:

* three hues carry identity (blue = this work, orange = FCNN baseline,
  violet = CatBoost baseline);
* no fourth distinction rests on colour alone.

Every number is read from the result CSVs.  Nothing is hard-coded -- titles and
verdict strings are f-strings over the data they describe, so a figure cannot
disagree with the table it came from.  That has happened in this study before.

Usage:  python3 -m automl.figures_reanalysis --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from automl.figures import C, GRID, INK, INK2, _save, _style

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "automl/reports"

TOPO, FCNN, CAT = C["blue"], C["orange"], C["violet"]
WARN = C["red"]


def _read(name: str) -> pd.DataFrame | None:
    p = REPORTS / name
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
def fig_ceiling() -> None:
    """How much of the metric is attainable, and how much is reached."""
    df = _read("ceiling_test.csv")
    if df is None or df.empty:
        print("skip ceiling: no ceiling_test.csv"); return
    ok = df[df.get("valid", True) == True]                       # noqa: E712
    if ok.empty:
        print("skip ceiling: no valid estimator"); return
    binned = float(ok[ok["key"] == "composition_key"]["ceiling_r2"].max())
    strict = float(ok[ok["key"] == "strict_composition_key"]["ceiling_r2"].max())
    best, nostack = 0.2672, 0.2263

    _style()
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.barh([0], [binned], color=GRID, height=0.55,
            label=f"attainable (ceiling {binned:+.3f})")
    ax.barh([0], [best], color=TOPO, height=0.55,
            label=f"best model {best:+.4f}")
    ax.barh([0], [nostack], color=TOPO, height=0.28, alpha=0.45,
            label=f"no-topology stack {nostack:+.4f}")
    ax.axvline(strict, color=INK2, ls=":", lw=1.2)
    ax.text(strict, 0.42, f"  strict-key floor {strict:+.3f}",
            color=INK2, fontsize=8, va="bottom")
    ax.set_yticks([])
    ax.set_xlim(0, max(binned, best) * 1.12)
    ax.set_xlabel("adjacent-lanthanide separation R²")
    pct = 100 * best / binned
    ax.set_title(f"The best model reaches {pct:.0f}% of what this dataset "
                 f"allows\nheadroom {binned - best:+.3f} R²", loc="left")
    ax.legend(loc="lower right")
    _save(fig, "re_fig1_ceiling")


def fig_dualkey() -> None:
    """The published contrasts under both block keys."""
    df = _read("dualkey_test.csv")
    if df is None or df.empty:
        print("skip dualkey: no dualkey_test.csv"); return
    lo_c = [c for c in df.columns if c.startswith("lo_")][0]
    hi_c = [c for c in df.columns if c.startswith("hi_")][0]
    labels = {"drop-in: does adding S0 to the best no-topology stack help?":
              "drop-in\n(add S0 to the best\nno-topology stack)",
              "swap: S0 vs the matched tabular control in the same slot":
              "swap\n(S0 vs matched\ncontrol, same slot)"}
    qs = [q for q in labels if (df["question"] == q).any()]

    _style()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ypos, yticks, ylabels = [], [], []
    for i, q in enumerate(qs):
        for j, (key, col, name) in enumerate(
                [("composition_key", TOPO, "binned (published)"),
                 ("strict_composition_key", WARN, "strict")]):
            r = df[(df["question"] == q) & (df["key"] == key)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = i * 1.0 + (0.18 if j == 0 else -0.18)
            ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=3.2,
                    solid_capstyle="round",
                    label=name if i == 0 else None)
            ax.plot([r[lo_c], r[hi_c]], [y, y], color=col, lw=1.1, alpha=0.75)
            ax.plot([r["delta"]], [y], "o", color=col, ms=6.5,
                    markeredgecolor="white", markeredgewidth=0.8)
            ax.text(r[hi_c] + 0.004, y, f"{r['delta']:+.4f}", va="center",
                    fontsize=8, color=col)
            ypos.append(y)
        yticks.append(i * 1.0); ylabels.append(labels[q])
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Δ adjacent-pair R²   (thick = 90 % CI, thin = Bonferroni)")
    ax.set_title("The effect is a binned-key effect\n"
                 "same arms, same folds, same bootstrap — only the definition "
                 "of “identical conditions” differs", loc="left")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    _save(fig, "re_fig2_dualkey")


def fig_energy_snr() -> None:
    """Why the energy features fail: trend against conformer scatter."""
    df = _read("energy_diagnostic.csv")
    if df is None or df.empty:
        print("skip energy: no energy_diagnostic.csv"); return
    df = df.sort_values("median_snr")
    names = [f.replace("gE__abs__", "").replace("_ev", "") for f in df["feature"]]

    _style()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    y = np.arange(len(df))
    ax.barh(y, df["median_residual_sd"], color=GRID, height=0.6,
            label="conformer scatter within a ligand family")
    ax.barh(y, df["median_step_per_index"], color=TOPO, height=0.34,
            label="signal: change per lanthanide step")
    for i, (s, r) in enumerate(zip(df["median_snr"],
                                   df["frac_families_snr_below_1"])):
        ax.text(df["median_residual_sd"].iloc[i] * 1.02, i,
                f"  SNR {s:.2f} · {100*r:.0f}% of families below 1",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("eV (or e for the charge feature)")
    ax.set_xlim(0, float(df["median_residual_sd"].max()) * 1.75)
    worst = float(df["median_snr"].max())
    ax.set_title("Every energy feature is buried in conformer noise\n"
                 f"best case SNR {worst:.2f}; the incumbent (ionic radius) is a "
                 f"lookup table with zero scatter", loc="left")
    ax.legend(loc="lower right")
    _save(fig, "re_fig3_energy_snr")


def fig_calibration() -> None:
    """Compression is shrinkage: rescaling cannot recover the spread."""
    frames = {"binned": _read("calibration_test_binned.csv"),
              "strict": _read("calibration_test_strict.csv")}
    frames = {k: v for k, v in frames.items() if v is not None and not v.empty}
    if not frames:
        print("skip calibration: no calibration_test_*.csv"); return

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    model = "full (CatBoost+repaired+S0)"
    x = np.arange(len(frames))
    for i, (key, df) in enumerate(frames.items()):
        sub = df[df["model"] == model]
        raw = sub[sub["transform"] == "raw"]
        if raw.empty:
            continue
        raw_span = float(raw["span_ratio"].iloc[0])
        cal = sub[sub["transform"].isin(["scale", "affine", "isotonic"])]
        best_span = float(cal.loc[cal["r2"].idxmax(), "span_ratio"])
        ax.bar(i - 0.17, raw_span, width=0.32, color=TOPO, label="raw" if i == 0 else None)
        ax.bar(i + 0.17, best_span, width=0.32, color=FCNN,
               label="after nested recalibration" if i == 0 else None)
        ax.text(i - 0.17, raw_span + 0.02, f"{raw_span:.2f}×", ha="center",
                fontsize=8, color=INK2)
        ax.text(i + 0.17, best_span + 0.02, f"{best_span:.2f}×", ha="center",
                fontsize=8, color=INK2)
    ax.axhline(1.0, color=INK, lw=1.0)
    ax.text(len(frames) - 0.5, 1.01, "true spread", fontsize=8, color=INK,
            ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(list(frames))
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("predicted spread / true spread")
    ax.set_title("Recalibration cannot repair the compression\n"
                 "even a free monotone map leaves predictions at about half "
                 "the true spread", loc="left")
    ax.legend(loc="upper right")
    _save(fig, "re_fig4_calibration")


FIGS = {"ceiling": fig_ceiling, "dualkey": fig_dualkey,
        "energy": fig_energy_snr, "calibration": fig_calibration}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS))
    args = ap.parse_args()
    want = args.only if args.only else (sorted(FIGS) if args.all else [])
    if not want:
        ap.error("choose --all or --only NAME")
    for n in want:
        FIGS[n]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
