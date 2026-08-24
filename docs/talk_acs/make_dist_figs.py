#!/usr/bin/env python3
"""Two single-column histograms for the manuscript.

fig_logD.{pdf,png}   log D over the 5,992 unique lanthanide(III) measurements
fig_logSF.{pdf,png}  log SF over the 1,230 adjacent-pair separations scored

Caption text (for the manuscript):
  Figure. (a) Distribution of measured log D for the 5,992 unique
  lanthanide(III) measurements. (b) Distribution of adjacent-pair
  separation factors, log SF = log D(heavier) - log D(lighter), for the
  1,230 pairs measured under identical conditions.

Usage:  PYTHONPATH=$PWD python3 docs/talk_acs/make_dist_figs.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

fonts = {f.name for f in font_manager.fontManager.ttflist}
FAMILY = next((f for f in ("Arial", "Helvetica", "Liberation Sans",
                           "DejaVu Sans") if f in fonts), "sans-serif")

plt.rcParams.update({
    "font.family": FAMILY, "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "savefig.dpi": 600})

GREY = "#5a5a5a"

# ------------------------------------------------------------------ data
m = pd.read_parquet(REPO / "automl/artifacts/matrix/matrix.parquet",
                    columns=["safe_exp_id", "log_D", "composition_key",
                             "lanthanide_index"])
c = pd.read_parquet(REPO / "collaborator_update/dataset.parquet",
                    columns=["safe_exp_id", "geometry_ok"])
logd = m["log_D"].dropna().to_numpy()

ok = m.merge(c, on="safe_exp_id")
ok = ok[ok["geometry_ok"].astype(bool)]
cells = ok.groupby(["composition_key", "lanthanide_index"],
                   as_index=False)["log_D"].mean()
dy = []
for _, blk in cells.groupby("composition_key"):
    blk = blk.sort_values("lanthanide_index")
    idx = blk["lanthanide_index"].to_numpy(int)
    y = blk["log_D"].to_numpy()
    for i in range(len(idx) - 1):
        if idx[i + 1] - idx[i] == 1:
            dy.append(y[i + 1] - y[i])
dy = np.asarray(dy)


def hist(x, bins, xlabel, xlim, name, xticks=None):
    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.20, top=0.95)
    ax.hist(x, bins=bins, color="0.35", edgecolor="white", linewidth=0.3)
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.minorticks_on()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of measurements" if name == "fig_logD"
                  else "Number of pairs")
    ax.text(0.03, 0.95, f"n = {len(x):,}", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color=GREY)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{ext}")
    plt.close(fig)


hist(logd, np.arange(-13, 5.01, 0.5), r"$\log D$", (-13, 5), "fig_logD",
     xticks=np.arange(-12, 5, 3))
hist(dy, np.arange(-2.0, 2.01, 0.1), r"$\log SF$ (adjacent pair)",
     (-2, 2), "fig_logSF", xticks=np.arange(-2, 2.1, 1))
print(f"font: {FAMILY}; log D n={len(logd)}, pairs n={len(dy)}")
