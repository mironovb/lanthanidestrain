#!/usr/bin/env python3
"""Distribution of log D over the unique lanthanide(III) measurements, and of
the adjacent-pair separations that are scored.

Usage:  PYTHONPATH=$PWD python3 docs/talk_acs/make_logd_dist.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

INK, GREY, BLUE, ORANGE = "#111111", "#7d838a", "#2a6db0", "#d55e00"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "figure.facecolor": "white", "savefig.dpi": 400,
    "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False})

m = pd.read_parquet(REPO / "automl/artifacts/matrix/matrix.parquet",
                    columns=["safe_exp_id", "log_D", "composition_key",
                             "lanthanide_index"])
c = pd.read_parquet(REPO / "collaborator_update/dataset.parquet",
                    columns=["safe_exp_id", "geometry_ok"])
logd = m["log_D"].dropna().to_numpy()

ok = m.merge(c, on="safe_exp_id")
ok = ok[ok["geometry_ok"].astype(bool)]
cells = ok.groupby(["composition_key", "lanthanide_index"], as_index=False)["log_D"].mean()
dy = []
for _, blk in cells.groupby("composition_key"):
    blk = blk.sort_values("lanthanide_index")
    idx = blk["lanthanide_index"].to_numpy(int)
    y = blk["log_D"].to_numpy()
    for i in range(len(idx) - 1):
        if idx[i + 1] - idx[i] == 1:
            dy.append(y[i + 1] - y[i])
dy = np.asarray(dy)

fig = plt.figure(figsize=(6.6, 2.7))


def panel(ax, x, color, xlabel, bins, xlim):
    ax.hist(x, bins=bins, color=color, edgecolor="white", linewidth=0.4,
            zorder=3)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    ax.axvline(med, color=INK, lw=1.0, zorder=4)
    ax.axvline(q1, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=4)
    ax.axvline(q3, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=4)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.04, 0.95,
            f"n = {len(x):,}\nmedian {med:+.2f}\n"
            f"IQR {q1:+.2f} to {q3:+.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
            color=INK, linespacing=1.35)


axA = fig.add_axes([0.09, 0.20, 0.40, 0.72])
panel(axA, logd, BLUE, r"$\log D$", np.arange(-13, 5.01, 0.5), (-13, 5))
axA.set_title("A", loc="left", fontsize=11, fontweight="bold")

axB = fig.add_axes([0.58, 0.20, 0.40, 0.72])
panel(axB, dy, ORANGE, r"$\log SF$, adjacent pairs",
      np.arange(-2.0, 2.01, 0.1), (-2.0, 2.0))
axB.axvline(0, color=GREY, lw=0.7, zorder=2)
axB.set_title("B", loc="left", fontsize=11, fontweight="bold")

fig.savefig(HERE / "logd_distribution.png")
print(f"wrote logd_distribution.png  (log D n={len(logd)}, pairs n={len(dy)}, "
      f"logD range {logd.min():.2f}..{logd.max():.2f}, "
      f"dy range {dy.min():.2f}..{dy.max():.2f})")
